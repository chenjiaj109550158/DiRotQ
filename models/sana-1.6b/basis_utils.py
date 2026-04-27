"""
Sana-1.6B PCA basis collection.

Hooks registered per block:
  - attn1.to_q  (self-attn input, shared for K/V)      -> H_sa     [hidden, hidden]
  - attn1.to_v  (self-attn value output, per-head)     -> H_sa_val [num_heads, head_dim, head_dim]
  - attn2.to_q  (cross-attn image-side query input)    -> H_ca     [hidden, hidden]
  - attn2.to_v  (cross-attn value output, per-head)    -> H_ca_val [num_heads, head_dim, head_dim]

Performance:
  - Small-D accumulators (D <= 4096): live on GPU as float32, addmm_ in-place.
  - Larger-D: partial XTX accumulated on GPU, flushed to CPU after each forward pass.
"""

import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

_GPU_H_MAX_D = 4096


def collect_basis(transformer, cache_files: list, cfg: dict, batch_size: int = 8) -> dict:
    """
    Replay calibration cache through the Sana transformer and compute PCA basis.

    Returns:
        basis_dict: {
            "layer.{i}.self_attn":         [hidden, hidden]                float32 eigenvectors,
            "layer.{i}.self_attn.value":   [num_heads, head_dim, head_dim] float32,
            "layer.{i}.cross_attn":        [hidden, hidden]                float32 eigenvectors,
            "layer.{i}.cross_attn.value":  [num_heads, head_dim, head_dim] float32,
        }
    """
    dims       = cfg["dims"]
    hidden     = dims["hidden"]
    head_dim   = dims["head"]
    num_heads  = dims["num_heads"]
    num_layers = dims["num_layers"]

    device      = next(transformer.parameters()).device
    model_dtype = next(transformer.parameters()).dtype

    on_gpu_hidden = hidden <= _GPU_H_MAX_D

    def _gpu(shape):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def _cpu(shape):
        return torch.zeros(shape, dtype=torch.float32)

    H_sa     = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_layers)]
    H_sa_val = [_gpu((num_heads, head_dim, head_dim))               for _ in range(num_layers)]
    H_ca     = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_layers)]
    H_ca_val = [_gpu((num_heads, head_dim, head_dim))               for _ in range(num_layers)]

    cnt_sa     = [0] * num_layers
    cnt_sa_val = [0] * num_layers
    cnt_ca     = [0] * num_layers
    cnt_ca_val = [0] * num_layers

    _pending = []  # (cpu_accumulator, gpu_partial)

    def _flush():
        if not _pending:
            return
        torch.cuda.synchronize()
        for H_cpu, p_gpu in _pending:
            H_cpu.add_(p_gpu.cpu())
        _pending.clear()

    def _input_hook(i, H_list, cnt_list):
        on_gpu = H_list[i].device.type == "cuda"
        def hook(module, args, output):
            x = args[0] if isinstance(args, tuple) else args
            if x.dim() == 2:
                x = x.unsqueeze(0)
            B, T, D = x.shape
            cnt_list[i] += B * T
            x_flat = x.reshape(-1, D).detach().float()
            if on_gpu:
                H_list[i].addmm_(x_flat.T, x_flat)
            else:
                _pending.append((H_list[i], x_flat.T @ x_flat))
        return hook

    def _value_hook(i, H_list, cnt_list):
        def hook(module, args, output):
            if output.dim() == 2:
                output = output.unsqueeze(0)
            B, T, _ = output.shape
            v = output.reshape(B * T, num_heads, head_dim).detach().float()
            H_list[i].add_(torch.einsum("nhd,nhe->hde", v, v))
            cnt_list[i] += B * T
        return hook

    hooks = []
    for i, block in enumerate(transformer.transformer_blocks):
        hooks.append(block.attn1.to_q.register_forward_hook(_input_hook(i, H_sa,     cnt_sa)))
        hooks.append(block.attn1.to_v.register_forward_hook(_value_hook(i, H_sa_val, cnt_sa_val)))
        hooks.append(block.attn2.to_q.register_forward_hook(_input_hook(i, H_ca,     cnt_ca)))
        hooks.append(block.attn2.to_v.register_forward_hook(_value_hook(i, H_ca_val, cnt_ca_val)))

    print(f"Registered {len(hooks)} hooks across {num_layers} blocks "
          f"(attn1 + attn2). Hidden accumulators on {'GPU' if on_gpu_hidden else 'CPU'}.")

    def _load_file(f):
        return torch.load(f, map_location="cpu", weights_only=False)

    def _run_batch(batch):
        args0 = batch[0]["input_args"]
        stacked_args = [
            torch.cat([d["input_args"][j] for d in batch], dim=0)
                .to(device=device,
                    dtype=model_dtype if batch[0]["input_args"][j].is_floating_point()
                          else batch[0]["input_args"][j].dtype)
            for j in range(len(args0))
        ]
        stacked_kwargs = {}
        for k, v in batch[0]["input_kwargs"].items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] == 1:
                    cat = torch.cat([d["input_kwargs"][k] for d in batch], dim=0)
                    stacked_kwargs[k] = (cat.to(device=device, dtype=model_dtype)
                                         if cat.is_floating_point() else cat.to(device))
                else:
                    stacked_kwargs[k] = v.to(device)
            elif isinstance(v, dict):
                stacked_kwargs[k] = {kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                                     for kk, vv in v.items() if vv is not None}
            else:
                stacked_kwargs[k] = v

        try:
            transformer(*stacked_args, **stacked_kwargs)
        except Exception as e:
            print(f"Warning: {e}")
        _flush()

    PREFETCH   = batch_size + 2
    num_batches = (len(cache_files) + batch_size - 1) // batch_size
    print(f"Running {len(cache_files)} files in {num_batches} batches of {batch_size}.")

    with torch.no_grad(), ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        io_queue  = deque()
        file_iter = iter(cache_files)

        for _ in range(PREFETCH):
            try:
                io_queue.append(pool.submit(_load_file, next(file_iter)))
            except StopIteration:
                break

        for _ in tqdm(range(num_batches), desc="Replaying calibration cache"):
            batch = []
            while len(batch) < batch_size and io_queue:
                try:
                    io_queue.append(pool.submit(_load_file, next(file_iter)))
                except StopIteration:
                    pass
                batch.append(io_queue.popleft().result())
            if batch:
                _run_batch(batch)

    for h in hooks:
        h.remove()

    def _to_f64_cpu(H, cnt):
        if cnt == 0:
            return H.float().cpu().double()
        if H.device.type == "cuda":
            return (H / cnt).cpu().double()
        return (H / cnt).double()

    print("Computing PCA eigendecompositions...")
    basis_dict = {}
    for i in tqdm(range(num_layers)):
        basis_dict[f"layer.{i}.self_attn"]        = _eigh(_to_f64_cpu(H_sa[i],     cnt_sa[i]))
        basis_dict[f"layer.{i}.self_attn.value"]  = _eigh_per_head(
            _to_f64_cpu(H_sa_val[i], cnt_sa_val[i]), num_heads
        )
        basis_dict[f"layer.{i}.cross_attn"]       = _eigh(_to_f64_cpu(H_ca[i],     cnt_ca[i]))
        basis_dict[f"layer.{i}.cross_attn.value"] = _eigh_per_head(
            _to_f64_cpu(H_ca_val[i], cnt_ca_val[i]), num_heads
        )

    return basis_dict


def _eigh(H: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    H = H + damping * H.diagonal().mean() * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    _, evec = torch.linalg.eigh(H)
    return evec.float()


def _eigh_per_head(H: torch.Tensor, num_heads: int, damping: float = 0.01) -> torch.Tensor:
    evec_all = torch.zeros_like(H)
    for h in range(num_heads):
        evec_all[h] = _eigh(H[h], damping)
    return evec_all
