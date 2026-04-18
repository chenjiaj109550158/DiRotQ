"""
PixArt-Sigma specific PCA basis collection.
Registers hooks on PixArt transformer blocks to accumulate activation covariances.

Performance:
  - Small-D layers (D ≤ _GPU_H_MAX_D): accumulator lives on GPU (float32),
    hooks use addmm_ / add_ — zero PCIe transfers during the forward pass.
  - Large-D layers (D = intermediate = 4608): partial XTX stays on GPU inside
    the hook, flushed to CPU after each full forward pass (non_blocking).
  - Calibration files are prefetched in a background thread pool.
  - Inputs are fed in the model's native dtype instead of fp32.
"""

import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

_GPU_H_MAX_D = 4096


def collect_basis(transformer, cache_files: list, cfg: dict, batch_size: int = 8) -> dict:
    """
    Replay calibration cache through the PixArt transformer and compute PCA basis.

    Hooks are registered on the following layers per block:
      - attn1.to_q    (self-attn input, shared with K/V)
      - attn1.to_v    (self-attn value output, per-head)
      - attn2.to_q    (cross-attn Q input)
      - attn2.to_v    (cross-attn value output, per-head)
      - ff.net[0].proj (FFN up-proj input)
      - ff.net[2]     (FFN down-proj input)

    Returns:
        basis_dict: {
            "layer.{i}.self_attn":          [hidden, hidden] float32 eigenvectors,
            "layer.{i}.self_attn.value":    [num_heads, head_dim, head_dim] float32,
            "layer.{i}.cross_attn_q":       [hidden, hidden] float32,
            "layer.{i}.cross_attn_q.value": [num_heads, head_dim, head_dim] float32,
            "layer.{i}.ffn":                [hidden, hidden] float32,
            "layer.{i}.ffn.down_proj":      [intermediate, intermediate] float32,
        }
    """
    dims       = cfg["dims"]
    hidden     = dims["hidden"]
    head_dim   = dims["head"]
    num_heads  = dims["num_heads"]
    inter      = dims["intermediate"]
    num_layers = dims["num_layers"]

    device     = next(transformer.parameters()).device
    model_dtype = next(transformer.parameters()).dtype

    on_gpu_hidden = hidden <= _GPU_H_MAX_D
    on_gpu_inter  = inter  <= _GPU_H_MAX_D

    def _gpu(shape):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def _cpu(shape):
        return torch.zeros(shape, dtype=torch.float32)

    H_sa     = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden))              for _ in range(num_layers)]
    H_sa_val = [_gpu((num_heads, head_dim, head_dim))                            for _ in range(num_layers)]
    H_ca_q   = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden))              for _ in range(num_layers)]
    H_ca_val = [_gpu((num_heads, head_dim, head_dim))                            for _ in range(num_layers)]
    H_ffn    = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden))              for _ in range(num_layers)]
    H_down   = [(_gpu if on_gpu_inter  else _cpu)((inter,  inter))               for _ in range(num_layers)]

    cnt_sa     = [0] * num_layers
    cnt_sa_val = [0] * num_layers
    cnt_ca_q   = [0] * num_layers
    cnt_ca_val = [0] * num_layers
    cnt_ffn    = [0] * num_layers
    cnt_down   = [0] * num_layers

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
            H_list[i].add_(torch.einsum('nhd,nhe->hde', v, v))
            cnt_list[i] += B * T
        return hook

    hooks = []
    for i, block in enumerate(transformer.transformer_blocks):
        hooks.append(block.attn1.to_q.register_forward_hook(_input_hook(i, H_sa, cnt_sa)))
        hooks.append(block.attn1.to_v.register_forward_hook(_value_hook(i, H_sa_val, cnt_sa_val)))
        hooks.append(block.attn2.to_q.register_forward_hook(_input_hook(i, H_ca_q, cnt_ca_q)))
        hooks.append(block.attn2.to_v.register_forward_hook(_value_hook(i, H_ca_val, cnt_ca_val)))
        hooks.append(block.ff.net[0].proj.register_forward_hook(_input_hook(i, H_ffn, cnt_ffn)))
        hooks.append(block.ff.net[2].register_forward_hook(_input_hook(i, H_down, cnt_down)))

    print(f"Registered {len(hooks)} hooks across {num_layers} blocks. "
          f"Hidden accumulators on {'GPU' if on_gpu_hidden else 'CPU'}, "
          f"inter accumulators on {'GPU' if on_gpu_inter else 'CPU'}.")

    def _load_file(f):
        return torch.load(f, map_location="cpu", weights_only=False)

    def _run_batch(batch):
        """Stack a list of raw data dicts and run one forward pass."""
        # Stack leading dim-0 tensors; share per-sample-agnostic kwargs from first item.
        args0 = batch[0]["input_args"]
        stacked_args = [
            torch.cat([d["input_args"][i] for d in batch], dim=0)
                .to(device=device, dtype=model_dtype if batch[0]["input_args"][i].is_floating_point() else batch[0]["input_args"][i].dtype)
            for i in range(len(args0))
        ]
        stacked_kwargs = {}
        for k, v in batch[0]["input_kwargs"].items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] == 1:  # has a batch dim → cat
                    cat = torch.cat([d["input_kwargs"][k] for d in batch], dim=0)
                    stacked_kwargs[k] = (cat.to(device=device, dtype=model_dtype)
                                         if cat.is_floating_point() else cat.to(device))
                else:  # shared (e.g. position ids) → use first
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

    PREFETCH = batch_size + 2  # keep NFS pipeline full

    num_batches = (len(cache_files) + batch_size - 1) // batch_size
    print(f"Running {len(cache_files)} files in {num_batches} batches of {batch_size}.")

    with torch.no_grad(), ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        io_queue = deque()
        file_iter = iter(cache_files)

        for _ in range(PREFETCH):
            try:
                io_queue.append(pool.submit(_load_file, next(file_iter)))
            except StopIteration:
                break

        for _ in tqdm(range(num_batches), desc="Replaying calibration cache"):
            batch = []
            while len(batch) < batch_size and io_queue:
                # Prefetch next before blocking on current
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
        basis_dict[f"layer.{i}.self_attn"]          = _eigh(_to_f64_cpu(H_sa[i],     cnt_sa[i]))
        basis_dict[f"layer.{i}.self_attn.value"]    = _eigh_per_head(_to_f64_cpu(H_sa_val[i], cnt_sa_val[i]), num_heads)
        basis_dict[f"layer.{i}.cross_attn_q"]       = _eigh(_to_f64_cpu(H_ca_q[i],   cnt_ca_q[i]))
        basis_dict[f"layer.{i}.cross_attn_q.value"] = _eigh_per_head(_to_f64_cpu(H_ca_val[i], cnt_ca_val[i]), num_heads)
        basis_dict[f"layer.{i}.ffn"]                = _eigh(_to_f64_cpu(H_ffn[i],    cnt_ffn[i]))
        basis_dict[f"layer.{i}.ffn.down_proj"]      = _eigh(_to_f64_cpu(H_down[i],   cnt_down[i]))

    return basis_dict


def _eigh(H: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    """Eigendecomposition of a covariance matrix. Returns eigenvectors (ascending)."""
    H = H + damping * H.diagonal().mean() * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    _, evec = torch.linalg.eigh(H)
    return evec.float()


def _eigh_per_head(H: torch.Tensor, num_heads: int, damping: float = 0.01) -> torch.Tensor:
    """Per-head eigendecomposition. H: [num_heads, d, d]. Returns [num_heads, d, d]."""
    evec_all = torch.zeros_like(H)
    for h in range(num_heads):
        evec_all[h] = _eigh(H[h], damping)
    return evec_all
