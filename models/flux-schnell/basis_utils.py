"""
FLUX.1-schnell specific PCA basis collection.

Registers hooks on FluxTransformerBlock (double) and FluxSingleTransformerBlock
(single) layers to accumulate activation covariances, then runs per-layer
eigendecomposition.

Performance:
  - Small-D layers (D ≤ GPU_H_MAX_D): accumulator lives on GPU, hooks use
    addmm_ / add_ — zero PCIe transfers during the forward pass.
  - Large-D layers (D = 12288): partial XTX stays on GPU inside the hook,
    flushed to CPU after each full forward pass via non_blocking transfers.
    This eliminates GPU stalls caused by synchronous CPU transfers mid-pass.
  - Calibration files are prefetched from NFS with a background thread pool
    so disk I/O overlaps with GPU compute.
  - Inputs are fed in the model's native dtype (bf16) instead of fp32.

Basis keys (match flux-schnell/model_utils.py:assign_online_rotations):
  Double blocks (transformer_blocks.{i}):
    layer.{i}.img_attn          [hidden, hidden]            img Q/K/V input
    layer.{i}.txt_attn          [hidden, hidden]            txt Q/K/V input
    layer.{i}.img_attn.value    [num_heads, head, head]     img V output per-head
    layer.{i}.txt_attn.value    [num_heads, head, head]     txt V output per-head
    layer.{i}.img_ffn           [hidden, hidden]            img FFN up input
    layer.{i}.img_ffn.down      [inter, inter]              img FFN down input
    layer.{i}.txt_ffn           [hidden, hidden]            txt FFN up input
    layer.{i}.txt_ffn.down      [inter, inter]              txt FFN down input

  Single blocks (single_transformer_blocks.{i}):
    single.{i}.attn             [hidden, hidden]            Q/K/V input
    single.{i}.mlp              [hidden, hidden]            MLP up input
    single.{i}.attn_out.value   [num_heads, head, head]     attn-out half of
                                                            split proj_out
                                                            (per-head)
    single.{i}.mlp.down         [inter, inter]              mlp half of split
                                                            proj_out
"""

import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from split_proj_out import split_flux_single_proj_out

# Threshold: accumulators with D ≤ this live on GPU (float32, addmm_).
# Larger accumulators live on CPU (float32) and are flushed asynchronously.
_GPU_H_MAX_D = 4096


def collect_basis(transformer, cache_files: list, cfg: dict) -> dict:
    dims              = cfg["dims"]
    hidden            = dims["hidden"]
    head_dim          = dims["head"]
    num_heads         = dims["num_heads"]
    inter             = dims["intermediate"]
    num_double_layers = dims["num_double_layers"]
    num_single_layers = dims["num_single_layers"]

    device = next(transformer.parameters()).device

    # Match the model's compute dtype so no silent fp32 upcast during forward.
    model_dtype = next(transformer.parameters()).dtype

    # Match SVDQuant's handling of the single-block proj_out: split it into
    # two Linear branches so we can collect per-half covariances independently.
    n_split = split_flux_single_proj_out(transformer)
    print(f"Split {n_split} single-block proj_out layers into attn/mlp halves.")

    # -----------------------------------------------------------------------
    # Allocate covariance accumulators.
    #
    # Small-D (D = hidden = 3072): GPU float32. No PCIe traffic.
    # Per-head (D = head_dim = 128): GPU float32, shape [H, d, d].
    # Large-D (D = inter = 12288): CPU float32. Flushed after each pass.
    # -----------------------------------------------------------------------
    def _gpu(shape):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def _cpu(shape):
        return torch.zeros(shape, dtype=torch.float32)

    on_gpu_hidden = hidden <= _GPU_H_MAX_D
    on_gpu_inter  = inter  <= _GPU_H_MAX_D

    # Double-block accumulators
    H_img_attn = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_double_layers)]
    H_txt_attn = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_double_layers)]
    H_img_val  = [_gpu((num_heads, head_dim, head_dim))               for _ in range(num_double_layers)]
    H_txt_val  = [_gpu((num_heads, head_dim, head_dim))               for _ in range(num_double_layers)]
    H_img_ffn  = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_double_layers)]
    H_txt_ffn  = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_double_layers)]
    H_img_down = [(_gpu if on_gpu_inter  else _cpu)((inter,  inter))  for _ in range(num_double_layers)]
    H_txt_down = [(_gpu if on_gpu_inter  else _cpu)((inter,  inter))  for _ in range(num_double_layers)]

    # Single-block accumulators
    H_sattn    = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_single_layers)]
    H_smlp     = [(_gpu if on_gpu_hidden else _cpu)((hidden, hidden)) for _ in range(num_single_layers)]
    H_sattnout = [_gpu((num_heads, head_dim, head_dim))               for _ in range(num_single_layers)]
    H_smlpdown = [(_gpu if on_gpu_inter  else _cpu)((inter,  inter))  for _ in range(num_single_layers)]

    cnt_img_attn = [0] * num_double_layers
    cnt_txt_attn = [0] * num_double_layers
    cnt_img_val  = [0] * num_double_layers
    cnt_txt_val  = [0] * num_double_layers
    cnt_img_ffn  = [0] * num_double_layers
    cnt_txt_ffn  = [0] * num_double_layers
    cnt_img_down = [0] * num_double_layers
    cnt_txt_down = [0] * num_double_layers
    cnt_sattn    = [0] * num_single_layers
    cnt_smlp     = [0] * num_single_layers
    cnt_sattnout = [0] * num_single_layers
    cnt_smlpdown = [0] * num_single_layers

    # -----------------------------------------------------------------------
    # Hook factories.
    #
    # GPU hooks: addmm_ / add_ — fully in-place, no allocation, no PCIe.
    # CPU-deferred hooks: compute XTX on GPU, append to _pending list so the
    #   forward pass is never stalled. _flush() drains the list after each pass.
    # -----------------------------------------------------------------------
    _pending = []  # list of (cpu_accumulator, gpu_partial)

    def _flush():
        """Transfer pending GPU partials to CPU accumulators without stalling."""
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
        # Per-head cov on GPU (head_dim is small).
        def hook(module, args, output):
            if output.dim() == 2:
                output = output.unsqueeze(0)
            B, T, _ = output.shape
            v = output.reshape(B * T, num_heads, head_dim).detach().float()
            H_list[i].add_(torch.einsum('nhd,nhe->hde', v, v))
            cnt_list[i] += B * T
        return hook

    def _attn_out_input_hook(i, H_list, cnt_list):
        """Per-head input cov for proj_out.linears[0]."""
        def hook(module, args, output):
            x = args[0] if isinstance(args, tuple) else args
            if x.dim() == 2:
                x = x.unsqueeze(0)
            B, T, D = x.shape
            v = x.reshape(B * T, num_heads, head_dim).detach().float()
            H_list[i].add_(torch.einsum('nhd,nhe->hde', v, v))
            cnt_list[i] += B * T
        return hook

    # -----------------------------------------------------------------------
    # Register hooks
    # -----------------------------------------------------------------------
    hooks = []

    for i, block in enumerate(transformer.transformer_blocks):
        hooks.append(block.attn.to_q.register_forward_hook(_input_hook(i, H_img_attn, cnt_img_attn)))
        hooks.append(block.attn.to_v.register_forward_hook(_value_hook(i, H_img_val,  cnt_img_val)))
        hooks.append(block.ff.net[0].proj.register_forward_hook(_input_hook(i, H_img_ffn,  cnt_img_ffn)))
        hooks.append(block.ff.net[2].register_forward_hook(      _input_hook(i, H_img_down, cnt_img_down)))
        hooks.append(block.attn.add_q_proj.register_forward_hook(_input_hook(i, H_txt_attn, cnt_txt_attn)))
        hooks.append(block.attn.add_v_proj.register_forward_hook(_value_hook(i, H_txt_val,  cnt_txt_val)))
        hooks.append(block.ff_context.net[0].proj.register_forward_hook(_input_hook(i, H_txt_ffn,  cnt_txt_ffn)))
        hooks.append(block.ff_context.net[2].register_forward_hook(      _input_hook(i, H_txt_down, cnt_txt_down)))

    for i, block in enumerate(transformer.single_transformer_blocks):
        hooks.append(block.attn.to_q.register_forward_hook(_input_hook(i, H_sattn, cnt_sattn)))
        hooks.append(block.proj_mlp.register_forward_hook(  _input_hook(i, H_smlp,  cnt_smlp)))
        hooks.append(block.proj_out.linears[0].register_forward_hook(
            _attn_out_input_hook(i, H_sattnout, cnt_sattnout)))
        hooks.append(block.proj_out.linears[1].register_forward_hook(
            _input_hook(i, H_smlpdown, cnt_smlpdown)))

    n_gpu = sum(1 for h in [H_img_attn[0], H_img_down[0]] if h.device.type == "cuda")
    print(f"Registered {len(hooks)} hooks across "
          f"{num_double_layers} double + {num_single_layers} single blocks. "
          f"Hidden accumulators on {'GPU' if on_gpu_hidden else 'CPU'}, "
          f"inter accumulators on {'GPU' if on_gpu_inter else 'CPU'}.")

    # -----------------------------------------------------------------------
    # Forward pass with background file prefetch
    # -----------------------------------------------------------------------
    def _load_file(f):
        return torch.load(f, map_location="cpu", weights_only=False)

    PREFETCH = 3

    with torch.no_grad(), ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        io_queue = deque()
        file_iter = iter(cache_files)

        # Pre-fill the prefetch window
        for _ in range(PREFETCH):
            try:
                io_queue.append(pool.submit(_load_file, next(file_iter)))
            except StopIteration:
                break

        for _ in tqdm(range(len(cache_files)), desc="Replaying calibration cache"):
            # Submit next I/O before waiting for current — keeps NFS busy
            try:
                io_queue.append(pool.submit(_load_file, next(file_iter)))
            except StopIteration:
                pass

            data = io_queue.popleft().result()

            input_args = [a.to(device=device, dtype=model_dtype) if a.is_floating_point()
                          else a.to(device) for a in data["input_args"]]
            input_kwargs = {}
            for k, v in data["input_kwargs"].items():
                if isinstance(v, torch.Tensor):
                    input_kwargs[k] = (v.to(device=device, dtype=model_dtype)
                                       if v.is_floating_point() else v.to(device))
                elif isinstance(v, dict):
                    input_kwargs[k] = {kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                                       for kk, vv in v.items() if vv is not None}
                else:
                    input_kwargs[k] = v

            try:
                transformer(*input_args, **input_kwargs)
            except Exception as e:
                print(f"Warning: {e}")

            # Drain pending large-D XTX transfers without stalling the forward pass
            _flush()

    for h in hooks:
        h.remove()

    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Normalise accumulators and compute PCA eigendecompositions.
    # GPU tensors are moved to CPU and cast to float64 only here.
    # -----------------------------------------------------------------------
    def _to_f64_cpu(H, cnt):
        """Normalise by count, move to CPU float64."""
        if cnt == 0:
            return H.float().cpu().double()
        if H.device.type == "cuda":
            return (H / cnt).cpu().double()
        return (H / cnt).double()

    print("Computing PCA eigendecompositions (CPU, float64)...")
    basis_dict = {}

    for i in tqdm(range(num_double_layers), desc="double blocks"):
        basis_dict[f"layer.{i}.img_attn"]       = _eigh(_to_f64_cpu(H_img_attn[i], cnt_img_attn[i]))
        basis_dict[f"layer.{i}.txt_attn"]       = _eigh(_to_f64_cpu(H_txt_attn[i], cnt_txt_attn[i]))
        basis_dict[f"layer.{i}.img_attn.value"] = _eigh_per_head(_to_f64_cpu(H_img_val[i], cnt_img_val[i]), num_heads)
        basis_dict[f"layer.{i}.txt_attn.value"] = _eigh_per_head(_to_f64_cpu(H_txt_val[i], cnt_txt_val[i]), num_heads)
        basis_dict[f"layer.{i}.img_ffn"]        = _eigh(_to_f64_cpu(H_img_ffn[i], cnt_img_ffn[i]))
        basis_dict[f"layer.{i}.txt_ffn"]        = _eigh(_to_f64_cpu(H_txt_ffn[i], cnt_txt_ffn[i]))
        basis_dict[f"layer.{i}.img_ffn.down"]   = _eigh(_to_f64_cpu(H_img_down[i], cnt_img_down[i]))
        basis_dict[f"layer.{i}.txt_ffn.down"]   = _eigh(_to_f64_cpu(H_txt_down[i], cnt_txt_down[i]))

    for i in tqdm(range(num_single_layers), desc="single blocks"):
        basis_dict[f"single.{i}.attn"]           = _eigh(_to_f64_cpu(H_sattn[i],    cnt_sattn[i]))
        basis_dict[f"single.{i}.mlp"]            = _eigh(_to_f64_cpu(H_smlp[i],     cnt_smlp[i]))
        basis_dict[f"single.{i}.attn_out.value"] = _eigh_per_head(_to_f64_cpu(H_sattnout[i], cnt_sattnout[i]), num_heads)
        basis_dict[f"single.{i}.mlp.down"]       = _eigh(_to_f64_cpu(H_smlpdown[i], cnt_smlpdown[i]))

    return basis_dict


def _eigh(H: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    """Eigendecomposition of a covariance matrix. Returns eigenvectors (ascending).

    H is accumulated in fp64 for numerical stability of the eigendecomposition,
    but the returned evec is cast to fp32.
    """
    H = H + damping * H.diagonal().mean() * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    _, evec = torch.linalg.eigh(H)
    return evec.float()


def _eigh_per_head(H: torch.Tensor, num_heads: int, damping: float = 0.01) -> torch.Tensor:
    """Per-head eigendecomposition. H: [num_heads, d, d]. Returns [num_heads, d, d]."""
    evec_all = torch.zeros_like(H)
    for h in range(num_heads):
        evec_all[h] = _eigh(H[h], damping)
    return evec_all
