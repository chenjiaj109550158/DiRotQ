"""
GPTQ weight quantization utilities for DiRotQ.

Two-phase workflow:
  1. collect_hessians()  — run calibration samples through the model with
                           forward hooks on ActQuantWrapper to accumulate per-layer
                           Hessian matrices H = 2/n * X^T X.
  2. gptq_quantize_weights() — apply the GPTQ algorithm to each layer using its H.

Correctness note: hooks capture x (pre-rotation input to ActQuantWrapper).
Since the rotation U is orthogonal:
  H_{x_unrot} ≈ U H_{x_rot} U^T = U (U^T H_x U) U^T = H_x
so using H_x gives the same column-importance ordering as H_{x_unrot}.
"""

import gc
import math
from pathlib import Path

import torch
from tqdm import tqdm

from .quant_utils import ActQuantWrapper, find_qlayers, round_to_nf4_codebook, NF4_MAX


# ---------------------------------------------------------------------------
# Phase 1: Hessian collection
# ---------------------------------------------------------------------------

def collect_hessians(transformer, calib_dir, device, num_calib_files=5120, batch_size=8):
    """
    Run qdiff calibration files through the transformer and accumulate
    per-layer Hessian matrices H = 2/n * X^T X.

    Hooks are placed on ActQuantWrapper modules to capture the input x
    (before rotation / activation quantization).

    Args:
        transformer:     Model with ActQuantWrapper layers, already on `device`.
        calib_dir:       Path to the qdiff cache directory (.pt files).
        device:          Compute device for forward passes.
        num_calib_files: How many cache files to use (default: all 5120).
        batch_size:      Number of calibration samples per forward pass.

    Returns:
        dict[layer_name -> torch.Tensor([in_features, in_features], float32, cpu)]
    """
    qlayers = find_qlayers(transformer, layers=[ActQuantWrapper])

    # H accumulators and token counters (kept on CPU to save GPU memory)
    layer_H = {name: torch.zeros(
        ql.module.in_features, ql.module.in_features, dtype=torch.float32
    ) for name, ql in qlayers.items()}
    layer_n = {name: 0 for name in qlayers}

    # Register hooks on ActQuantWrapper to capture input x
    hooks = []
    for name, ql in qlayers.items():
        in_features = ql.module.in_features

        def _make_hook(n, d):
            def _hook(module, inp, out):
                x = inp[0].detach().float().reshape(-1, d)  # [B*tokens, in_features]
                layer_n[n] += x.shape[0]
                layer_H[n] += (x.cpu().T @ x.cpu())
            return _hook

        hooks.append(ql.register_forward_hook(_make_hook(name, in_features)))

    # Load and batch calibration files
    calib_path = Path(calib_dir)
    calib_files = sorted(calib_path.glob("*.pt"))[:num_calib_files]
    print(f"Collecting Hessians from {len(calib_files)} calibration files "
          f"(batch_size={batch_size})...")

    transformer.eval()
    batch_args = []
    batch_kwargs_lists = {}  # key -> list of tensors

    def _run_batch():
        if not batch_args:
            return
        # Stack along batch dim (dim 0)
        latents = torch.cat([a[0] for a in batch_args], dim=0).to(device)
        stacked_kwargs = {}
        for k in batch_kwargs_lists:
            vals = batch_kwargs_lists[k]
            if isinstance(vals[0], torch.Tensor):
                stacked_kwargs[k] = torch.cat(vals, dim=0).to(device)
            else:
                stacked_kwargs[k] = vals[0]  # scalar/non-tensor: use first
        with torch.no_grad():
            transformer(latents, **stacked_kwargs)
        batch_args.clear()
        for k in batch_kwargs_lists:
            batch_kwargs_lists[k].clear()

    for f in tqdm(calib_files, desc="Calibration forward passes"):
        data = torch.load(f, map_location="cpu", weights_only=False)
        batch_args.append([a.half() for a in data["input_args"]])
        for k, v in data["input_kwargs"].items():
            if k not in batch_kwargs_lists:
                batch_kwargs_lists[k] = []
            if isinstance(v, torch.Tensor):
                batch_kwargs_lists[k].append(v.half())
            else:
                batch_kwargs_lists[k].append(v)
        if len(batch_args) >= batch_size:
            _run_batch()

    _run_batch()  # flush remaining

    for h in hooks:
        h.remove()

    # Normalize: H = 2/n * Σ X^T X
    for name in layer_H:
        n = layer_n[name]
        if n > 0:
            layer_H[name] = (2.0 / n) * layer_H[name]
        else:
            print(f"WARNING: no calibration data collected for layer {name}")

    gc.collect()
    return layer_H


# ---------------------------------------------------------------------------
# Phase 2: GPTQ weight quantization
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gptq_quantize_layer(W, H, bits, groupsize, sym,
                          damp_pct, block_size, num_inv_tries, device,
                          nvfp4=False):
    """
    Apply GPTQ to a single weight matrix.

    Args:
        W:    [out_features, in_features] float32 weight.
        H:    [in_features, in_features] float32 Hessian (2/n * X^T X).
        bits, groupsize, sym: quantization config.
        damp_pct: Hessian diagonal damping fraction.
        block_size: GPTQ column-block size.
        num_inv_tries: max Cholesky retry attempts.
        device: compute device.
        nvfp4: if True, use NF4 codebook rounding instead of INT4.

    Returns:
        W_q: [out_features, in_features] float32 dequantized weight,
             or None if Hessian inversion failed.
    """
    W = W.clone().float().to(device)
    H = H.clone().float().to(device)
    out_features, in_features = W.shape

    if nvfp4:
        sym = True  # NF4 is always symmetric

    maxq = (2 ** (bits - 1) - 1) if sym else (2 ** bits - 1)

    # Pre-compute per-group scales from the original (unpermuted) W.
    # These scales are kept fixed throughout the GPTQ loop.
    if groupsize > 0 and in_features % groupsize == 0:
        W_g = W.reshape(out_features, in_features // groupsize, groupsize)
        if nvfp4:
            scales = W_g.abs().amax(dim=-1).clamp(min=1e-5) / NF4_MAX
            zeros  = torch.zeros_like(scales)
        elif sym:
            scales = W_g.abs().amax(dim=-1).clamp(min=1e-5) / maxq  # [out, n_groups]
            zeros  = torch.zeros_like(scales)
        else:
            wmin   = torch.minimum(W_g.amin(dim=-1), torch.zeros(out_features, in_features // groupsize, device=device))
            wmax   = torch.maximum(W_g.amax(dim=-1), torch.zeros(out_features, in_features // groupsize, device=device))
            scales = (wmax - wmin).clamp(min=1e-5) / maxq
            zeros  = torch.round(-wmin / scales)
    else:
        # Per-channel fallback (rarely hit; guard)
        if nvfp4:
            scales = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / NF4_MAX
            zeros  = torch.zeros_like(scales)
        elif sym:
            scales = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / maxq
            zeros  = torch.zeros_like(scales)
        else:
            wmin   = torch.minimum(W.amin(dim=1, keepdim=True), torch.zeros(out_features, 1, device=device))
            wmax   = torch.maximum(W.amax(dim=1, keepdim=True), torch.zeros(out_features, 1, device=device))
            scales = (wmax - wmin).clamp(min=1e-5) / maxq
            zeros  = torch.round(-wmin / scales)
        groupsize = in_features  # treat as single group

    def _quant_col(w_col, orig_col_idx):
        """Quantize and dequantize one column vector [out_features]."""
        g = orig_col_idx // groupsize
        s = scales[:, g]  # [out_features]
        z = zeros[:, g]
        if nvfp4:
            w_norm = w_col / s
            w_q = round_to_nf4_codebook(w_norm)
            return w_q * s
        elif sym:
            q = torch.clamp(torch.round(w_col / s), -(maxq + 1), maxq)
            return q * s
        else:
            q = torch.clamp(torch.round(w_col / s) + z, 0, maxq)
            return s * (q - z)

    # Handle dead (zero-diagonal) columns
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W_proc = W.clone()
    W_proc[:, dead] = 0.0

    # Permute columns by descending Hessian diagonal (importance)
    importance = torch.diag(H)
    perm     = torch.argsort(importance, descending=True)
    inv_perm = torch.argsort(perm)
    W_proc = W_proc[:, perm]
    H      = H[perm][:, perm]

    # Dampen diagonal
    H_diag      = H.diagonal()
    damp_val    = damp_pct * H_diag.mean()
    H_diag     += damp_val

    # Cholesky inversion (with retry on numerical failure)
    H_inv = None
    for _ in range(num_inv_tries):
        try:
            L     = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv = torch.linalg.cholesky(H_inv, upper=True)
            break
        except RuntimeError:
            H_diag += (damp_pct * 0.1) * H_diag.mean()

    if H_inv is None:
        return None

    # GPTQ column-wise quantization loop
    W_q = torch.zeros_like(W_proc)

    for c_start in range(0, in_features, block_size):
        c_end    = min(c_start + block_size, in_features)
        W_block  = W_proc[:, c_start:c_end].clone()
        H_block  = H_inv[c_start:c_end, c_start:c_end]
        Err      = torch.zeros_like(W_block)

        for _c in range(c_end - c_start):
            c_abs    = c_start + _c
            w_col    = W_block[:, _c]                  # [out_features]
            d        = H_block[_c, _c]                 # scalar
            orig_col = perm[c_abs].item()              # original column index

            w_q_col = _quant_col(w_col, orig_col)
            W_q[:, c_abs] = w_q_col

            err_col = (w_col - w_q_col) / d           # propagation error
            Err[:, _c] = err_col
            # Greedy error compensation for remaining columns in this block
            W_block[:, _c + 1:] -= (
                err_col.unsqueeze(1) * H_block[_c, _c + 1:].unsqueeze(0)
            )

        # Propagate error to all columns after this block
        W_proc[:, c_end:] -= Err @ H_inv[c_start:c_end, c_end:]

    # Un-permute columns back to original order
    W_q = W_q[:, inv_perm]
    return W_q


def gptq_quantize_weights(model, hessians, bits=4, groupsize=64, sym=True,
                           skip_names=None,
                           damp_pct=0.01, block_size=128, num_inv_tries=250,
                           device="cuda", nvfp4=False):
    """
    Apply GPTQ weight quantization to all ActQuantWrapper layers.

    Falls back to RTN for any layer whose Hessian is missing or whose
    Cholesky inversion fails.

    Args:
        model:     Module containing ActQuantWrapper layers.
        hessians:  dict[layer_name -> H tensor] from collect_hessians().
        bits, groupsize, sym: quantization config.
        skip_names: list of name substrings to skip.
        damp_pct, block_size, num_inv_tries: GPTQ hyperparameters.
        device:    Compute device for GPTQ arithmetic.
        nvfp4:     If True, use NF4 codebook rounding instead of INT4.
    """
    if skip_names is None:
        skip_names = []

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    n_gptq = 0
    n_rtn  = 0

    for name, qlayer in tqdm(qlayers.items(), desc="GPTQ weight quantization"):
        if any(skip in name for skip in skip_names):
            continue

        W = qlayer.module.weight.data
        orig_dtype = W.dtype
        W_fp32 = W.float()

        H = hessians.get(name)
        W_q = None
        if H is not None:
            W_q = _gptq_quantize_layer(
                W_fp32, H, bits, groupsize, sym,
                damp_pct, block_size, num_inv_tries, device,
                nvfp4=nvfp4,
            )

        if W_q is not None:
            qlayer.module.weight.data = W_q.to(orig_dtype)
            n_gptq += 1
        else:
            # Fallback: RTN
            if H is None:
                print(f"  WARNING: no Hessian for {name}, falling back to RTN")
            else:
                print(f"  WARNING: GPTQ inversion failed for {name}, falling back to RTN")
            _rtn_quantize_layer(qlayer, W_fp32, bits, groupsize, sym, orig_dtype,
                                nvfp4=nvfp4)
            n_rtn += 1

    print(f"GPTQ: {n_gptq} layers quantized with GPTQ, {n_rtn} with RTN fallback.")


def _rtn_quantize_layer(qlayer, W_fp32, bits, groupsize, sym, orig_dtype,
                         nvfp4=False):
    """RTN quantization fallback for a single layer."""
    out_features, in_features = W_fp32.shape
    if groupsize > 0 and in_features % groupsize == 0:
        W_g  = W_fp32.reshape(out_features, -1, groupsize)
        if nvfp4:
            scale = W_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / NF4_MAX
            W_norm = W_g / scale
            W_q = round_to_nf4_codebook(W_norm) * scale
        else:
            maxq = (2 ** (bits - 1) - 1) if sym else (2 ** bits - 1)
            if sym:
                scale = W_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / maxq
                W_q   = torch.clamp(torch.round(W_g / scale), -(maxq + 1), maxq) * scale
            else:
                wmin  = torch.minimum(W_g.amin(dim=-1, keepdim=True), torch.zeros_like(W_g[:, :, :1]))
                wmax  = torch.maximum(W_g.amax(dim=-1, keepdim=True), torch.zeros_like(W_g[:, :, :1]))
                scale = (wmax - wmin).clamp(min=1e-5) / maxq
                zero  = torch.round(-wmin / scale)
                W_q   = scale * (torch.clamp(torch.round(W_g / scale) + zero, 0, maxq) - zero)
        qlayer.module.weight.data = W_q.reshape(out_features, in_features).to(orig_dtype)
