"""
GPTQ weight quantization utilities for DiRotQ.

Two-phase workflow:
  1. collect_hessians()  — run calibration samples through the model with
                           forward hooks on ActQuantWrapper to accumulate per-layer
                           Hessian matrices H = 2/n * X^T X.
  2. gptq_quantize_weights() — apply the GPTQ algorithm to each layer using its H.

Rotated-basis GPTQ with weight-side high-precision tail
-------------------------------------------------------
Hessians are collected on pre-rotation inputs (X_pre). Before GPTQ runs on a
layer, we fuse the rotation into both W and H:

  Hidden PCA:   H_rot = U.T @ H_pre @ U
  Per-head PCA: H_rot via einsum over BlockDiag(U_ph)
  Hadamard:     H_rot_low = H_dense @ H_pre[:low,:low] @ H_dense.T

GPTQ runs in the rotated basis on the low block only. The last
`high_bits_length` columns (the fp16 tail) are sliced off and left exact.

After GPTQ the layer weight already lives in the rotated basis; the layer
is flagged `_unrot_fused=True` so the patched forward skips the unrotation.
"""

import gc
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from .quant_utils import (
    ActQuantWrapper, find_qlayers, round_to_nf4_codebook, NF4_MAX, _match_skip,
    _rotate_and_split_W, _quant_group_int, _quant_group_nvfp4,
)
from .hadamard_utils import fast_hadamard_transform


class _NewlineStream:
    """tqdm writes progress by overwriting with \\r, which turns log files
    into one giant line. This stream replaces \\r with \\n so each update
    lands on its own line — openable in any editor."""
    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        self._stream.write(s.replace("\r", "\n"))

    def flush(self):
        self._stream.flush()


def _log_tqdm(iterable, desc, **kwargs):
    """tqdm that stays TTY-friendly interactively, but emits newline-per-update
    when redirected to a file (e.g. nohup log)."""
    if sys.stdout.isatty():
        return tqdm(iterable, desc=desc, **kwargs)
    return tqdm(iterable, desc=desc, file=_NewlineStream(sys.stdout),
                mininterval=10.0, **kwargs)


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
    # Enable TF32 tensor cores for the fp32 Hessian matmuls only if the GPU
    # actually supports them (Ampere, sm_80+). On older GPUs (V100 sm_70,
    # T4 sm_75) TF32 is a no-op/unsupported; leave strict fp32 in place.
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(
            device if isinstance(device, int) else 0
        )
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"TF32 enabled for Hessian matmuls (compute capability sm_{major}{minor}).")
        else:
            print(f"TF32 unsupported on sm_{major}{minor}; using strict fp32 for Hessian matmuls.")

    # Match calibration input dtype to the transformer so bf16 models
    # (e.g. FLUX) don't silently lose dynamic range via fp16 cast.
    model_dtype = next(transformer.parameters()).dtype

    qlayers = find_qlayers(transformer, layers=[ActQuantWrapper])

    # Hybrid placement: layers whose Hessian fits cheaply live on GPU and use
    # in-place addmm_ (no PCIe traffic, no per-call alloc). Big-D layers
    # (FLUX ffn_down @12288, single_block proj_out.linears.1 @12288) accumulate
    # on CPU to avoid eating tens of GB of GPU RAM.
    GPU_HESSIAN_MAX_D = 4096

    layer_H = {}
    layer_on_gpu = {}
    for name, ql in qlayers.items():
        d = ql.module.in_features
        if d <= GPU_HESSIAN_MAX_D:
            layer_H[name] = torch.zeros(d, d, dtype=torch.float32, device=device)
            layer_on_gpu[name] = True
        else:
            layer_H[name] = torch.zeros(d, d, dtype=torch.float32)
            layer_on_gpu[name] = False

    n_gpu = sum(layer_on_gpu.values())
    gpu_mem_mb = sum(ql.module.in_features ** 2 * 4
                     for name, ql in qlayers.items() if layer_on_gpu[name]) / (1024 ** 2)
    print(f"Hessian placement: {n_gpu}/{len(qlayers)} on GPU "
          f"(~{gpu_mem_mb:.0f} MiB), {len(qlayers) - n_gpu} on CPU (D > {GPU_HESSIAN_MAX_D}).")

    layer_n = {name: 0 for name in qlayers}

    # Register hooks on ActQuantWrapper to capture input x.
    # GPU layers accumulate in-place via addmm_ — no PCIe traffic.
    # CPU layers: partial XTX stays on GPU inside the hook and is flushed to
    # CPU after each full forward pass (_flush_pending), so the forward pass
    # is never stalled by synchronous GPU→CPU memcpy.
    _pending = []  # list of (layer_name, gpu_partial) for CPU layers

    def _flush_pending():
        if not _pending:
            return
        torch.cuda.synchronize()
        for n, p_gpu in _pending:
            layer_H[n] += p_gpu.cpu()
        _pending.clear()

    hooks = []
    for name, ql in qlayers.items():
        in_features = ql.module.in_features

        def _make_hook(n, d, on_gpu):
            if on_gpu:
                def _hook(module, inp, out):
                    x = inp[0].detach().float().reshape(-1, d)
                    layer_n[n] += x.shape[0]
                    layer_H[n].addmm_(x.T, x)
            else:
                def _hook(module, inp, out):
                    x = inp[0].detach().float().reshape(-1, d)
                    layer_n[n] += x.shape[0]
                    _pending.append((n, x.T @ x))  # stays on GPU, flushed after pass
            return _hook

        hooks.append(ql.register_forward_hook(
            _make_hook(name, in_features, layer_on_gpu[name])
        ))

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
                # Only concat per-sample tensors (leading dim 1). Tensors
                # without a batch dim — e.g. FLUX img_ids [L, 3] and
                # txt_ids [512, 3] — are shared position IDs across the
                # batch and must be passed as-is, not cat'd.
                if vals[0].ndim >= 1 and vals[0].shape[0] == 1:
                    stacked_kwargs[k] = torch.cat(vals, dim=0).to(device)
                else:
                    stacked_kwargs[k] = vals[0].to(device)
            else:
                stacked_kwargs[k] = vals[0]  # scalar/non-tensor: use first
        with torch.no_grad():
            transformer(latents, **stacked_kwargs)
        _flush_pending()  # drain deferred CPU transfers without stalling forward pass
        batch_args.clear()
        for k in batch_kwargs_lists:
            batch_kwargs_lists[k].clear()

    # Group files into batches upfront so each thread-pool task loads a whole
    # batch. tqdm then counts BATCHES not individual files.
    file_batches = [calib_files[i:i + batch_size]
                    for i in range(0, len(calib_files), batch_size)]
    num_batches  = len(file_batches)

    # Prefetch 2 full batches ahead so NFS I/O overlaps GPU compute.
    PREFETCH = 2

    def _load_batch(files):
        return [torch.load(f, map_location="cpu", weights_only=False) for f in files]

    def _enqueue_batch(batch_data):
        for data in batch_data:
            batch_args.append([a.to(model_dtype) if a.is_floating_point() else a
                               for a in data["input_args"]])
            for k, v in data["input_kwargs"].items():
                if k not in batch_kwargs_lists:
                    batch_kwargs_lists[k] = []
                if isinstance(v, torch.Tensor):
                    batch_kwargs_lists[k].append(
                        v.to(model_dtype) if v.is_floating_point() else v
                    )
                else:
                    batch_kwargs_lists[k].append(v)

    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        io_queue  = deque()
        batch_iter = iter(file_batches)

        for _ in range(PREFETCH):
            try:
                io_queue.append(pool.submit(_load_batch, next(batch_iter)))
            except StopIteration:
                break

        for _ in _log_tqdm(range(num_batches), desc="Calibration forward passes"):
            try:
                io_queue.append(pool.submit(_load_batch, next(batch_iter)))
            except StopIteration:
                pass
            _enqueue_batch(io_queue.popleft().result())
            _run_batch()  # always a full batch (or final partial)

    for h in hooks:
        h.remove()

    # Normalize H = 2/n * Σ X^T X in place, then move any GPU-resident
    # accumulators to CPU before returning — the returned dict is saved via
    # torch.save and consumed by _gptq_quantize_layer which handles CPU tensors.
    for name in layer_H:
        n = layer_n[name]
        if n > 0:
            layer_H[name].mul_(2.0 / n)
            if layer_H[name].device.type == "cuda":
                layer_H[name] = layer_H[name].cpu()
        else:
            print(f"WARNING: no calibration data collected for layer {name}")

    gc.collect()
    torch.cuda.empty_cache()
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
    # Handles partial last group by zero-padding to next multiple of groupsize.
    if groupsize > 0 and groupsize < in_features:
        pad = (-in_features) % groupsize
        W_for_scale = torch.nn.functional.pad(W, (0, pad)) if pad > 0 else W
        n_groups = W_for_scale.shape[1] // groupsize
        W_g = W_for_scale.reshape(out_features, n_groups, groupsize)
        if nvfp4:
            scales = W_g.abs().amax(dim=-1).clamp(min=1e-5) / NF4_MAX
            zeros  = torch.zeros_like(scales)
        elif sym:
            scales = W_g.abs().amax(dim=-1).clamp(min=1e-5) / maxq
            zeros  = torch.zeros_like(scales)
        else:
            wmin   = torch.minimum(W_g.amin(dim=-1), torch.zeros(out_features, n_groups, device=device))
            wmax   = torch.maximum(W_g.amax(dim=-1), torch.zeros(out_features, n_groups, device=device))
            scales = (wmax - wmin).clamp(min=1e-5) / maxq
            zeros  = torch.round(-wmin / scales)
    else:
        # Per-channel (groupsize <= 0 or >= in_features)
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
        groupsize = in_features

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

    # Cholesky inversion (with retry on numerical failure OR near-zero diag)
    #
    # A "successful" Cholesky can still leave near-zero diagonals in H_inv
    # (the upper Cholesky factor of H^{-1}).  Those tiny d values cause
    # err_col = (w - w_q) / d  to explode during error propagation, silently
    # producing NaN weights.  Detect this and add more damping before retrying.
    H_inv = None
    for _ in range(num_inv_tries):
        try:
            L     = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv = torch.linalg.cholesky(H_inv, upper=True)
            # Reject if any diagonal is < 1e-4 × mean — those columns would
            # produce catastrophic error propagation (err / d -> Inf -> NaN).
            diag = H_inv.diagonal()
            if diag.min() < 1e-4 * diag.mean():
                H_inv = None          # force retry
                H_diag += damp_pct * H_diag.mean()
                continue
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
    """Apply GPTQ weight quantization to all ActQuantWrapper layers.

    For layers with hidden-dim rotation (bits<16): rotate H into the PCA
    basis (H_rot = U^T H U), slice the top-left [n_low, n_low] block
    matching the quantized column region, run GPTQ on W_rot[:, :n_low]
    against H_rot[:n_low, :n_low], then stitch the fp16 tail back.

    For layers with per-head rotation: rotate H into per-head PCA basis
    via einsum, extract the low-low block, run GPTQ.

    For layers with Hadamard rotation: rotate H[:low,:low] into the FWHT
    basis, run GPTQ on the rotated low block.

    For layers with no rotation or bits=16 activations: plain GPTQ in
    the original basis.

    Any layer that undergoes rotate+split weight quant is flagged
    `_unrot_fused=True` so the patched forward reads the rotated weight.
    """
    if skip_names is None:
        skip_names = []

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    n_gptq = 0
    n_rtn_fail = 0

    def _rtn_low(W_low, gs):
        if nvfp4:
            return _quant_group_nvfp4(W_low, gs)
        return _quant_group_int(W_low, bits, gs, sym)

    for name, qlayer in _log_tqdm(qlayers.items(), desc="GPTQ weight quantization"):
        if _match_skip(name, skip_names):
            continue

        W = qlayer.module.weight.data
        orig_dtype = W.dtype
        rotate_hidden = (qlayer.quantizer.bits < 16 and qlayer.rotation is not None)
        rotate_head   = (qlayer.quantizer.bits < 16 and qlayer.rotation_per_head is not None)
        rotate_had    = (qlayer.quantizer.bits < 16 and getattr(qlayer, 'use_hadamard', False))

        if rotate_hidden:
            # Rotate W, slice off fp16 tail. Both W_low and W_tail live on W's
            # current device; move W_low to `device` for GPTQ math.
            W_low, W_tail, stitch, _ = _rotate_and_split_W(qlayer, W)
            n_low = W_low.shape[1]

            H_raw = hessians.get(name)
            W_low_q = None
            if H_raw is not None:
                # H_rot = U.T @ H_raw @ U, then top-left [n_low, n_low] block.
                U = qlayer.rotation.to(device=device, dtype=torch.float32)  # [D, D]
                H_dev = H_raw.to(device=device, dtype=torch.float32)
                H_rot = U.T @ H_dev @ U
                H_low = H_rot[:n_low, :n_low].contiguous()

                # Rescale damping: the low block keeps small-eigenvalue PCA
                # directions, so mean(diag(H_low)) << mean(diag(H_rot)).
                # Scale damp_pct so absolute damping matches the full Hessian.
                mean_full = H_rot.diagonal().mean().clamp(min=1e-12)
                mean_low  = H_low.diagonal().mean().clamp(min=1e-12)
                damp_scale = float(torch.clamp(mean_full / mean_low, min=1.0).item())

                del H_rot, H_dev
                W_low_q = _gptq_quantize_layer(
                    W_low.to(device), H_low, bits, groupsize, sym,
                    damp_pct * damp_scale, block_size, num_inv_tries, device,
                    nvfp4=nvfp4,
                )
                del H_low

            if W_low_q is None:
                if H_raw is None:
                    print(f"  WARNING: no Hessian for {name}, falling back to rotated RTN")
                else:
                    print(f"  WARNING: GPTQ inversion failed for {name}, falling back to rotated RTN")
                W_low_q = _rtn_low(W_low, groupsize)
                n_rtn_fail += 1
            else:
                n_gptq += 1

            qlayer.module.weight.data = stitch(W_low_q).to(orig_dtype)
            qlayer._unrot_fused = True

        elif rotate_head:
            # Per-head rotation: rotate H into per-head PCA basis, extract
            # the low-low block across all heads, run GPTQ.
            #
            # H_rot = BlockDiag(U_ph).T @ H_raw @ BlockDiag(U_ph)
            # computed via einsum to avoid materializing the full BD matrix.
            # Low block: first d_q cols of each head, grouped contiguously
            # to match W_low layout from _rotate_and_split_W.
            W_low, W_tail, stitch, gs_over = _rotate_and_split_W(qlayer, W)
            gs = gs_over if gs_over is not None else groupsize
            n_low = W_low.shape[1]

            H_raw = hessians.get(name)
            W_low_q = None
            if H_raw is not None:
                U_ph = qlayer.rotation_per_head.to(device=device, dtype=torch.float32)
                n_heads = int(qlayer.num_heads)
                hd = int(qlayer.head_dim)
                hlen = int(qlayer.quantizer.high_bits_length)
                d_q = hd - hlen

                H_dev = H_raw.to(device=device, dtype=torch.float32)
                H_4d = H_dev.reshape(n_heads, hd, n_heads, hd)

                # Step 1: contract k  →  temp[h1, i, h2, l]
                temp = torch.einsum('aki, akbl -> aibl', U_ph, H_4d)
                # Step 2: contract l  →  H_rot_4d[h1, i, h2, j]
                H_rot_4d = torch.einsum('aibl, blj -> aibj', temp, U_ph)
                del temp, H_4d, H_dev

                # Extract low-low block: first d_q channels per head
                H_rot_low = H_rot_4d[:, :d_q, :, :d_q].reshape(
                    n_heads * d_q, n_heads * d_q
                ).contiguous()

                # Damping rescale (same logic as hidden rotation)
                mean_full_ph = H_rot_4d.reshape(n_heads * hd, n_heads * hd).diagonal().mean().clamp(min=1e-12)
                mean_low_ph  = H_rot_low.diagonal().mean().clamp(min=1e-12)
                damp_scale_ph = float(torch.clamp(mean_full_ph / mean_low_ph, min=1.0).item())
                del H_rot_4d

                W_low_q = _gptq_quantize_layer(
                    W_low.to(device), H_rot_low, bits, gs, sym,
                    damp_pct * damp_scale_ph, block_size, num_inv_tries, device,
                    nvfp4=nvfp4,
                )
                del H_rot_low

            if W_low_q is None:
                if H_raw is None:
                    print(f"  WARNING: no Hessian for {name}, falling back to rotated RTN")
                else:
                    print(f"  WARNING: GPTQ inversion failed for {name}, falling back to rotated RTN")
                W_low_q = _rtn_low(W_low, gs)
                n_rtn_fail += 1
            else:
                n_gptq += 1

            qlayer.module.weight.data = stitch(W_low_q).to(orig_dtype)
            qlayer._unrot_fused = True

        elif rotate_had:
            # Hadamard rotation: rotate H_raw[:low, :low] into FWHT basis,
            # run GPTQ on the rotated low block.
            #
            # H_rot_low = H_dense @ H_raw[:low, :low] @ H_dense.T
            # where H_dense = FWHT @ diag(sign_flips)  (same as weight rotation).
            W_low, W_tail, stitch, gs_over = _rotate_and_split_W(qlayer, W)
            gs = gs_over if gs_over is not None else groupsize
            n_low = W_low.shape[1]

            H_raw = hessians.get(name)
            W_low_q = None
            if H_raw is not None:
                low_dim = int(qlayer.hadamard_low_dim)
                H_dense = torch.eye(low_dim, dtype=torch.float32, device=device)
                H_dense = fast_hadamard_transform(H_dense)
                if qlayer.hadamard_sign_flips is not None:
                    sf = qlayer.hadamard_sign_flips.to(device, dtype=torch.float32)
                    H_dense = sf.unsqueeze(0) * H_dense

                H_raw_low = H_raw[:low_dim, :low_dim].to(device=device, dtype=torch.float32)
                H_rot_low = (H_dense @ H_raw_low @ H_dense.T).contiguous()

                # Damping rescale: Hadamard is orthogonal so trace is
                # preserved, but the low block may still be ill-conditioned.
                H_dev = H_raw.to(device=device, dtype=torch.float32)
                mean_full_had = H_dev.diagonal().mean().clamp(min=1e-12)
                mean_low_had  = H_rot_low.diagonal().mean().clamp(min=1e-12)
                damp_scale_had = float(torch.clamp(mean_full_had / mean_low_had, min=1.0).item())
                del H_raw_low, H_dense, H_dev

                W_low_q = _gptq_quantize_layer(
                    W_low.to(device), H_rot_low, bits, gs, sym,
                    damp_pct * damp_scale_had, block_size, num_inv_tries, device,
                    nvfp4=nvfp4,
                )
                del H_rot_low

            if W_low_q is None:
                if H_raw is None:
                    print(f"  WARNING: no Hessian for {name}, falling back to rotated RTN")
                else:
                    print(f"  WARNING: GPTQ inversion failed for {name}, falling back to rotated RTN")
                W_low_q = _rtn_low(W_low, gs)
                n_rtn_fail += 1
            else:
                n_gptq += 1

            qlayer.module.weight.data = stitch(W_low_q).to(orig_dtype)
            qlayer._unrot_fused = True

        else:
            # No rotation or bits=16: plain GPTQ in original basis.
            W_fp32 = W.float()
            H_raw = hessians.get(name)
            W_q = None
            if H_raw is not None:
                W_q = _gptq_quantize_layer(
                    W_fp32, H_raw, bits, groupsize, sym,
                    damp_pct, block_size, num_inv_tries, device,
                    nvfp4=nvfp4,
                )
            if W_q is None:
                if H_raw is None:
                    print(f"  WARNING: no Hessian for {name}, falling back to RTN")
                else:
                    print(f"  WARNING: GPTQ inversion failed for {name}, falling back to RTN")
                W_q = _rtn_low(W_fp32, groupsize)
                n_rtn_fail += 1
            else:
                n_gptq += 1
            qlayer.module.weight.data = W_q.to(orig_dtype)

    print(f"GPTQ: {n_gptq} GPTQ, {n_rtn_fail} RTN fallback (Cholesky failure).")
