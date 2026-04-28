"""
Wrapper that swaps the cuBLAS bf16 rotation in fused_w4a4_pca for the
int8-mma rotation kernel. Same fused W4A4+tail+bias kernel after.

This is the path-2 light approach:
  1. Read x (bf16)
  2. Triton int8 rotation → x_rot (bf16, written to global)
  3. Read x_rot, quantize to int4 in-kernel, W4A4 mma + tail + bias, write y

Two writes (x_rot, y) and two reads (x, x_rot) of activation per layer.
Not "1 read 1 write" — but the int4-packed rotation output (one_write)
turned out to add more pack/quant overhead than it saves. So we keep
the cleaner two-kernel path.

Calibration (one-time): U_low and U_high get quantized to int8 with
per-column scales. Stored alongside the existing rotation matrix.
"""

from __future__ import annotations

import torch
import triton

from speedup.kernels.int8_rotation import (
    int8_rotation_forward, quantize_U_int8, compute_scale_x,
)
from speedup.kernels.fused_w4a4_pca import _fused_w4a4_tail_kernel


def precompute_int8_U(U: torch.Tensor) -> dict:
    """Quantize U to int8 once at calibration time."""
    U_int8, scale_U = quantize_U_int8(U)
    return {"U_int8": U_int8, "scale_U": scale_U}


def fused_w4a4_pca_int8rot_forward(
    x: torch.Tensor,
    U_int8: torch.Tensor,
    scale_U: torch.Tensor,
    w_low_packed: torch.Tensor,
    w_low_scales: torch.Tensor,
    w_tail: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    gs: int = 64,
    scale_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-to-end forward: int8 rotation kernel + fused Triton W4A4 kernel.

    Shapes:
        x:             [M, K] bf16
        U_int8:        [K, K] int8 (precomputed)
        scale_U:       [K] fp32 (per-column)
        w_low_packed:  [N, n_low/2] uint8/int8
        w_low_scales:  [N, n_low/gs] bf16
        w_tail:        [N, hlen] bf16 or None
        bias:          [N] bf16 or None
    """
    assert x.dim() == 2
    M, K = x.shape
    N = w_low_packed.shape[0]
    n_low = w_low_packed.shape[1] * 2
    hlen = K - n_low

    # ---- Phase 0: int8 rotation, bf16 output ----
    x_rot = int8_rotation_forward(x, U_int8, scale_U, scale_x=scale_x)
    if not x_rot.is_contiguous():
        x_rot = x_rot.contiguous()

    # ---- Phase 1+2: fused Triton kernel (same as bf16-rotation path) ----
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    # Per-shape tile selection (mirrors fused_w4a4_pca.fused_w4a4_pca_forward).
    if K >= 4096 or n_low >= 6000:
        BM, BN, num_warps, num_stages = 128, 256, 8, 3
    elif K >= 2048:
        BM, BN, num_warps, num_stages = 128, 256, 8, 3
    else:
        BM, BN, num_warps, num_stages = 32, 128, 8, 3
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    have_tail = w_tail is not None and hlen > 0
    have_bias = bias is not None
    w_tail_arg = w_tail if have_tail else torch.empty(1, dtype=x.dtype, device=x.device)
    bias_arg = bias if have_bias else torch.empty(1, dtype=x.dtype, device=x.device)

    if w_low_packed.dtype != torch.int8:
        w_low_packed_i8 = w_low_packed.view(torch.int8)
    else:
        w_low_packed_i8 = w_low_packed

    _fused_w4a4_tail_kernel[grid](
        x_rot, w_low_packed_i8, w_low_scales, w_tail_arg, bias_arg, y,
        M, N,
        x_rot.stride(0), x_rot.stride(1),
        w_low_packed_i8.stride(0), w_low_packed_i8.stride(1),
        w_low_scales.stride(0), w_low_scales.stride(1),
        w_tail_arg.stride(0) if have_tail else 0,
        w_tail_arg.stride(1) if have_tail else 0,
        y.stride(0), y.stride(1),
        K_TOTAL=K,
        N_LOW=n_low,
        HLEN=hlen if have_tail else 0,
        GS=gs,
        BM=BM, BN=BN,
        HAS_BIAS=have_bias,
        HAS_TAIL=have_tail,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y
