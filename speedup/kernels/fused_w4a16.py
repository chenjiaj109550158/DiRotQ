"""
Fused W4A16 Triton kernel for the act-skip case (DiRotQ's --skip-quant-layers,
e.g. pixart's `ff.net.2`).

What this kernel does in ONE pass:
    bf16 act × int4 weight → bf16 output (+ bias)

Why we need our own:
    `torch._weight_int4pack_mm` is a mobile-targeted primitive: it dequants
    weights to bf16 internally with no tensor-core utilization, so on
    RTX 4090 at compute-bound shapes (M=4096) it's ~8.5× slower than fp16
    cuBLAS. A properly tuned bf16 mma + on-the-fly weight dequant kernel
    pays only the int4-weight bandwidth saving (4× less weight traffic) and
    runs at fp16 mma throughput, which on this hardware is the right model.

Layout assumptions:
    x:           [M, K] bf16 (activation)
    W_packed:    [N, K/2] uint8/int8 (each byte holds 2 int4 codes;
                                       low nibble = first, high = second —
                                       same convention as triton_w4a4.pack_w4)
    W_scales:    [N, K/gs] bf16 (per-output-channel, per-group scale)
    bias:        [N] bf16 (optional)
    output:      [M, N] bf16
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_fused_kernel(
    X_ptr, W_packed_ptr, W_scales_ptr, BIAS_ptr, Y_ptr,
    M, N, K,
    stride_x_m, stride_x_k,
    stride_w_n, stride_w_k,        # W_packed in BYTES along K
    stride_ws_n, stride_ws_g,
    stride_y_m, stride_y_n,
    GS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """W4A16 GEMM: bf16 × int4-dequant → bf16. One iteration per group (BK = GS)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    n_groups = K // GS
    for g in range(n_groups):
        # Load x [BM, GS] bf16
        offs_k = g * GS + tl.arange(0, GS)
        x_offs = offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(X_ptr + x_offs, mask=m_mask[:, None], other=0.0)

        # Load W_packed bytes [BN, GS/2]
        offs_byte = g * (GS // 2) + tl.arange(0, GS // 2)
        w_offs = offs_n[:, None] * stride_w_n + offs_byte[None, :] * stride_w_k
        w_bytes = tl.load(W_packed_ptr + w_offs, mask=n_mask[:, None], other=0).to(tl.int8)

        # Unpack int4 → int8
        w_low_n = w_bytes & 0xF
        w_high_n = (w_bytes >> 4) & 0xF
        w_low_n = tl.where(w_low_n >= 8, w_low_n - 16, w_low_n).to(tl.int8)
        w_high_n = tl.where(w_high_n >= 8, w_high_n - 16, w_high_n).to(tl.int8)
        w_codes = tl.interleave(w_low_n, w_high_n)            # [BN, GS] int8

        # Per-output-channel scale for this group
        ws_offs = offs_n * stride_ws_n + g * stride_ws_g
        w_scale = tl.load(W_scales_ptr + ws_offs, mask=n_mask, other=0.0).to(tl.float32)

        # Dequantize W to bf16 (in-register) and run bf16 mma
        w_bf = (w_codes.to(tl.float32) * w_scale[:, None]).to(tl.bfloat16)
        # mma: x [BM, GS] @ w_bf.T [GS, BN] → fp32
        w_t = tl.trans(w_bf)
        mma = tl.dot(x, w_t, out_dtype=tl.float32)
        acc += mma

    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    out_offs = offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    tl.store(Y_ptr + out_offs, acc.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def w4a16_fused_forward(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    gs: int = 64,
) -> torch.Tensor:
    """End-to-end W4A16 forward via the fused Triton kernel.

    Shapes:
        x:        [M, K] bf16
        w_packed: [N, K/2] uint8/int8
        w_scales: [N, K/gs] bf16
        bias:     [N] bf16 or None
    """
    assert x.dim() == 2 and w_packed.dim() == 2 and w_scales.dim() == 2
    M, K = x.shape
    N = w_packed.shape[0]
    assert w_packed.shape[1] * 2 == K, \
        f"K={K} but w_packed.shape[1]*2={w_packed.shape[1]*2}"
    assert K % gs == 0, f"K={K} not divisible by gs={gs}"

    if w_packed.dtype != torch.int8:
        w_packed_i8 = w_packed.view(torch.int8)
    else:
        w_packed_i8 = w_packed

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Tuned for M=4096, K up to ~5000 on RTX 4090. Same shape family as the
    # fused W4A4 kernel; reuse the BM/BN/warps choice for consistency.
    BM = 64
    BN = 128
    num_warps = 4
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    bias_arg = bias if bias is not None else torch.empty(1, dtype=x.dtype, device=x.device)

    _w4a16_fused_kernel[grid](
        x, w_packed_i8, w_scales, bias_arg, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed_i8.stride(0), w_packed_i8.stride(1),
        w_scales.stride(0), w_scales.stride(1),
        y.stride(0), y.stride(1),
        GS=gs, BM=BM, BN=BN,
        HAS_BIAS=(bias is not None),
        num_warps=num_warps,
    )
    return y
