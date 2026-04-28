"""
Fused W4A4+tail Triton kernel for DiRotQ (PCA rotation, no algorithm change).

Pipeline (per linear layer):

    x [M, K] bf16
        → cuBLAS matmul: x_rot = x @ U                 (rotation)
            ↓
    [single fused Triton kernel below — eats all the per-step overhead:]
        1. read x_rot (strided view: low part [M, n_low], tail [M, hlen])
        2. quantize x_rot[:, :n_low] to int4 in REGISTERS, per token-group
        3. unpack int4 weights, run int8 mma (2× fp16 throughput), accumulate
        4. read x_rot tail + W_tail, run bf16 mma in same kernel
        5. add bias, write y as bf16
            ↓
    y [M, N] bf16

Why two kernels and not one mega-kernel:
  The rotation x @ U is just a regular bf16 GEMM and cuBLAS is hard to beat
  on it (~165 TFLOPS on RTX 4090). Re-implementing it inside a Triton kernel
  is feasible but unlikely to beat cuBLAS. The 366% overhead the user
  identified came from steps 1-5 — those are what we fuse here.

Tradeoffs vs. SVDQuant's nunchaku kernel:
  - SVDQuant uses a CUDA kernel with inline PTX `mma.m16n8k64.s4.s4.s32`
    (true int4 tensor cores, 4× fp16 throughput). Triton 3.2's `tl.dot`
    doesn't expose that instruction; the closest is int8 mma (2×). Closing
    that gap requires writing the kernel in CUDA C++.
  - SVDQuant has no fp16 tail — they use 1D smoothing + LoRA. We keep
    DiRotQ's fp16 tail in this kernel since the user said "no algorithm
    change". The tail GEMM runs in registers as a final accumulate, so it
    contributes only its compute cost (no separate kernel launch / BW).

API:
    fused_w4a4_pca_forward(
        x,              # [M, K] bf16, the layer's input
        U,              # [K, K] bf16 PCA rotation matrix
        w_low_packed,   # [N, n_low/2] uint8, packed int4 (low region of W_rot)
        w_low_scales,   # [N, n_low/gs] bf16, per-output-channel group scales
        w_tail,         # [N, hlen] bf16, fp16 tail weight  (None if hlen=0)
        bias,           # [N] bf16  (None if no bias)
        gs=64,
    ) -> y [M, N] bf16
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_w4a4_tail_kernel(
    # Activation (already PCA-rotated, contiguous [M, K]):
    XROT_ptr,
    # Weights:
    WLOW_ptr,           # int8 [N, n_low/2] (packed int4)
    WSCALE_ptr,         # bf16 [N, n_low/gs]
    WTAIL_ptr,          # bf16 [N, hlen]   (or null when HAS_TAIL=False)
    # Bias / output:
    BIAS_ptr,           # bf16 [N]
    Y_ptr,              # bf16 [M, N]
    # Sizes (runtime):
    M, N,
    # Strides (runtime):
    stride_xrot_m, stride_xrot_k,
    stride_wlow_n, stride_wlow_kbyte,
    stride_wscale_n, stride_wscale_g,
    stride_wtail_n, stride_wtail_k,
    stride_y_m, stride_y_n,
    # Compile-time:
    K_TOTAL: tl.constexpr,
    N_LOW: tl.constexpr,
    HLEN: tl.constexpr,
    GS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    """One fused kernel: int4 quant + W4A4 (int8 mma path) + fp16 tail + bias.

    Constraints:
      - GS (group size) must equal the K-tile size so each mma operates on a
        single quantization group. Default 64.
      - N_LOW must be a multiple of GS, even (for int4 packing).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # =======================================================================
    # Phase 1: W4A4 main loop — iterate K_low in groups of GS
    # =======================================================================
    # IMPORTANT: use plain `range`, NOT `tl.static_range`. With static_range,
    # Triton fully unrolls the loop — fine for pixart's n_groups=16, but at
    # flux's n_groups=168 the binary explodes and registers spill, costing
    # ~6× the runtime. Plain `range` runs the loop at GPU runtime.
    n_groups = N_LOW // GS
    for g in range(n_groups):
        # Load x_rot chunk [BM, GS] bf16 (strided directly into x_rot)
        offs_k = g * GS + tl.arange(0, GS)
        xrot_offs = offs_m[:, None] * stride_xrot_m + offs_k[None, :] * stride_xrot_k
        x_chunk = tl.load(XROT_ptr + xrot_offs, mask=m_mask[:, None], other=0.0)

        # ---- Per-row int4 quantization, in registers ----
        x_chunk_f = x_chunk.to(tl.float32)
        x_max = tl.max(tl.abs(x_chunk_f), axis=1)             # [BM]
        a_scale = tl.maximum(x_max / 7.0, 1e-6)               # [BM]
        codes_f = x_chunk_f / a_scale[:, None]                # [BM, GS]
        # round (extra.cuda.libdevice fallbacks via add 0.5 trick)
        codes_round = tl.where(
            codes_f >= 0,
            (codes_f + 0.5).to(tl.int32),
            (codes_f - 0.5).to(tl.int32),
        )
        codes_clamped = tl.maximum(tl.minimum(codes_round, 7), -8)
        codes = codes_clamped.to(tl.int8)                     # [BM, GS] int8 in [-8, 7]

        # ---- Unpack W_low int4 → int8 ----
        offs_byte = g * (GS // 2) + tl.arange(0, GS // 2)
        w_offs = offs_n[:, None] * stride_wlow_n + offs_byte[None, :] * stride_wlow_kbyte
        w_bytes = tl.load(WLOW_ptr + w_offs, mask=n_mask[:, None], other=0).to(tl.int8)
        w_low_n = w_bytes & 0xF                               # low nibble, 0..15
        w_high_n = (w_bytes >> 4) & 0xF                        # high nibble
        # Sign-extend 4-bit → 8-bit
        w_low_n = tl.where(w_low_n >= 8, w_low_n - 16, w_low_n).to(tl.int8)
        w_high_n = tl.where(w_high_n >= 8, w_high_n - 16, w_high_n).to(tl.int8)
        # Interleave to recover [BN, GS]
        w_unpacked = tl.interleave(w_low_n, w_high_n)         # [BN, GS] int8

        # Load W scale [BN] for this group
        ws_offs = offs_n * stride_wscale_n + g * stride_wscale_g
        w_scale = tl.load(WSCALE_ptr + ws_offs, mask=n_mask, other=0.0).to(tl.float32)

        # ---- int8 mma: codes [BM, GS] @ w_unpacked.T [GS, BN] → int32 ----
        w_t = tl.trans(w_unpacked)                            # [GS, BN]
        mma = tl.dot(codes, w_t, out_dtype=tl.int32)          # [BM, BN] int32

        # Multiply by per-token × per-channel scales, accumulate
        scales = a_scale[:, None] * w_scale[None, :]
        acc += mma.to(tl.float32) * scales

    # =======================================================================
    # Phase 2: fp16 tail — read x_rot tail + W_tail, accumulate
    # =======================================================================
    if HAS_TAIL:
        TAIL_BK: tl.constexpr = 64
        n_tail_iters: tl.constexpr = (HLEN + TAIL_BK - 1) // TAIL_BK
        # tl.static_range OK here (small count for both pixart and flux: 3-24)
        for ti in tl.static_range(n_tail_iters):
            offs_t = ti * TAIL_BK + tl.arange(0, TAIL_BK)
            t_mask = offs_t < HLEN

            xtail_offs = (offs_m[:, None] * stride_xrot_m
                          + (N_LOW + offs_t)[None, :] * stride_xrot_k)
            x_tail_chunk = tl.load(
                XROT_ptr + xtail_offs,
                mask=m_mask[:, None] & t_mask[None, :], other=0.0)

            wt_offs = (offs_n[:, None] * stride_wtail_n
                       + offs_t[None, :] * stride_wtail_k)
            w_tail_chunk = tl.load(
                WTAIL_ptr + wt_offs,
                mask=n_mask[:, None] & t_mask[None, :], other=0.0)

            wt_t = tl.trans(w_tail_chunk)                     # [TAIL_BK, BN]
            tail_mma = tl.dot(x_tail_chunk, wt_t, out_dtype=tl.float32)
            acc += tail_mma

    # =======================================================================
    # Bias + write
    # =======================================================================
    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    out_offs = offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(Y_ptr + out_offs, acc.to(tl.bfloat16), mask=out_mask)


def fused_w4a4_pca_forward(
    x: torch.Tensor,
    U: torch.Tensor,
    w_low_packed: torch.Tensor,
    w_low_scales: torch.Tensor,
    w_tail: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    gs: int = 64,
) -> torch.Tensor:
    """End-to-end forward: PCA rotation (cuBLAS) + fused Triton kernel.

    Shapes:
        x:             [M, K] bf16
        U:             [K, K] bf16 (PCA rotation; can be passed as fp16 too)
        w_low_packed:  [N, n_low/2] uint8 or int8
        w_low_scales:  [N, n_low/gs] bf16
        w_tail:        [N, hlen] bf16  (or None if hlen=0)
        bias:          [N] bf16        (or None)
    """
    assert x.dim() == 2 and U.dim() == 2 and w_low_packed.dim() == 2
    M, K = x.shape
    assert U.shape == (K, K), f"U shape {U.shape} != ({K},{K})"

    N = w_low_packed.shape[0]
    n_low = w_low_packed.shape[1] * 2
    assert n_low % gs == 0
    hlen = K - n_low
    if w_tail is not None:
        assert w_tail.shape == (N, hlen), f"w_tail shape mismatch: {w_tail.shape} vs ({N},{hlen})"

    # ---- Phase 0: cuBLAS rotation ----
    # x @ U: bf16 input/output. cuBLAS optimal.
    x_rot = x @ U  # [M, K] bf16, contiguous
    if not x_rot.is_contiguous():
        x_rot = x_rot.contiguous()

    # ---- Phase 1+2: fused Triton kernel ----
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    # Per-shape tile selection. Pixart uses small K (1152); flux is bigger
    # (3072, 12288). Bigger shapes benefit from larger tiles (BM=128, BN=256).
    # Pixart small shapes do better with BM=32 due to launch + L2 dynamics.
    if K >= 4096 or n_low >= 6000:
        # flux ff_up / ff_down regime (large K and/or large N_LOW)
        BM, BN, num_warps, num_stages = 128, 256, 8, 3
    elif K >= 2048:
        # flux attn regime
        BM, BN, num_warps, num_stages = 128, 256, 8, 3
    else:
        # pixart regime (small K)
        BM, BN, num_warps, num_stages = 32, 128, 8, 3
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    # Use a dummy w_tail/bias tensor when not provided so pointer is non-null
    have_tail = w_tail is not None and hlen > 0
    have_bias = bias is not None
    w_tail_arg = w_tail if have_tail else torch.empty(1, dtype=x.dtype, device=x.device)
    bias_arg = bias if have_bias else torch.empty(1, dtype=x.dtype, device=x.device)

    # Cast packed weight to int8 for mma path
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
