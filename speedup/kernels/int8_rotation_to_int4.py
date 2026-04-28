"""
speedup/kernels/int8_rotation_to_int4.py

Triton kernel that fuses:
  1. Read x [M, K] bf16 from global memory
  2. int8-quantize x in registers (per-row scale, computed elsewhere)
  3. int8 mma against U_low_int8 [K, n_low] → x_rot_low [BM, BN]  (BN spans GS columns at a time)
  4. Per-row int4 quantization of x_rot_low → codes + scales
  5. Pack two int4 codes per byte
  6. Write packed [M, n_low/2] int8 + scales [M, n_low/gs] bf16 to global

This is the "one path write" SVDQuant-style fusion: the activation is
read once (bf16) and written once (int4-packed for the next layer).

The kernel is parameterized so each thread block handles BN = GS columns
of the rotation output (one int4 quantization group per block). This
keeps the per-row int4 scale computation simple — one scale per block.

The high region (`x_rot_high = x @ U_high`) goes through the regular bf16
int8_rotation_forward kernel separately.

Layout:
    x:          [M, K] bf16
    U_low_int8: [K, n_low] int8
    scale_x:    [M] fp32 (precomputed per-row)
    scale_U:    [n_low] fp32 (precomputed per-column, calibrated)
    OUTPUT packed: [M, n_low/2] int8
    OUTPUT scales: [M, n_low/gs] bf16
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_rot_to_int4_kernel(
    X_ptr,                  # [M, K] bf16
    U_LOW_INT8_ptr,         # [K, N_LOW] int8
    SCALE_X_ptr,            # [M] fp32
    SCALE_U_LOW_ptr,        # [N_LOW] fp32
    OUT_PACKED_ptr,         # [M, N_LOW/2] int8 (output)
    OUT_SCALES_ptr,         # [M, N_LOW/GS] bf16 (output)
    M, N_LOW, K,
    sx_m, sx_k,
    su_k, su_n,
    sp_m, sp_n,
    ss_m, ss_g,
    BM: tl.constexpr,
    GS: tl.constexpr,        # group size = BN per block
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """One block per (BM × GS) output region.

    Each block produces one int4 quantization group worth of x_rot output
    (GS columns) for BM rows, packed and scaled.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_g = tl.cdiv(N_LOW, GS)
    num_pid_in_group = GROUP_M * num_pid_g
    grp = pid // num_pid_in_group
    first_m = grp * GROUP_M
    grp_size_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_pid_in_group) % grp_size_m)
    pid_g = (pid % num_pid_in_group) // grp_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_g * GS + tl.arange(0, GS)
    m_mask = offs_m < M
    n_mask = offs_n < N_LOW

    # Per-row x scale & per-column U scale (constants for this block)
    scale_x = tl.load(SCALE_X_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    scale_u = tl.load(SCALE_U_LOW_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    # int32 mma accumulator for x_rot[BM, GS]
    acc = tl.zeros((BM, GS), dtype=tl.int32)

    offs_k_init = tl.arange(0, BK)
    x_ptrs = X_ptr + offs_m[:, None] * sx_m + offs_k_init[None, :] * sx_k
    u_ptrs = U_LOW_INT8_ptr + offs_k_init[:, None] * su_k + offs_n[None, :] * su_n

    for k_off in range(0, tl.cdiv(K, BK)):
        k_mask = (k_off * BK + offs_k_init) < K
        # Load + quantize x to int8 in registers
        x_bf = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        x_codes_f = x_bf.to(tl.float32) / scale_x[:, None]
        x_codes = tl.where(
            x_codes_f >= 0,
            (x_codes_f + 0.5).to(tl.int32),
            (x_codes_f - 0.5).to(tl.int32),
        )
        x_codes = tl.maximum(tl.minimum(x_codes, 127), -128).to(tl.int8)

        u_int8 = tl.load(u_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0).to(tl.int8)
        acc = tl.dot(x_codes, u_int8, acc, out_dtype=tl.int32)

        x_ptrs += BK * sx_k
        u_ptrs += BK * su_k

    # x_rot[BM, GS] in fp32
    x_rot_fp32 = acc.to(tl.float32) * scale_x[:, None] * scale_u[None, :]

    # int4 quantization: per-row max-abs / 7 → scale_a [BM]
    x_rot_max = tl.max(tl.abs(x_rot_fp32), axis=1)
    scale_a = tl.maximum(x_rot_max / 7.0, 1e-9)
    codes_f = x_rot_fp32 / scale_a[:, None]
    codes_i = tl.where(
        codes_f >= 0,
        (codes_f + 0.5).to(tl.int32),
        (codes_f - 0.5).to(tl.int32),
    )
    codes_i = tl.maximum(tl.minimum(codes_i, 7), -8)
    codes = codes_i.to(tl.int8)

    # Pack two int4 per byte: low nibble = codes[:, 0::2], high = codes[:, 1::2]
    # This matches triton_w4a4.pack_w4 / fused_w4a4_pca convention.
    HALF: tl.constexpr = GS // 2
    # Use tl.reshape + tl.split to extract even/odd along last dim.
    codes_2d = tl.reshape(codes, (BM, HALF, 2))
    even, odd = tl.split(codes_2d)            # each [BM, HALF] int8
    packed = ((even & 0xF) | ((odd & 0xF) << 4)).to(tl.int8)

    # Write packed
    out_offs = offs_m[:, None] * sp_m + (pid_g * HALF + tl.arange(0, HALF))[None, :] * sp_n
    tl.store(OUT_PACKED_ptr + out_offs, packed,
             mask=m_mask[:, None] & ((pid_g * HALF + tl.arange(0, HALF))[None, :] < (N_LOW // 2)))

    # Write scale (one per row per group)
    scale_offs = offs_m * ss_m + pid_g * ss_g
    tl.store(OUT_SCALES_ptr + scale_offs, scale_a.to(tl.bfloat16), mask=m_mask)


def int8_rotation_to_int4_low(
    x: torch.Tensor,
    U_low_int8: torch.Tensor,
    scale_U_low: torch.Tensor,
    *,
    scale_x: torch.Tensor | None = None,
    gs: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused int8 rotation + int4 quant of low region.

    Inputs:
        x:           [M, K] bf16
        U_low_int8:  [K, n_low] int8
        scale_U_low: [n_low] fp32

    Returns:
        out_packed:  [M, n_low/2] int8
        out_scales:  [M, n_low/gs] bf16
    """
    M, K = x.shape
    K2, n_low = U_low_int8.shape
    assert K == K2
    assert n_low % gs == 0

    if scale_x is None:
        from speedup.kernels.int8_rotation import compute_scale_x
        scale_x = compute_scale_x(x)
    if scale_x.dtype != torch.float32:
        scale_x = scale_x.float().contiguous()
    if scale_U_low.dtype != torch.float32:
        scale_U_low = scale_U_low.float().contiguous()

    out_packed = torch.empty((M, n_low // 2), dtype=torch.int8, device=x.device)
    out_scales = torch.empty((M, n_low // gs), dtype=torch.bfloat16, device=x.device)

    # One block per (BM × GS) tile of the output. We split along M (row tiles)
    # and along the group dimension.
    # Larger BM = better tensor-core utilization. Tuned for K=3072+.
    BM = 128
    BK = 64
    GROUP_M = 8
    grid = (triton.cdiv(M, BM) * triton.cdiv(n_low, gs),)
    _int8_rot_to_int4_kernel[grid](
        x, U_low_int8, scale_x, scale_U_low,
        out_packed, out_scales,
        M, n_low, K,
        x.stride(0), x.stride(1),
        U_low_int8.stride(0), U_low_int8.stride(1),
        out_packed.stride(0), out_packed.stride(1),
        out_scales.stride(0), out_scales.stride(1),
        BM=BM, GS=gs, BK=BK, GROUP_M=GROUP_M,
        num_warps=8, num_stages=3,
    )
    return out_packed, out_scales
