"""
speedup/kernels/int8_rotation.py

Triton kernel: bf16 activation × int8 weight matmul, fused with bf16 cast on output.

Used to replace the cuBLAS bf16 matmul that DiRotQ uses for the rotation step
`x @ U`. The weight U is pre-quantized to int8 at calibration (per-column scale),
so at runtime we read int8 weight (4× less BW) and run int8 mma (theoretical
4× compute) but write bf16 (so we don't pay the int32 output BW penalty).

The activation x is also int8-quantized in-kernel from its bf16 source. A
per-row scale is needed; we compute it once outside the kernel from the
input activation.

This kernel only wins above some K-threshold. On RTX 4090:
    K=1152 → ~tied with cuBLAS bf16 (kernel too small to leave fp16 peak)
    K=3072 → ~2× faster
    K=12288 → ~2.5×+ faster

Two variants:
    - `int8_rotation_forward`: x [M, K] bf16 + U [K, N] int8 → y [M, N] bf16
        (uses precomputed per-row scale_x and per-column scale_U)
    - `int8_rotation_with_scale_compute`: same but computes scale_x in-kernel

Internal kernel takes scale_a/scale_b as POINTERS to fp32 tensors so they're
loaded inside the kernel (Triton 3.2 doesn't accept scalar tensors as kernel
args in arithmetic context — must `tl.load` them).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_rotation_kernel(
    X_ptr,          # [M, K] bf16  (activation, will be int8-quantized in-kernel)
    U_ptr,          # [K, N] int8  (precomputed)
    SCALE_X_ptr,    # [M] fp32     (per-row scale of x)
    SCALE_U_ptr,    # [N] fp32     (per-col scale of U)
    Y_ptr,          # [M, N] bf16
    M, N, K,
    sx_m, sx_k,
    su_k, su_n,
    sy_m, sy_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Block-tile int8 mma with fp32 dequant + bf16 output."""
    # L2-friendly program ID swizzle
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Load per-row x scale and per-column U scale (live in registers throughout)
    scale_x_row = tl.load(SCALE_X_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    scale_u_col = tl.load(SCALE_U_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    # int32 mma accumulator
    acc = tl.zeros((BM, BN), dtype=tl.int32)

    # Tile across K
    offs_k_init = tl.arange(0, BK)
    x_ptrs = X_ptr + offs_m[:, None] * sx_m + offs_k_init[None, :] * sx_k
    u_ptrs = U_ptr + offs_k_init[:, None] * su_k + offs_n[None, :] * su_n

    for k_off in range(0, tl.cdiv(K, BK)):
        # K-mask for the tail tile
        k_mask = (k_off * BK + offs_k_init) < K

        # Load x as bf16, quantize to int8 in-register using per-row scale
        x_bf = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Quantize: codes = round(x / scale_x) clamped to [-128, 127]
        x_codes_f = x_bf.to(tl.float32) / scale_x_row[:, None]
        x_codes = tl.where(
            x_codes_f >= 0,
            (x_codes_f + 0.5).to(tl.int32),
            (x_codes_f - 0.5).to(tl.int32),
        )
        x_codes = tl.maximum(tl.minimum(x_codes, 127), -128).to(tl.int8)

        # Load U as int8 (precomputed)
        u_int8 = tl.load(u_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0).to(tl.int8)

        # int8 mma → int32
        acc = tl.dot(x_codes, u_int8, acc, out_dtype=tl.int32)

        x_ptrs += BK * sx_k
        u_ptrs += BK * su_k

    # Apply dequant scales + cast to bf16
    out_fp32 = acc.to(tl.float32) * scale_x_row[:, None] * scale_u_col[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n
    tl.store(y_ptrs, out_fp32.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def quantize_U_int8(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize U [K, N] bf16 to (U_int8, scale_U) — per-column scale.

    Done once at calibration. scale_U is fp32 to be compatible with kernel.
    """
    assert U.dim() == 2
    max_per_col = U.float().abs().amax(dim=0).clamp(min=1e-9)
    scale_U = (max_per_col / 127.0).contiguous()
    U_int8 = (U.float() / scale_U).round().clamp(-128, 127).to(torch.int8).contiguous()
    return U_int8, scale_U


def compute_scale_x(x: torch.Tensor) -> torch.Tensor:
    """Per-row scale for int8 quantization of activation x [M, K].

    Performance note: stay in bf16 for the abs/amax reductions to avoid
    materializing a full fp32 copy of x. The fp32 cast happens only on the
    [M] result tensor, which is tiny.
    """
    return (x.abs().amax(dim=1).float() / 127.0).clamp(min=1e-9).contiguous()


@triton.jit
def _fast_scale_x_kernel(
    X_ptr, OUT_ptr, M, K,
    sx_m, sx_k,
    BM: tl.constexpr, BK: tl.constexpr,
):
    """One pass over x to compute per-row max(|x|)/127 directly into fp32.

    Replaces 5 separate PyTorch ops (.abs() / .amax() / .float() / .div() /
    .clamp()) which each launch a kernel and read/write x. This kernel
    reads x once and writes a [M] fp32 output. About 3-5× faster on big x.
    """
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    m_mask = offs_m < M

    max_abs = tl.zeros((BM,), dtype=tl.float32)
    for k_off in range(0, tl.cdiv(K, BK)):
        offs_k = k_off * BK + tl.arange(0, BK)
        x = tl.load(X_ptr + offs_m[:, None] * sx_m + offs_k[None, :] * sx_k,
                    mask=m_mask[:, None] & (offs_k < K)[None, :], other=0.0)
        chunk_max = tl.max(tl.abs(x.to(tl.float32)), axis=1)
        max_abs = tl.maximum(max_abs, chunk_max)
    scale = tl.maximum(max_abs / 127.0, 1e-9)
    tl.store(OUT_ptr + offs_m, scale, mask=m_mask)


def compute_scale_x_fast(x: torch.Tensor) -> torch.Tensor:
    """Fast Triton-based per-row scale_x. Replaces the 5-op PyTorch path."""
    M, K = x.shape
    out = torch.empty(M, dtype=torch.float32, device=x.device)
    BM = 32
    BK = 256
    grid = (triton.cdiv(M, BM),)
    _fast_scale_x_kernel[grid](x, out, M, K,
        x.stride(0), x.stride(1),
        BM=BM, BK=BK, num_warps=4)
    return out


# Tile picks — populated empirically for common shapes. Falls back to default.
_TILE_TABLE = {
    # (K, N): (BM, BN, BK, num_warps, num_stages)
    # flux-dev rotation shapes
    (3072, 3072):  (128, 256, 64, 8, 3),  # attn QKV / out
    (3072, 12288): (128, 256, 64, 8, 3),  # ff_up / proj_mlp
    (12288, 3072): (128, 256, 64, 8, 3),  # ff_down / proj_out_mlp (1.32× over bf16)
    (3072, 2688):  (128, 256, 64, 8, 3),
    (3072, 9216):  (128, 256, 64, 8, 3),
    (12288, 1536): (128, 256, 64, 8, 3),
    (4608, 4032):  (128, 256, 64, 8, 3),
    # pixart-sigma (no win, kept for completeness)
    (1152, 960):  (128, 256, 32, 8, 3),
    (1152, 1152): (128, 256, 32, 8, 3),
    (1152, 4608): (128, 256, 64, 8, 3),
}


def int8_rotation_forward(
    x: torch.Tensor,
    U_int8: torch.Tensor,
    scale_U: torch.Tensor,
    *,
    scale_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run int8 mma for `y = x @ U` with bf16 output.

    Args:
        x:        [M, K] bf16
        U_int8:   [K, N] int8
        scale_U:  [N] fp32 (per-column)
        scale_x:  [M] fp32 (per-row), optional — computed if None

    Returns y [M, N] bf16.
    """
    assert x.dim() == 2 and U_int8.dim() == 2
    M, K = x.shape
    K2, N = U_int8.shape
    assert K == K2

    if scale_x is None:
        # Triton kernel wins above ~4K K (memory-bound regime); below that
        # the multiple-op PyTorch path is fine.
        if x.shape[1] >= 4096:
            scale_x = compute_scale_x_fast(x)
        else:
            scale_x = compute_scale_x(x)
    if scale_x.dtype != torch.float32:
        scale_x = scale_x.float().contiguous()
    if scale_U.dtype != torch.float32:
        scale_U = scale_U.float().contiguous()

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    BM, BN, BK, nw, ns = _TILE_TABLE.get((K, N), (128, 128, 64, 8, 3))
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    _int8_rotation_kernel[grid](
        x, U_int8, scale_x, scale_U, y,
        M, N, K,
        x.stride(0), x.stride(1),
        U_int8.stride(0), U_int8.stride(1),
        y.stride(0), y.stride(1),
        BM=BM, BN=BN, BK=BK, GROUP_M=8,
        num_warps=nw, num_stages=ns,
    )
    return y
