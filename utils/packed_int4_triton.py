"""Triton packed W4A4 GEMM used by the FLUX real-quant accuracy audit.

The kernel reads two K-ordered nibbles per byte directly.  Activations are
unsigned INT4 with a per-row/group zero point; weights are signed two's-
complement INT4.  Each K64 partial dot is accumulated as INT32, scaled by the
matching activation/weight scales in FP32, then accumulated into FP32 C.

This is a correctness and memory kernel, not a latency-optimized Nunchaku
replacement.  It intentionally has no software fallback on CUDA failures.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the fail-closed wrapper
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _packed_w4a4_kernel(
        a_ptr,
        b_ptr,
        a_scale_ptr,
        a_zero_ptr,
        b_scale_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K_PADDED: tl.constexpr,
        stride_am: tl.constexpr,
        stride_bm: tl.constexpr,
        stride_asm: tl.constexpr,
        stride_bsm: tl.constexpr,
        stride_cm: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K_PADDED, BLOCK_K):
            absolute_k = k0 + offs_k
            byte_k = absolute_k // 2
            shift = (absolute_k & 1) * 4
            a_byte = tl.load(
                a_ptr + offs_m[:, None] * stride_am + byte_k[None, :],
                mask=mask_m[:, None],
                other=0,
            )
            b_byte = tl.load(
                b_ptr + offs_n[:, None] * stride_bm + byte_k[None, :],
                mask=mask_n[:, None],
                other=0,
            )
            a_code = ((a_byte >> shift[None, :]) & 0xF).to(tl.int16)
            b_code = ((b_byte >> shift[None, :]) & 0xF).to(tl.int16)
            b_code = tl.where(b_code >= 8, b_code - 16, b_code)
            group = k0 // BLOCK_K
            a_zero = tl.load(
                a_zero_ptr + offs_m * stride_asm + group,
                mask=mask_m,
                other=0,
            ).to(tl.int16)
            a_code = (a_code - a_zero[:, None]).to(tl.int8)
            b_code = b_code.to(tl.int8)
            partial = tl.dot(a_code, tl.trans(b_code), out_dtype=tl.int32)
            a_scale = tl.load(
                a_scale_ptr + offs_m * stride_asm + group,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            b_scale = tl.load(
                b_scale_ptr + offs_n * stride_bsm + group,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            accumulator += partial.to(tl.float32) * a_scale[:, None] * b_scale[None, :]

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
            accumulator,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def packed_w4a4_gemm(
    a_payload: torch.Tensor,
    b_payload: torch.Tensor,
    a_scales: torch.Tensor,
    a_zeros: torch.Tensor,
    b_scales: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 ``A @ B.T`` without materializing either dense operand."""
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA packed INT4 path")
    tensors = (a_payload, b_payload, a_scales, a_zeros, b_scales)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("packed Triton GEMM forbids CPU fallback")
    if a_payload.dtype != torch.uint8 or b_payload.dtype != torch.uint8:
        raise TypeError("packed operands must be uint8")
    if not (a_payload.is_contiguous() and b_payload.is_contiguous()):
        raise ValueError("packed operands must be contiguous")
    m, k_bytes = a_payload.shape
    n, b_k_bytes = b_payload.shape
    if k_bytes != b_k_bytes:
        raise ValueError("packed activation/weight K mismatch")
    k_padded = k_bytes * 2
    if k_padded % 64:
        raise ValueError("packed FLUX INT4 K must be padded to K64")
    groups = k_padded // 64
    a_scales = a_scales.reshape(m, groups).contiguous()
    a_zeros = a_zeros.reshape(m, groups).contiguous()
    b_scales = b_scales.reshape(n, groups).contiguous()
    output = torch.empty((m, n), dtype=torch.float32, device=a_payload.device)
    block_m, block_n = 32, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _packed_w4a4_kernel[grid](
        a_payload,
        b_payload,
        a_scales,
        a_zeros,
        b_scales,
        output,
        M=m,
        N=n,
        K_PADDED=k_padded,
        stride_am=a_payload.stride(0),
        stride_bm=b_payload.stride(0),
        stride_asm=a_scales.stride(0),
        stride_bsm=b_scales.stride(0),
        stride_cm=output.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
    return output

