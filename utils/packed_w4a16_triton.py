"""Triton W4A16 GEMM for the FLUX adaptive-norm memory experiment.

The persistent B operand is signed two's-complement INT4 with K64 BF16
scales.  The activation remains in its native FP16/BF16 dtype.  Weight
values are decoded only in registers; no dense reconstructed weight is
materialized on CUDA.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _packed_w4a16_kernel(
        a_ptr,
        b_ptr,
        b_scale_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        K_PADDED: tl.constexpr,
        stride_am: tl.constexpr,
        stride_bm: tl.constexpr,
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
            valid_k = absolute_k < K
            byte_k = absolute_k // 2
            shift = (absolute_k & 1) * 4
            activation = tl.load(
                a_ptr + offs_m[:, None] * stride_am + absolute_k[None, :],
                mask=mask_m[:, None] & valid_k[None, :],
                other=0.0,
            )
            packed = tl.load(
                b_ptr + offs_n[:, None] * stride_bm + byte_k[None, :],
                mask=mask_n[:, None],
                other=0,
            )
            code = ((packed >> shift[None, :]) & 0xF).to(tl.int16)
            code = tl.where(code >= 8, code - 16, code)
            group = k0 // BLOCK_K
            scale = tl.load(
                b_scale_ptr + offs_n * stride_bsm + group,
                mask=mask_n,
                other=0.0,
            )
            # Decode to the activation dtype in registers, matching the
            # serialized BF16-scale W4A16 runtime contract.
            decoded = (code.to(tl.float32) * scale[:, None].to(tl.float32)).to(
                activation.dtype
            )
            accumulator += tl.dot(
                activation,
                tl.trans(decoded),
                out_dtype=tl.float32,
            )

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
            accumulator,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def packed_w4a16_gemm(
    activation: torch.Tensor,
    weight_payload: torch.Tensor,
    weight_scales: torch.Tensor,
    logical_k: int,
) -> torch.Tensor:
    """Return FP32 ``activation @ decoded_weight.T`` from packed W4 storage."""
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA packed W4A16 path")
    if any(t.device.type != "cuda" for t in (activation, weight_payload, weight_scales)):
        raise RuntimeError("packed W4A16 CUDA execution forbids CPU fallback")
    if activation.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("W4A16 activation must be FP16 or BF16")
    if weight_payload.dtype != torch.uint8 or weight_scales.dtype != torch.bfloat16:
        raise TypeError("W4A16 requires uint8 payload and BF16 scales")
    if activation.ndim != 2 or weight_payload.ndim != 2 or weight_scales.ndim != 2:
        raise ValueError("W4A16 operands must be two-dimensional")
    if not (activation.is_contiguous() and weight_payload.is_contiguous()):
        raise ValueError("W4A16 activation/payload must be contiguous")
    m, k = activation.shape
    n, k_bytes = weight_payload.shape
    k_padded = k_bytes * 2
    if k != logical_k or k_padded % 64 or weight_scales.shape != (n, k_padded // 64):
        raise ValueError("W4A16 payload/scale/logical-K mismatch")
    output = torch.empty((m, n), dtype=torch.float32, device=activation.device)
    block_m, block_n = 16, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _packed_w4a16_kernel[grid](
        activation,
        weight_payload,
        weight_scales,
        output,
        M=m,
        N=n,
        K=k,
        K_PADDED=k_padded,
        stride_am=activation.stride(0),
        stride_bm=weight_payload.stride(0),
        stride_bsm=weight_scales.stride(0),
        stride_cm=output.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
    return output
