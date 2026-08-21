"""Memory-oriented fused Triton kernels for FLUX shared-PCA real INT4.

The kernels preserve the repository's existing arithmetic contract:

* activations use asymmetric unsigned INT4 with one BF16 scale/zero per K64;
* weights use signed two's-complement INT4 with one frozen scale per K64;
* low partials accumulate in INT32 and are scaled/accumulated in FP32;
* the protected rank-64 BF16 branch is accumulated once, followed by bias;
* the public result is materialized directly as BF16.

This module is deliberately fail-closed.  It has no CPU implementation and is
only selected by the explicit ``--real-int4-kernel-mode fused`` runtime flag.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised by the fail-closed wrappers
    triton = None
    tl = None
    libdevice = None


if triton is not None:

    @triton.jit
    def _quantize_pack_u4_kernel(
        x_ptr,
        payload_ptr,
        scale_ptr,
        zero_ptr,
        M: tl.constexpr,
        LOGICAL_K: tl.constexpr,
        GROUPS: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_pm: tl.constexpr,
        stride_sm: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, GROUP_SIZE)
        absolute_k = pid_g * GROUP_SIZE + offs_k
        valid_m = offs_m < M
        valid_k = absolute_k < LOGICAL_K
        values = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + absolute_k[None, :] * stride_xk,
            mask=valid_m[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float32)

        xmax = tl.max(values, axis=1).to(tl.bfloat16)
        xmin = tl.min(values, axis=1).to(tl.bfloat16)
        zero_group = (xmin == 0.0) & (xmax == 0.0)
        xmin = tl.where(zero_group, -1.0, xmin).to(tl.bfloat16)
        xmax = tl.where(zero_group, 1.0, xmax).to(tl.bfloat16)
        value_range = (xmax - xmin).to(tl.bfloat16)
        value_range = tl.maximum(value_range, 1.0e-5).to(tl.bfloat16)
        scale_f32 = (value_range / 15.0).to(tl.bfloat16)
        # The reference rounds scale and zero to the native FLUX BF16 dtype
        # before code selection.
        scale_bf16 = scale_f32.to(tl.bfloat16)
        zero_input = ((-xmin).to(tl.bfloat16) / scale_bf16).to(tl.bfloat16)
        zero_f32 = libdevice.rint(zero_input.to(tl.float32))
        zero_bf16 = zero_f32.to(tl.bfloat16)
        normalized = (
            values.to(tl.bfloat16) / scale_bf16[:, None]
        ).to(tl.bfloat16)
        rounded = libdevice.rint(normalized.to(tl.float32)).to(tl.bfloat16)
        codes = (rounded + zero_bf16[:, None]).to(tl.bfloat16)
        codes = tl.maximum(0.0, tl.minimum(15.0, codes)).to(tl.uint8)

        pairs = tl.reshape(codes, (BLOCK_M, GROUP_SIZE // 2, 2))
        low, high = tl.split(pairs)
        packed = low | (high << 4)
        byte_k = pid_g * (GROUP_SIZE // 2) + tl.arange(0, GROUP_SIZE // 2)
        tl.store(
            payload_ptr + offs_m[:, None] * stride_pm + byte_k[None, :],
            packed,
            mask=valid_m[:, None],
        )
        tl.store(
            scale_ptr + offs_m * stride_sm + pid_g,
            scale_bf16,
            mask=valid_m,
        )
        tl.store(
            zero_ptr + offs_m * stride_sm + pid_g,
            zero_bf16,
            mask=valid_m,
        )


    @triton.jit
    def _packed_w4a4_high_kernel(
        a_ptr,
        b_ptr,
        a_scale_ptr,
        a_zero_ptr,
        b_scale_ptr,
        high_output_ptr,
        bias_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K_PADDED: tl.constexpr,
        stride_am: tl.constexpr,
        stride_bm: tl.constexpr,
        stride_asm: tl.constexpr,
        stride_bsm: tl.constexpr,
        stride_hom: tl.constexpr,
        stride_cm: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_HIGH: tl.constexpr,
        HAS_BIAS: tl.constexpr,
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
            partial = tl.dot(a_code, tl.trans(b_code.to(tl.int8)), out_dtype=tl.int32)
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

        if HAS_HIGH:
            high = tl.load(
                high_output_ptr + offs_m[:, None] * stride_hom + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += high
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            accumulator += bias[None, :]
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
            accumulator.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _flatten_rows_view(x: torch.Tensor) -> torch.Tensor:
    """Return a 2-D strided view without copying a sliced last dimension."""
    if x.ndim < 2 or x.stride(-1) != 1:
        raise ValueError("fused INT4 activation requires unit last-dimension stride")
    rows = math.prod(x.shape[:-1])
    row_stride = x.stride(-2)
    expected = row_stride
    for dim in range(x.ndim - 2, -1, -1):
        if x.stride(dim) != expected:
            raise ValueError("activation leading dimensions cannot be flattened without a copy")
        expected *= x.shape[dim]
    return x.as_strided((rows, x.shape[-1]), (row_stride, 1))


def quantize_pack_u4(x: torch.Tensor, group_size: int = 64):
    """Fused BF16 asymmetric activation quantization and nibble packing."""
    if triton is None:
        raise RuntimeError("Triton is required for fused INT4 activation packing")
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise RuntimeError("fused INT4 activation packing requires CUDA BF16")
    if group_size != 64:
        raise ValueError("fused INT4 activation packing currently requires K64")
    flat = _flatten_rows_view(x)
    m, logical_k = flat.shape
    padded_k = triton.cdiv(logical_k, group_size) * group_size
    groups = padded_k // group_size
    payload = torch.empty((m, padded_k // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((m, groups), dtype=x.dtype, device=x.device)
    zeros = torch.empty_like(scales)
    block_m = 8
    grid = (triton.cdiv(m, block_m), groups)
    _quantize_pack_u4_kernel[grid](
        flat,
        payload,
        scales,
        zeros,
        M=m,
        LOGICAL_K=logical_k,
        GROUPS=groups,
        stride_xm=flat.stride(0),
        stride_xk=flat.stride(1),
        stride_pm=payload.stride(0),
        stride_sm=scales.stride(0),
        BLOCK_M=block_m,
        GROUP_SIZE=group_size,
        num_warps=4,
        num_stages=1,
    )
    return payload, scales, zeros, logical_k, padded_k


def packed_w4a4_high_gemm(
    a_payload: torch.Tensor,
    b_payload: torch.Tensor,
    a_scales: torch.Tensor,
    a_zeros: torch.Tensor,
    b_scales: torch.Tensor,
    high_output: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Fuse low W4A4, protected BF16, bias and final BF16 materialization."""
    if triton is None:
        raise RuntimeError("Triton is required for fused split INT4 GEMM")
    tensors = (a_payload, b_payload, a_scales, a_zeros, b_scales)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("fused split INT4 GEMM forbids CPU fallback")
    if a_scales.dtype != torch.bfloat16 or a_zeros.dtype != torch.bfloat16:
        raise TypeError("fused split INT4 GEMM requires BF16 activation metadata")
    if a_payload.dtype != torch.uint8 or b_payload.dtype != torch.uint8:
        raise TypeError("packed operands must be uint8")
    m, k_bytes = a_payload.shape
    n, b_k_bytes = b_payload.shape
    if k_bytes != b_k_bytes:
        raise ValueError("packed activation/weight K mismatch")
    k_padded = 2 * k_bytes
    if k_padded % 64:
        raise ValueError("fused split INT4 GEMM requires K64 padding")
    groups = k_padded // 64
    a_scales = a_scales.reshape(m, groups).contiguous()
    a_zeros = a_zeros.reshape(m, groups).contiguous()
    b_scales = b_scales.reshape(n, groups).contiguous()
    has_high = high_output is not None
    if has_high:
        assert high_output is not None
        if high_output.device != a_payload.device:
            raise RuntimeError("protected branch device mismatch")
        if high_output.dtype != torch.bfloat16 or high_output.numel() != m * n:
            raise TypeError("fused protected output must be BF16 [M,N]")
        high_output = high_output.reshape(m, n).contiguous()
    if bias is not None and (bias.device != a_payload.device or bias.shape != (n,)):
        raise ValueError("bias device/shape mismatch")

    # Dummy device pointers keep one static signature; constexpr flags ensure
    # they are never dereferenced for absent branches.
    high_arg = high_output if high_output is not None else a_scales
    bias_arg = bias if bias is not None else b_scales
    output = torch.empty((m, n), dtype=torch.bfloat16, device=a_payload.device)
    block_m, block_n = 32, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _packed_w4a4_high_kernel[grid](
        a_payload,
        b_payload,
        a_scales,
        a_zeros,
        b_scales,
        high_arg,
        bias_arg,
        output,
        M=m,
        N=n,
        K_PADDED=k_padded,
        stride_am=a_payload.stride(0),
        stride_bm=b_payload.stride(0),
        stride_asm=a_scales.stride(0),
        stride_bsm=b_scales.stride(0),
        stride_hom=high_arg.stride(0),
        stride_cm=output.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        HAS_HIGH=has_high,
        HAS_BIAS=bias is not None,
        num_warps=4,
        num_stages=3,
    )
    return output
