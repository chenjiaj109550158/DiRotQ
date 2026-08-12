"""Independent sequential-FP32, packed-vectorized, and FP64 GEMM references."""

from __future__ import annotations

import numpy as np
import torch

from kernels.blackwell_e0_probe.packing import E0M3_MAGNITUDES, E2M1_MAGNITUDES
from kernels.blackwell_e0_probe.gemm_probe.packing import (
    CanonicalInputs,
    unpack_canonical_a,
    unpack_canonical_a_scales,
    unpack_canonical_b,
    unpack_canonical_b_scales,
    validate_canonical,
)


def _decode(codes: torch.Tensor, fmt: str) -> np.ndarray:
    if fmt not in {"e2m1", "e0m3"}:
        raise ValueError(f"unsupported format {fmt}")
    magnitudes = E2M1_MAGNITUDES if fmt == "e2m1" else E0M3_MAGNITUDES
    lut = np.asarray((*magnitudes, *(-value for value in magnitudes)), dtype=np.float32)
    return lut[codes.cpu().numpy()]


def decoded_operands_and_scales(inputs: CanonicalInputs, a_format: str,
                                b_format: str):
    validate_canonical(inputs)
    shape = inputs.shape
    a_codes = unpack_canonical_a(inputs.packed_a, shape)[:shape.m, :shape.k]
    b_codes = unpack_canonical_b(inputs.packed_b, shape)[:shape.k, :shape.n]
    a = _decode(a_codes, a_format)
    b = _decode(b_codes, b_format)
    a_block = unpack_canonical_a_scales(inputs.a_scales, shape)[:shape.m, :shape.k_blocks]
    b_block = unpack_canonical_b_scales(inputs.b_scales, shape)[:shape.n, :shape.k_blocks]
    a_scale = np.repeat(a_block.numpy(), 16, axis=1)[:, :shape.k].astype(np.float32)
    b_scale = np.repeat(b_block.numpy(), 16, axis=1)[:, :shape.k].T.astype(np.float32)
    return a, b, a_scale, b_scale


def sequential_fp32_gemm(inputs: CanonicalInputs, a_format: str,
                         b_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate K in order with IEEE FP32 elementwise outer products."""
    a, b, a_scale, b_scale = decoded_operands_and_scales(inputs, a_format, b_format)
    av = np.multiply(a, a_scale, dtype=np.float32)
    bv = np.multiply(b, b_scale, dtype=np.float32)
    raw = np.zeros((inputs.shape.m, inputs.shape.n), dtype=np.float32)
    for k in range(inputs.shape.k):
        product = np.multiply(av[:, k, None], bv[k, None, :], dtype=np.float32)
        raw = np.add(raw, product, dtype=np.float32)
    global_product = np.float32(np.float32(inputs.alpha_a) * np.float32(inputs.alpha_b))
    scaled = np.multiply(raw, global_product, dtype=np.float32)
    return torch.from_numpy(raw.copy()), torch.from_numpy(scaled.copy())


def vectorized_packed_gemm(inputs: CanonicalInputs, a_format: str,
                           b_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    a, b, a_scale, b_scale = decoded_operands_and_scales(inputs, a_format, b_format)
    at = torch.from_numpy(np.multiply(a, a_scale, dtype=np.float32))
    bt = torch.from_numpy(np.multiply(b, b_scale, dtype=np.float32))
    raw = torch.matmul(at, bt).float()
    global_product = np.float32(np.float32(inputs.alpha_a) * np.float32(inputs.alpha_b))
    return raw, (raw * float(global_product)).float()


def decoded_fp64_gemm(inputs: CanonicalInputs, a_format: str,
                      b_format: str) -> torch.Tensor:
    a, b, a_scale, b_scale = decoded_operands_and_scales(inputs, a_format, b_format)
    global_product = float(np.float64(inputs.alpha_a) * np.float64(inputs.alpha_b))
    result = ((a.astype(np.float64) * a_scale.astype(np.float64))
              @ (b.astype(np.float64) * b_scale.astype(np.float64)))
    return torch.from_numpy(result * global_product)


def fp32_comparison_bound(inputs: CanonicalInputs, a_format: str,
                          b_format: str, expected: torch.Tensor) -> torch.Tensor:
    a, b, a_scale, b_scale = decoded_operands_and_scales(inputs, a_format, b_format)
    av = np.abs(a.astype(np.float64) * a_scale.astype(np.float64))
    bv = np.abs(b.astype(np.float64) * b_scale.astype(np.float64))
    sum_abs = torch.from_numpy(av @ bv).float()
    unit_roundoff = 2.0 ** -24
    gamma_k = (inputs.shape.k * unit_roundoff) / (1 - inputs.shape.k * unit_roundoff)
    global_product = abs(float(np.float32(np.float32(inputs.alpha_a) * np.float32(inputs.alpha_b))))
    # Two FP32 accumulation schemes versus the exact dot product, followed by
    # one global-product evaluation and one output multiply on each path.
    reduction = 2 * gamma_k * sum_abs * global_product
    scale_rounding = 2 * unit_roundoff * expected.abs()
    return reduction + scale_rounding


__all__ = [
    "decoded_fp64_gemm", "decoded_operands_and_scales", "fp32_comparison_bound",
    "sequential_fp32_gemm", "vectorized_packed_gemm",
]
