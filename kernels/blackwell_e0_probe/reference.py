"""Independent scalar and vectorized references for one FP4 16x8x64 MMA."""

from __future__ import annotations

import struct

import torch

from .packing import (
    A_SCALE_SHAPE, A_SHAPE, B_SCALE_SHAPE, B_SHAPE, E0M3_MAGNITUDES,
    E2M1_MAGNITUDES, FORMATS, K, K_BLOCK, M, N, decode_nibbles,
    unpack_a, unpack_a_scales, unpack_b, unpack_b_scales,
)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _scalar_nibble(byte: int, element_index: int) -> int:
    return (byte >> (4 * (element_index & 1))) & 0xF


def _scalar_fp4(code: int, fmt: str) -> float:
    levels = E2M1_MAGNITUDES if fmt == "e2m1" else E0M3_MAGNITUDES
    if fmt not in FORMATS or not 0 <= code <= 15:
        raise ValueError("invalid scalar FP4 operand")
    value = levels[code & 7]
    return -value if code & 8 else value


def _scalar_e4m3(byte: int) -> float:
    """Decode the finite E4M3FN byte format without using PyTorch float8."""
    if byte & 0x80:
        raise ValueError("UE4M3 block-scale byte must be nonnegative")
    exponent, mantissa = (byte >> 3) & 0xF, byte & 0x7
    if exponent == 0:
        value = mantissa * (2.0 ** -9)
    else:
        if exponent == 0xF and mantissa == 0x7:
            raise ValueError("E4M3 NaN byte is invalid for a block scale")
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return value


def scalar_mma(
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    a_scale_bytes: torch.Tensor,
    b_scale_bytes: torch.Tensor,
    a_format: str,
    b_format: str,
    a_global_scale: float,
    b_global_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Readable, sequential FP32 accumulator used to create golden output."""
    pa, pb = packed_a.cpu().tolist(), packed_b.cpu().tolist()
    sa, sb = a_scale_bytes.cpu().tolist(), b_scale_bytes.cpu().tolist()
    if len(pa) != M*K//2 or len(pb) != K*N//2:
        raise ValueError("packed operand length does not match m16n8k64")
    if len(sa) != A_SCALE_SHAPE[0]*A_SCALE_SHAPE[1] or len(sb) != B_SCALE_SHAPE[0]*B_SCALE_SHAPE[1]:
        raise ValueError("scale byte count does not match m16n8k64")
    raw = torch.empty(M, N, dtype=torch.float32)
    for m in range(M):
        for n in range(N):
            accumulator = 0.0
            for k in range(K):
                a_linear = m*K + k
                b_linear = n*K + k  # B is serialized column-major.
                av = _scalar_fp4(_scalar_nibble(pa[a_linear//2], a_linear), a_format)
                bv = _scalar_fp4(_scalar_nibble(pb[b_linear//2], b_linear), b_format)
                av = _f32(av * _scalar_e4m3(sa[m*4 + k//K_BLOCK]))
                bv = _f32(bv * _scalar_e4m3(sb[n*4 + k//K_BLOCK]))
                accumulator = _f32(accumulator + _f32(av * bv))
            raw[m, n] = accumulator
    global_product = _f32(_f32(a_global_scale) * _f32(b_global_scale))
    fully_scaled = (raw * global_product).float()
    return raw, fully_scaled


def vectorized_mma(
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    a_scale_bytes: torch.Tensor,
    b_scale_bytes: torch.Tensor,
    a_format: str,
    b_format: str,
    a_global_scale: float,
    b_global_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference usable on CPU or CUDA; accumulator is FP32."""
    a_codes, b_codes = unpack_a(packed_a), unpack_b(packed_b)
    a = decode_nibbles(a_codes, a_format)
    b = decode_nibbles(b_codes, b_format)
    a_scales = unpack_a_scales(a_scale_bytes).repeat_interleave(K_BLOCK, dim=1)
    b_scales = unpack_b_scales(b_scale_bytes).repeat_interleave(K_BLOCK, dim=1).T
    raw = torch.matmul(a.float() * a_scales, b.float() * b_scales).float()
    fully_scaled = raw * float(a_global_scale) * float(b_global_scale)
    return raw, fully_scaled.float()


def decoded_operands(
    packed_a: torch.Tensor, packed_b: torch.Tensor, a_format: str, b_format: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return decode_nibbles(unpack_a(packed_a), a_format), decode_nibbles(unpack_b(packed_b), b_format)


__all__ = ["decoded_operands", "scalar_mma", "vectorized_mma"]
