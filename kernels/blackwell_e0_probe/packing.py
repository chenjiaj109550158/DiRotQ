"""Logical FP4 tile packing for Blackwell E0/E2 hardware probes.

This is a transparent handoff format, not a claim about undocumented SASS
register-fragment encoding.  ``HARDWARE_ENCODING_TO_VERIFY`` marks that
boundary explicitly.
"""

from __future__ import annotations

import torch


M, N, K = 16, 8, 64
K_BLOCK = 16
A_SHAPE = (M, K)
B_SHAPE = (K, N)
A_SCALE_SHAPE = (M, K // K_BLOCK)
B_SCALE_SHAPE = (N, K // K_BLOCK)
FORMATS = ("e2m1", "e0m3")
HARDWARE_ENCODING_TO_VERIFY = "HARDWARE_ENCODING_TO_VERIFY"

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E0M3_MAGNITUDES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def _magnitudes(fmt: str) -> tuple[float, ...]:
    if fmt == "e2m1":
        return E2M1_MAGNITUDES
    if fmt == "e0m3":
        return E0M3_MAGNITUDES
    raise ValueError(f"unsupported FP4 format {fmt!r}")


def decode_nibbles(codes: torch.Tensor, fmt: str) -> torch.Tensor:
    """Decode the probe's sign-magnitude nibble contract to FP32.

    Bit 3 is sign and bits 2:0 select a magnitude.  E2M1 follows the
    documented E2M1 value map.  E0M3 uses the experiment's logical INT-like
    0..7 magnitude map; its correspondence to SASS bits remains unverified.
    """
    if codes.dtype != torch.uint8 or bool((codes > 15).any()):
        raise ValueError("FP4 codes must be uint8 nibbles in [0, 15]")
    magnitudes = torch.tensor(_magnitudes(fmt), device=codes.device)
    values = magnitudes[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -values, values).float()


def encode_values(values: torch.Tensor, fmt: str) -> torch.Tensor:
    """Encode exactly representable values; this function does not quantize."""
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("FP4 logical values must be finite floating tensors")
    levels = torch.tensor(_magnitudes(fmt), dtype=torch.float32, device=values.device)
    magnitude = values.float().abs()
    matches = magnitude.unsqueeze(-1) == levels
    if not bool(matches.any(-1).all()):
        raise ValueError(f"input contains a value not representable as {fmt}")
    index = matches.float().argmax(-1).to(torch.uint8)
    sign = torch.signbit(values).to(torch.uint8) << 3
    return index | sign


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack a flat even-length stream: first element low, second high."""
    if codes.dtype != torch.uint8 or codes.ndim != 1 or codes.numel() % 2:
        raise ValueError("nibble stream must be flat uint8 with even length")
    if bool((codes > 15).any()):
        raise ValueError("nibble value exceeds 4 bits")
    return codes[0::2] | (codes[1::2] << 4)


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 1 or count < 0:
        raise ValueError("packed payload must be flat uint8")
    if packed.numel() * 2 < count:
        raise ValueError("packed payload is too short")
    output = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    output[0::2] = packed & 0xF
    output[1::2] = packed >> 4
    return output[:count]


def pack_a(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical A[16,64] in row-major (m outer, k inner) order."""
    if tuple(codes.shape) != A_SHAPE:
        raise ValueError(f"A codes must have shape {A_SHAPE}")
    return pack_nibbles(codes.contiguous().view(-1))


def unpack_a(packed: torch.Tensor) -> torch.Tensor:
    return unpack_nibbles(packed, M * K).reshape(A_SHAPE)


def pack_b(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B[64,8] in column-major (n outer, k inner) order."""
    if tuple(codes.shape) != B_SHAPE:
        raise ValueError(f"B codes must have shape {B_SHAPE}")
    return pack_nibbles(codes.T.contiguous().view(-1))


def unpack_b(packed: torch.Tensor) -> torch.Tensor:
    return unpack_nibbles(packed, K * N).reshape(N, K).T.contiguous()


def encode_e4m3_bytes(scales: torch.Tensor) -> torch.Tensor:
    """Round nonnegative scales to E4M3FN and return their literal bytes."""
    if not scales.is_floating_point() or not torch.isfinite(scales).all():
        raise ValueError("E4M3 scales must be finite floating tensors")
    if bool((scales < 0).any()):
        raise ValueError("block scales must be nonnegative")
    # abs() canonicalizes -0.0.  The nonnegative E4M3FN byte range is the
    # software contract used for the public UE4M3 scale operand.
    rounded = scales.float().abs().to(torch.float8_e4m3fn)
    if not torch.isfinite(rounded.float()).all():
        raise OverflowError("E4M3 scale overflow or NaN encoding")
    return rounded.contiguous().view(torch.uint8)


def decode_e4m3_bytes(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.dtype != torch.uint8:
        raise ValueError("E4M3 storage must be uint8")
    if bool((encoded & 0x80).any()):
        raise ValueError("UE4M3 block-scale bytes must be nonnegative")
    decoded = encoded.contiguous().view(torch.float8_e4m3fn).float()
    if not torch.isfinite(decoded).all():
        raise ValueError("E4M3 byte stream contains NaN")
    return decoded


def pack_a_scales(scales: torch.Tensor) -> torch.Tensor:
    """Scale index is m*4 + floor(k/16)."""
    if tuple(scales.shape) != A_SCALE_SHAPE:
        raise ValueError(f"A scales must have shape {A_SCALE_SHAPE}")
    return encode_e4m3_bytes(scales.contiguous().view(-1))


def unpack_a_scales(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.numel() != M * (K // K_BLOCK):
        raise ValueError("A must carry exactly 64 E4M3 scale bytes")
    return decode_e4m3_bytes(encoded).reshape(A_SCALE_SHAPE)


def pack_b_scales(scales: torch.Tensor) -> torch.Tensor:
    """Scale index is n*4 + floor(k/16), matching B column-major order."""
    if tuple(scales.shape) != B_SCALE_SHAPE:
        raise ValueError(f"B scales must have shape {B_SCALE_SHAPE}")
    return encode_e4m3_bytes(scales.contiguous().view(-1))


def unpack_b_scales(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.numel() != N * (K // K_BLOCK):
        raise ValueError("B must carry exactly 32 E4M3 scale bytes")
    return decode_e4m3_bytes(encoded).reshape(B_SCALE_SHAPE)


__all__ = [
    "A_SCALE_SHAPE", "A_SHAPE", "B_SCALE_SHAPE", "B_SHAPE", "E0M3_MAGNITUDES",
    "E2M1_MAGNITUDES", "FORMATS", "HARDWARE_ENCODING_TO_VERIFY", "K", "K_BLOCK",
    "M", "N", "decode_e4m3_bytes", "decode_nibbles", "encode_e4m3_bytes",
    "encode_values", "pack_a", "pack_a_scales", "pack_b", "pack_b_scales",
    "pack_nibbles", "unpack_a", "unpack_a_scales", "unpack_b", "unpack_b_scales",
    "unpack_nibbles",
]
