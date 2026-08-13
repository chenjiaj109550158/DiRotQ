"""Hardware-faithful references for the native Blackwell activation quantizer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch

from ..packing import E0M3_MAGNITUDES, E2M1_MAGNITUDES


UE4M3_ONE = 0x38
GLOBAL_DIVISOR = 2688.0


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def padded_shape(m: int, k: int) -> tuple[int, int]:
    if m <= 0 or k <= 0:
        raise ValueError("M and K must be positive")
    return ((m + 15) // 16 * 16, (k + 63) // 64 * 64)


def native_scale_size(mp: int, kp: int) -> int:
    if mp <= 0 or mp % 16 or kp <= 0 or kp % 64:
        raise ValueError("native scale shape requires Mp%16==0 and Kp%64==0")
    return ((mp + 127) // 128) * (kp // 64) * 512


def native_scale_offset(row: int, block: int, mp: int, kp: int) -> int:
    """CUTLASS v4.0.0 Sm1xxBlockScaledConfig<16> SFA byte offset."""
    outer = (mp + 127) // 128 * 128
    blocks = kp // 16
    if not 0 <= row < outer or not 0 <= block < blocks:
        raise IndexError("native scale coordinate outside padded atom shape")
    k_tiles = kp // 64
    atom = (row // 128) * k_tiles + block // 4
    within = (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4
    return atom * 512 + within


def e4m3_decode_byte(byte: int) -> float:
    if not 0 <= byte <= 0x7E:
        raise ValueError("UE4M3 byte must be a nonnegative finite E4M3FN value")
    exponent, mantissa = (byte >> 3) & 0xF, byte & 7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


_E4M3_VALUES = tuple(e4m3_decode_byte(byte) for byte in range(0x7F))


def e4m3_encode_scalar(value: float) -> int:
    """Nonnegative E4M3FN SATFINITE conversion with nearest-even ties."""
    value = _f32(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("block scale must be finite and nonnegative")
    if value >= _E4M3_VALUES[-1]:
        return 0x7E
    best = 0
    best_distance = abs(value - _E4M3_VALUES[0])
    for byte, candidate in enumerate(_E4M3_VALUES[1:], 1):
        distance = abs(value - candidate)
        if distance < best_distance or (distance == best_distance and not (byte & 1)):
            best, best_distance = byte, distance
    return best


def canonical_scales_from_native(native: torch.Tensor, mp: int, kp: int) -> torch.Tensor:
    if native.dtype != torch.uint8 or native.ndim != 1 or native.numel() != native_scale_size(mp, kp):
        raise ValueError("native SFA byte array has the wrong dtype or size")
    blocks = kp // 16
    result = torch.empty((mp, blocks), dtype=torch.uint8)
    source = native.cpu()
    for row in range(mp):
        for block in range(blocks):
            result[row, block] = source[native_scale_offset(row, block, mp, kp)]
    return result


def native_scales_from_canonical(canonical: torch.Tensor, mp: int, kp: int) -> torch.Tensor:
    if canonical.dtype != torch.uint8 or tuple(canonical.shape) != (mp, kp // 16):
        raise ValueError("canonical SFA byte array has the wrong dtype or shape")
    outer = (mp + 127) // 128 * 128
    result = torch.full((native_scale_size(mp, kp),), UE4M3_ONE, dtype=torch.uint8)
    for row in range(outer):
        for block in range(kp // 16):
            value = int(canonical[row, block]) if row < mp else UE4M3_ONE
            result[native_scale_offset(row, block, mp, kp)] = value
    return result


def _levels(fmt: str) -> tuple[float, ...]:
    if fmt == "e2m1":
        return E2M1_MAGNITUDES
    if fmt == "e0m3":
        return E0M3_MAGNITUDES
    raise ValueError(f"unsupported activation format {fmt!r}")


@dataclass(frozen=True)
class QuantizedActivation:
    m: int
    k: int
    mp: int
    kp: int
    fmt: str
    alpha: float
    payload: torch.Tensor
    native_scales: torch.Tensor

    @property
    def canonical_scales(self) -> torch.Tensor:
        return canonical_scales_from_native(self.native_scales, self.mp, self.kp)


def _validate_source(source: torch.Tensor) -> None:
    if source.dtype not in (torch.bfloat16, torch.float16) or source.ndim != 2:
        raise ValueError("source must be row-major BF16 or FP16 [M,K]")
    if not source.is_contiguous():
        raise ValueError("source must be contiguous")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("source contains NaN or Inf")
    if source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("source shape must be nonzero")


def scalar_quantize(source: torch.Tensor, fmt: str) -> QuantizedActivation:
    """Readable scalar reference; all arithmetic points are explicit FP32."""
    _validate_source(source)
    levels = _levels(fmt)
    maximum_code = levels[-1]
    m, k = map(int, source.shape)
    mp, kp = padded_shape(m, k)
    values = source.cpu().float()
    amax = max(abs(float(value)) for value in values.view(-1))
    alpha = 1.0 if amax == 0.0 else _f32(_f32(amax) / _f32(GLOBAL_DIVISOR))
    payload = torch.zeros((mp, kp // 2), dtype=torch.uint8)
    canonical = torch.full((mp, kp // 16), UE4M3_ONE, dtype=torch.uint8)
    for row in range(m):
        for block in range((k + 15) // 16):
            begin = block * 16
            block_values = [_f32(float(values[row, index]) / alpha) for index in range(begin, min(begin + 16, k))]
            block_amax = max((abs(value) for value in block_values), default=0.0)
            scale_byte = UE4M3_ONE if block_amax == 0.0 else e4m3_encode_scalar(_f32(block_amax / maximum_code))
            scale = e4m3_decode_byte(scale_byte)
            canonical[row, block] = scale_byte
            codes: list[int] = []
            for value in block_values:
                normalized = math.copysign(math.inf, value) if scale == 0.0 else _f32(value / scale)
                magnitude = abs(normalized)
                best = len(levels) - 1 if magnitude >= levels[-1] else 0
                if best == 0:
                    best_distance = abs(magnitude - levels[0])
                    for index, level in enumerate(levels[1:], 1):
                        distance = abs(magnitude - level)
                        if distance < best_distance:  # ties deliberately keep the lower index
                            best, best_distance = index, distance
                sign = 8 if math.copysign(1.0, normalized) < 0.0 and best != 0 else 0
                codes.append(best | sign)
            codes.extend([0] * (16 - len(codes)))
            for pair in range(8):
                payload[row, begin // 2 + pair] = codes[2 * pair] | (codes[2 * pair + 1] << 4)
    return QuantizedActivation(m, k, mp, kp, fmt, alpha, payload,
                               native_scales_from_canonical(canonical, mp, kp))


def vectorized_quantize(source: torch.Tensor, fmt: str) -> QuantizedActivation:
    """Vectorized reference using the verified PyTorch E4M3FN byte contract."""
    _validate_source(source)
    levels_tuple = _levels(fmt)
    levels = torch.tensor(levels_tuple, dtype=torch.float32)
    m, k = map(int, source.shape)
    mp, kp = padded_shape(m, k)
    values = source.cpu().float()
    amax = float(values.abs().max())
    alpha = 1.0 if amax == 0.0 else _f32(_f32(amax) / _f32(GLOBAL_DIVISOR))
    padded = torch.zeros((mp, kp), dtype=torch.float32)
    padded[:m, :k] = values / alpha
    blocks = padded.reshape(mp, kp // 16, 16)
    block_amax = blocks.abs().amax(-1)
    raw_scale = block_amax / levels_tuple[-1]
    raw_scale[block_amax == 0] = 1.0
    scale_bytes = raw_scale.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    scale_values = scale_bytes.contiguous().view(torch.float8_e4m3fn).float()
    normalized = (blocks / scale_values.unsqueeze(-1)).clamp(
        min=-levels_tuple[-1], max=levels_tuple[-1])
    distances = (normalized.abs().unsqueeze(-1) - levels).abs()
    codes = distances.argmin(-1).to(torch.uint8)
    signs = torch.signbit(normalized).to(torch.uint8) << 3
    codes |= torch.where(codes == 0, torch.zeros_like(signs), signs)
    flat_codes = codes.reshape(mp, kp)
    payload = flat_codes[:, 0::2] | (flat_codes[:, 1::2] << 4)
    return QuantizedActivation(m, k, mp, kp, fmt, alpha, payload.contiguous(),
                               native_scales_from_canonical(scale_bytes, mp, kp))


def decode_quantized(quantized: QuantizedActivation) -> torch.Tensor:
    packed = quantized.payload
    codes = torch.empty((quantized.mp, quantized.kp), dtype=torch.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    levels = torch.tensor(_levels(quantized.fmt), dtype=torch.float32)
    decoded = levels[(codes & 7).long()]
    decoded = torch.where((codes & 8) != 0, -decoded, decoded)
    scale_bytes = quantized.canonical_scales
    scales = scale_bytes.contiguous().view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    return (decoded * scales * quantized.alpha)[:quantized.m, :quantized.k]


__all__ = [
    "GLOBAL_DIVISOR", "QuantizedActivation", "UE4M3_ONE",
    "canonical_scales_from_native", "decode_quantized", "e4m3_decode_byte",
    "e4m3_encode_scalar", "native_scale_offset", "native_scale_size",
    "native_scales_from_canonical", "padded_shape", "scalar_quantize",
    "vectorized_quantize",
]
