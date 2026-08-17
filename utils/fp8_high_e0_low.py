"""References for FP8-high / hardware-E0-low DiRotQ experiments.

This module intentionally contains no custom kernel.  Plain E4M3 uses the
public PyTorch ``float8_e4m3fn`` encoding and MXFP8 uses a transparent
software representation: signed E4M3 payload plus one UE8M0 scale per K32
block.  The reconstructed tensors are suitable for accuracy-only fake-quant
runs on Ada; they are not evidence of native E0M3 or MXFP8 execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

import torch


HighFormat = Literal["bf16", "e4m3", "mxfp8"]
E4M3_MAX = 448.0
MX_BLOCK_SIZE = 32
MX_SCALE_BIAS = 127
MX_MIN_SCALE_EXP = -127
MX_MAX_SCALE_EXP = 127


@dataclass(frozen=True)
class RankContract:
    hidden_dim: int
    head_dim: int
    num_heads: int
    high_hidden: int
    low_hidden: int
    high_per_head: int
    low_per_head: int


@dataclass
class PlainE4M3Result:
    payload: torch.Tensor
    scale: torch.Tensor
    decoded_fp32: torch.Tensor
    reconstructed: torch.Tensor
    saturation_count: int


@dataclass
class MXFP8Result:
    payload: torch.Tensor
    scale_bytes: torch.Tensor
    decoded_fp32: torch.Tensor
    reconstructed: torch.Tensor
    original_shape: tuple[int, ...]
    padded_k: int
    saturation_count: int


class HighFormatStats:
    """Device-side aggregate high-activation reconstruction statistics."""

    def __init__(self):
        self.square_error = None
        self.square_source = None
        self.elements = None
        self.calls = None
        self.saturation_count = None
        self.scale_count = None
        self.scale_sum = None
        self.scale_square_sum = None
        self.scale_min = None
        self.scale_max = None
        # Integer log2 bins [-127, 127].  For MXFP8 these are exactly the
        # decoded UE8M0 exponents; for plain E4M3 they summarize the per-call
        # FP32 global scale without changing its value.
        self.scale_log2_histogram = None

    @torch.no_grad()
    def observe(self, source: torch.Tensor, reconstructed: torch.Tensor, fmt: str) -> None:
        if source.device != reconstructed.device:
            raise RuntimeError("high-format stats forbid CPU/device fallback")
        error = (source.float() - reconstructed.float()).square().sum(dtype=torch.float64)
        energy = source.float().square().sum(dtype=torch.float64)
        count = torch.tensor(source.numel(), dtype=torch.int64, device=source.device)
        one = torch.ones((), dtype=torch.int64, device=source.device)
        if fmt == "e4m3":
            amax = source.float().abs().amax()
            scales = torch.where(amax == 0, torch.ones_like(amax), amax / E4M3_MAX).reshape(1)
            normalized = source.float() / scales[0]
        elif fmt == "mxfp8":
            k = source.shape[-1]
            padded_k = ((k + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE) * MX_BLOCK_SIZE
            flat = source.float().reshape(-1, k)
            padded = torch.nn.functional.pad(flat, (0, padded_k - k))
            blocks = padded.reshape(flat.shape[0], -1, MX_BLOCK_SIZE)
            scale_bytes = _mx_scale_bytes(blocks.abs().amax(dim=-1))
            scales = decode_ue8m0(scale_bytes)
            normalized = blocks / scales.unsqueeze(-1)
        elif fmt == "bf16":
            scales = torch.ones(1, dtype=torch.float32, device=source.device)
            normalized = torch.zeros(1, dtype=torch.float32, device=source.device)
        else:
            raise ValueError(f"unsupported high-format statistics mode: {fmt}")
        saturation = (normalized.abs() > E4M3_MAX).sum(dtype=torch.int64)
        scale_values = scales.float().reshape(-1)
        scale_logs = torch.floor(torch.log2(scale_values)).clamp(-127, 127).to(torch.int64)
        scale_histogram = torch.bincount(scale_logs + 127, minlength=255)
        scale_count = torch.tensor(scale_values.numel(), dtype=torch.int64, device=source.device)
        scale_sum = scale_values.sum(dtype=torch.float64)
        scale_square_sum = scale_values.square().sum(dtype=torch.float64)
        scale_min, scale_max = scale_values.amin(), scale_values.amax()
        if self.square_error is None:
            self.square_error, self.square_source = error, energy
            self.elements, self.calls = count, one
            self.saturation_count = saturation
            self.scale_count = scale_count
            self.scale_sum, self.scale_square_sum = scale_sum, scale_square_sum
            self.scale_min, self.scale_max = scale_min, scale_max
            self.scale_log2_histogram = scale_histogram
        else:
            self.square_error.add_(error)
            self.square_source.add_(energy)
            self.elements.add_(count)
            self.calls.add_(one)
            self.saturation_count.add_(saturation)
            self.scale_count.add_(scale_count)
            self.scale_sum.add_(scale_sum)
            self.scale_square_sum.add_(scale_square_sum)
            self.scale_min.copy_(torch.minimum(self.scale_min, scale_min))
            self.scale_max.copy_(torch.maximum(self.scale_max, scale_max))
            self.scale_log2_histogram.add_(scale_histogram)

    def snapshot(self) -> dict:
        if self.square_error is None:
            return {"calls": 0, "elements": 0, "sse": 0.0, "relative_sse": 0.0}
        sse = float(self.square_error.cpu())
        energy = float(self.square_source.cpu())
        scale_count = int(self.scale_count.cpu())
        scale_mean = float(self.scale_sum.cpu()) / scale_count
        scale_variance = max(
            0.0, float(self.scale_square_sum.cpu()) / scale_count - scale_mean ** 2
        )
        histogram = self.scale_log2_histogram.cpu().tolist()
        return {
            "calls": int(self.calls.cpu()), "elements": int(self.elements.cpu()),
            "sse": sse, "source_square_sum": energy,
            "relative_sse": sse / energy if energy else 0.0,
            "saturation_count": int(self.saturation_count.cpu()),
            "saturation_rate": int(self.saturation_count.cpu()) / int(self.elements.cpu()),
            "scale_count": scale_count,
            "scale_min": float(self.scale_min.cpu()),
            "scale_max": float(self.scale_max.cpu()),
            "scale_mean": scale_mean,
            "scale_std": math.sqrt(scale_variance),
            "scale_log2_histogram": {
                str(index - 127): int(value)
                for index, value in enumerate(histogram) if value
            },
        }


def derive_rank_contract(
    *, hidden_dim: int, head_dim: int, num_heads: int,
    high_fraction: float, multiplier: int = 1, group_size: int = 16,
) -> RankContract:
    """Derive effective ranks from the production SANA configuration.

    The flat hidden rank is first rounded exactly like SANA activation setup
    (up to the K16 activation group) and only then multiplied.  The per-head
    rank is never globally rounded, preserving head isolation.
    """
    if multiplier < 1:
        raise ValueError("rank multiplier must be positive")
    raw_hidden = round(high_fraction * hidden_dim)
    base_hidden = ((raw_hidden + group_size - 1) // group_size) * group_size
    base_head = round(high_fraction * head_dim)
    high_hidden = multiplier * base_hidden
    high_head = multiplier * base_head
    if high_hidden >= hidden_dim or high_head >= head_dim:
        raise ValueError("high rank leaves no low residual subspace")
    return RankContract(
        hidden_dim=hidden_dim,
        head_dim=head_dim,
        num_heads=num_heads,
        high_hidden=high_hidden,
        low_hidden=hidden_dim - high_hidden,
        high_per_head=high_head,
        low_per_head=head_dim - high_head,
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _production_random_orthogonal(size: int, device: torch.device) -> torch.Tensor:
    """Exact QR/sign convention used by ``random_orthogonal_matrix``."""
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def generate_matched_residual_rotations(
    contract: RankContract, *, seed: int = 42, device: str | torch.device = "cpu",
) -> dict:
    """Regenerate rank-matched residual R using production algorithm/order.

    Production draws hidden-low first and head-low second after one root seed.
    Only low residual matrices are randomized; protected directions are an
    identity tail.  The intermediate R_down draw is deliberately omitted
    because SANA's active wrapped linears do not use it and it occurs after
    both relevant draws, so omission cannot change R1/R2.
    """
    device = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        r1_low = _production_random_orthogonal(contract.low_hidden, device)
        r2_low = _production_random_orthogonal(contract.low_per_head, device)
    r1 = torch.block_diag(
        r1_low, torch.eye(contract.high_hidden, dtype=torch.float64, device=device)
    )
    r2 = torch.block_diag(
        r2_low, torch.eye(contract.high_per_head, dtype=torch.float64, device=device)
    )
    return {
        "R1": r1.cpu(),
        "R2": r2.cpu(),
        "R1_low": r1_low.cpu(),
        "R2_low": r2_low.cpu(),
        "seed": seed,
        "algorithm": "torch-float64-gaussian-qr-diag-sign;hidden-low-then-head-low",
        "contract": contract.__dict__,
    }


def validate_residual_rotation(rotation: dict, contract: RankContract, atol: float = 1e-10) -> dict:
    r1 = rotation["R1"].double()
    r2 = rotation["R2"].double()
    if r1.shape != (contract.hidden_dim, contract.hidden_dim):
        raise ValueError("R1 shape does not match rank contract")
    if r2.shape != (contract.head_dim, contract.head_dim):
        raise ValueError("R2 shape does not match rank contract")
    error1 = float((r1.T @ r1 - torch.eye(contract.hidden_dim)).abs().max())
    error2 = float((r2.T @ r2 - torch.eye(contract.head_dim)).abs().max())
    if error1 > atol or error2 > atol:
        raise ValueError(f"residual rotation is not orthogonal: R1={error1}, R2={error2}")
    # The protected tail must be identity and isolated from residual channels.
    low1, low2 = contract.low_hidden, contract.low_per_head
    tail1 = r1[low1:, low1:]
    tail2 = r2[low2:, low2:]
    isolation = max(
        float(r1[:low1, low1:].abs().max()),
        float(r1[low1:, :low1].abs().max()),
        float(r2[:low2, low2:].abs().max()),
        float(r2[low2:, :low2].abs().max()),
    )
    identity_error = max(
        float((tail1 - torch.eye(contract.high_hidden)).abs().max()),
        float((tail2 - torch.eye(contract.high_per_head)).abs().max()),
    )
    if isolation > atol or identity_error > atol:
        raise ValueError("protected tail is not an isolated identity block")
    return {
        "R1_orthogonality_max_abs": error1,
        "R2_orthogonality_max_abs": error2,
        "tail_isolation_max_abs": isolation,
        "tail_identity_max_abs": identity_error,
        "R1_sha256": tensor_sha256(r1),
        "R2_sha256": tensor_sha256(r2),
    }


def _canonicalize_e4m3_zero(payload: torch.Tensor) -> torch.Tensor:
    if payload.dtype != torch.uint8:
        raise TypeError("E4M3 payload must be uint8")
    return torch.where(payload == 0x80, torch.zeros_like(payload), payload)


def decode_e4m3_bytes(payload: torch.Tensor) -> torch.Tensor:
    """Independent scalar-definition decoder for finite E4M3FN bytes."""
    if payload.dtype != torch.uint8:
        raise TypeError("E4M3 payload must be uint8")
    raw = payload.to(torch.int16)
    sign = torch.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    subnormal = exponent == 0
    value = torch.where(
        subnormal,
        mantissa.float() * (2.0 ** -9),
        (1.0 + mantissa.float() / 8.0)
        * torch.pow(torch.tensor(2.0, device=payload.device), exponent.float() - 7.0),
    )
    invalid = (exponent == 0x0F) & (mantissa == 0x07)
    value = value * sign
    value = torch.where(invalid, torch.full_like(value, float("nan")), value)
    return value


def _e4m3_payload_from_scaled(scaled: torch.Tensor) -> tuple[torch.Tensor, int]:
    finite = torch.nan_to_num(scaled.float(), nan=0.0, posinf=E4M3_MAX, neginf=-E4M3_MAX)
    saturation_count = int((finite.abs() > E4M3_MAX).sum().item())
    finite = finite.clamp(-E4M3_MAX, E4M3_MAX).contiguous()
    encoded = finite.to(torch.float8_e4m3fn).view(torch.uint8)
    return _canonicalize_e4m3_zero(encoded), saturation_count


def quantize_plain_e4m3(source: torch.Tensor) -> PlainE4M3Result:
    """One-full-call/tensor-scale signed E4M3 fake quantization."""
    if not source.is_floating_point() or not torch.isfinite(source).all():
        raise ValueError("plain E4M3 source must be a finite floating tensor")
    amax = source.float().abs().amax()
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / E4M3_MAX).float()
    payload, saturation_count = _e4m3_payload_from_scaled(source.float() / scale)
    decoded = decode_e4m3_bytes(payload)
    reconstructed_fp32 = decoded * scale
    return PlainE4M3Result(
        payload=payload,
        scale=scale,
        decoded_fp32=decoded,
        reconstructed=reconstructed_fp32.to(source.dtype),
        saturation_count=saturation_count,
    )


def decode_ue8m0(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode finite UE8M0/E8M0 bytes; 0xff is the NaN encoding."""
    if scale_bytes.dtype != torch.uint8:
        raise TypeError("UE8M0 scales must be uint8")
    invalid = scale_bytes == 0xFF
    exponent = scale_bytes.to(torch.int16) - MX_SCALE_BIAS
    decoded = torch.pow(
        torch.tensor(2.0, device=scale_bytes.device), exponent.float()
    )
    return torch.where(invalid, torch.full_like(decoded, float("nan")), decoded)


def _mx_scale_bytes(block_amax: torch.Tensor) -> torch.Tensor:
    # OCP MX v1.0 conversion minimum semantics: the largest power of two not
    # exceeding block amax, divided by the largest power-of-two E4M3 value
    # (256).  Zero blocks use the canonical scale one.
    nonzero = block_amax > 0
    exponent = torch.floor(torch.log2(torch.where(
        nonzero, block_amax, torch.ones_like(block_amax)
    ))) - 8.0
    exponent = exponent.clamp(MX_MIN_SCALE_EXP, MX_MAX_SCALE_EXP).to(torch.int16)
    exponent = torch.where(nonzero, exponent, torch.zeros_like(exponent))
    return (exponent + MX_SCALE_BIAS).to(torch.uint8)


def quantize_mxfp8_e4m3(source: torch.Tensor) -> MXFP8Result:
    """Canonical row-major software MXFP8-E4M3 reference (K32)."""
    if not source.is_floating_point() or not torch.isfinite(source).all():
        raise ValueError("MXFP8 source must be a finite floating tensor")
    if source.ndim < 1:
        raise ValueError("MXFP8 requires a K dimension")
    original_shape = tuple(source.shape)
    k = source.shape[-1]
    padded_k = ((k + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE) * MX_BLOCK_SIZE
    pad = padded_k - k
    flat = source.float().reshape(-1, k)
    padded = torch.nn.functional.pad(flat, (0, pad)) if pad else flat
    blocks = padded.reshape(flat.shape[0], -1, MX_BLOCK_SIZE)
    scale_bytes = _mx_scale_bytes(blocks.abs().amax(dim=-1))
    scales = decode_ue8m0(scale_bytes)
    payload, saturation_count = _e4m3_payload_from_scaled(
        blocks / scales.unsqueeze(-1)
    )
    decoded_blocks = decode_e4m3_bytes(payload) * scales.unsqueeze(-1)
    decoded_flat = decoded_blocks.reshape(flat.shape[0], padded_k)[:, :k]
    decoded_fp32 = decoded_flat.reshape(original_shape)
    return MXFP8Result(
        payload=payload.reshape(flat.shape[0], padded_k),
        scale_bytes=scale_bytes,
        decoded_fp32=decoded_fp32,
        reconstructed=decoded_fp32.to(source.dtype),
        original_shape=original_shape,
        padded_k=padded_k,
        saturation_count=saturation_count,
    )


def scalar_e4m3_encode(value: float) -> int:
    """Readable independent RN-even/SATFINITE E4M3 scalar encoder."""
    if not math.isfinite(value):
        value = 0.0 if math.isnan(value) else math.copysign(E4M3_MAX, value)
    value = max(-E4M3_MAX, min(E4M3_MAX, value))
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    candidates: list[tuple[float, int]] = []
    for code in range(0x00, 0x7F):
        decoded = float(decode_e4m3_bytes(torch.tensor(code, dtype=torch.uint8)))
        candidates.append((decoded, code))
    best_error = min(abs(decoded - magnitude) for decoded, _ in candidates)
    tied = [code for decoded, code in candidates if abs(decoded - magnitude) == best_error]
    even = [code for code in tied if (code & 1) == 0]
    code = min(even if even else tied)
    if code == 0:
        return 0
    return sign | code


def scalar_mxfp8_reference(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slow independent scalar MXFP8 encoder/decoder for test cross-checks."""
    if source.ndim != 2:
        raise ValueError("scalar MXFP8 reference accepts a 2-D [rows,K] tensor")
    rows, k = source.shape
    padded_k = ((k + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE) * MX_BLOCK_SIZE
    payload = torch.zeros((rows, padded_k), dtype=torch.uint8)
    scale_bytes = torch.full((rows, padded_k // MX_BLOCK_SIZE), 127, dtype=torch.uint8)
    decoded = torch.zeros((rows, k), dtype=torch.float32)
    cpu = source.detach().float().cpu()
    for row in range(rows):
        for block_index, start in enumerate(range(0, padded_k, MX_BLOCK_SIZE)):
            valid = [float(cpu[row, col]) for col in range(start, min(start + 32, k))]
            amax = max((abs(value) for value in valid), default=0.0)
            exponent = 0 if amax == 0 else math.floor(math.log2(amax)) - 8
            exponent = max(MX_MIN_SCALE_EXP, min(MX_MAX_SCALE_EXP, exponent))
            scale_byte = exponent + MX_SCALE_BIAS
            scale_bytes[row, block_index] = scale_byte
            scale = 2.0 ** exponent
            for offset in range(MX_BLOCK_SIZE):
                col = start + offset
                value = float(cpu[row, col]) if col < k else 0.0
                code = scalar_e4m3_encode(value / scale)
                payload[row, col] = code
                if col < k:
                    decoded[row, col] = float(
                        decode_e4m3_bytes(torch.tensor(code, dtype=torch.uint8))
                    ) * scale
    return payload.to(source.device), scale_bytes.to(source.device), decoded.to(source.device)


def quantize_high(source: torch.Tensor, fmt: HighFormat) -> torch.Tensor:
    if fmt == "bf16":
        return source
    if fmt == "e4m3":
        return quantize_plain_e4m3(source).reconstructed
    if fmt == "mxfp8":
        return quantize_mxfp8_e4m3(source).reconstructed
    raise ValueError(f"unsupported high format {fmt!r}")


def serialized_weight_bytes(
    *, out_features: int, low_rank: int, high_rank: int, high_format: HighFormat,
    metadata_bytes: int = 0, alignment: int = 1,
) -> dict[str, int]:
    """All-inclusive active transformed-weight byte accounting."""
    low_padded = ((low_rank + 15) // 16) * 16
    low_payload = out_features * low_padded // 2
    low_scales = out_features * (low_padded // 16)
    low_global = 4
    if high_format == "bf16":
        high_payload = out_features * high_rank * 2
        high_scales = 0
    elif high_format == "e4m3":
        high_payload = out_features * high_rank
        high_scales = 4
    elif high_format == "mxfp8":
        high_padded = ((high_rank + 31) // 32) * 32
        high_payload = out_features * high_padded
        high_scales = out_features * (high_padded // 32)
    else:
        raise ValueError(f"unsupported high format {high_format!r}")
    raw_total = low_payload + low_scales + low_global + high_payload + high_scales + metadata_bytes
    total = ((raw_total + alignment - 1) // alignment) * alignment
    return {
        "low_payload": low_payload,
        "low_scales": low_scales,
        "low_global": low_global,
        "high_payload": high_payload,
        "high_scales": high_scales,
        "metadata": metadata_bytes,
        "alignment_padding": total - raw_total,
        "total": total,
    }
