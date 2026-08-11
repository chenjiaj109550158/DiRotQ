"""Pure-PyTorch activation fake quantization for E2M1/E0M3 mixtures.

The hardware-oriented formats use one FP32 global scale for the complete
low-precision operand supplied to this module, then independent E4M3 scales
for 1x16 blocks along its final (K) axis.  Tile padding is added only after
the global scale is calculated.  The legacy ``nvfp4`` entry point remains a
separate path so DiRotQ's original high-precision ``amax / 6`` block scales
are preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E0M3_MAGNITUDES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

FP4_BLOCK_SIZE = 16
TILE_ROWS = 16
TILE_COLS = 64
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0 ** -9
GLOBAL_SCALED_MAX = 2688.0
E2M1_MAX_BLOCK_SCALE = GLOBAL_SCALED_MAX / E2M1_MAGNITUDES[-1]  # 448
E0M3_MAX_BLOCK_SCALE = GLOBAL_SCALED_MAX / E0M3_MAGNITUDES[-1]  # 384

_magnitude_cache: dict[tuple[tuple[float, ...], torch.device], torch.Tensor] = {}
_legacy_e2m1_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


@dataclass
class FormatSelectionStats:
    """Lightweight device-side E2M1/E0M3 selection counters.

    Forward calls update one shared CUDA int64 tensor and never call
    ``.item()``.  ``snapshot`` performs the single host synchronization after
    generation has completed.  No selector maps or activations are retained.
    """

    selection_unit: str
    _counts: torch.Tensor | None = None  # [E2M1, E0M3]
    _error_sums: torch.Tensor | None = None  # [signal energy, reconstruction SSE]

    def record(self, choose_e0: torch.Tensor) -> None:
        if choose_e0.dtype != torch.bool:
            raise TypeError("choose_e0 must be a boolean tensor")
        if self._counts is None:
            self._counts = torch.zeros(2, dtype=torch.int64, device=choose_e0.device)
        elif self._counts.device != choose_e0.device:
            raise RuntimeError("format-stat counter device changed during generation")
        e0_count = choose_e0.sum(dtype=torch.int64)
        delta = torch.stack((choose_e0.numel() - e0_count, e0_count))
        self._counts.add_(delta)

    def record_reconstruction(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> None:
        """Accumulate low-precision-region signal energy and reconstruction SSE.

        Reductions stay on the activation device and only their two scalar results
        are promoted to float64 for accumulation.  No activation tensor or selector
        map is retained, and host synchronization occurs only in ``snapshot``.
        """
        if original.shape != reconstructed.shape:
            raise ValueError("original and reconstructed shapes must match")
        if original.device != reconstructed.device:
            raise ValueError("original and reconstructed devices must match")
        signal = original.float().square().sum()
        error = (original - reconstructed).float().square().sum()
        delta = torch.stack((signal, error)).to(torch.float64)
        if self._error_sums is None:
            self._error_sums = torch.zeros(
                2, dtype=torch.float64, device=original.device
            )
        elif self._error_sums.device != original.device:
            raise RuntimeError("reconstruction-stat counter device changed during generation")
        self._error_sums.add_(delta)

    def snapshot(self) -> dict[str, int | float | str]:
        if self._counts is None:
            e2_count, e0_count = 0, 0
        else:
            e2_count, e0_count = self._counts.detach().cpu().tolist()
        if self._error_sums is None:
            signal_energy, reconstruction_sse = 0.0, 0.0
        else:
            signal_energy, reconstruction_sse = self._error_sums.detach().cpu().tolist()
        total = e2_count + e0_count
        if reconstruction_sse == 0.0:
            qsnr_db = math.inf if signal_energy > 0.0 else 0.0
        elif signal_energy == 0.0:
            qsnr_db = -math.inf
        else:
            qsnr_db = 10.0 * math.log10(signal_energy / reconstruction_sse)
        return {
            "selection_unit": self.selection_unit,
            "e2m1_count": e2_count,
            "e0m3_count": e0_count,
            "total_count": total,
            "e0m3_ratio": (e0_count / total) if total else 0.0,
            "signal_energy": signal_energy,
            "reconstruction_sse": reconstruction_sse,
            "qsnr_db": qsnr_db,
        }


def _check_input(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension [..., K]")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype")


def _flatten_operand(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Return a finite fp32 [M, K] view/copy and the original shape."""
    _check_input(x)
    shape = x.shape
    if x.numel() == 0:
        return x.float().reshape(-1, shape[-1]), shape
    work = torch.nan_to_num(
        x.float(), nan=0.0, posinf=E4M3_MAX * 7.0, neginf=-E4M3_MAX * 7.0
    )
    return work.reshape(-1, shape[-1]), shape


def _get_magnitudes(values: tuple[float, ...], device: torch.device) -> torch.Tensor:
    key = (values, device)
    if key not in _magnitude_cache:
        _magnitude_cache[key] = torch.tensor(values, device=device, dtype=torch.float32)
    return _magnitude_cache[key]


def _round_magnitude(x: torch.Tensor, magnitudes: tuple[float, ...]) -> torch.Tensor:
    """Nearest-magnitude rounding with sign restored (symmetric at ties)."""
    cb = _get_magnitudes(magnitudes, x.device)
    midpoints = (cb[:-1] + cb[1:]) * 0.5
    indices = torch.bucketize(x.abs().contiguous(), midpoints)
    return torch.sign(x) * cb[indices]


def _round_e4m3_scale(scale: torch.Tensor, zero_block: torch.Tensor) -> torch.Tensor:
    """Round positive block scales to finite E4M3 values.

    E4M3FN has finite range [0, 448] and a smallest positive subnormal of
    2**-9.  Exact-zero blocks use scale 1 (their reconstruction remains zero).
    Nonzero scales that underflow are raised to the smallest legal subnormal.
    """
    finite = torch.nan_to_num(scale.float(), nan=0.0, posinf=E4M3_MAX, neginf=0.0)
    rounded = finite.clamp(min=0.0, max=E4M3_MAX).to(torch.float8_e4m3fn).float().abs()
    rounded = rounded.clamp(min=E4M3_MIN_SUBNORMAL, max=E4M3_MAX)
    return torch.where(zero_block, torch.ones_like(rounded), rounded)


def _validate_hardware_clip_ratio(clip_ratio: float) -> None:
    if clip_ratio != 1.0:
        raise ValueError(
            "hardware-faithful FP4 modes require clip_ratio=1.0 "
            "because their scale definition is fixed"
        )


def _hardware_global_scale(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``Z_scaled`` and the single scalar FP32 global scale ``s32``.

    ``flat`` is exactly the low-precision operand reshaped from ``[..., K]``
    to ``[M, K]``.  Consequently the reduction spans every M/K element in the
    low-precision region, excludes any high-precision tail already split by
    ``ActQuantizer``, and occurs before block/tile padding.
    """
    if flat.numel() == 0:
        return flat, torch.ones((), dtype=torch.float32, device=flat.device)
    global_amax = flat.abs().amax()
    s32 = torch.where(
        global_amax == 0,
        torch.ones_like(global_amax),
        global_amax / GLOBAL_SCALED_MAX,
    )
    return flat / s32, s32


def _e4m3_block_scale(
    blocks: torch.Tensor,
    magnitudes: tuple[float, ...],
) -> torch.Tensor:
    """Calculate candidate-specific E4M3 scales for fp32 ``[..., 16]`` blocks."""
    if blocks.shape[-1] != FP4_BLOCK_SIZE:
        raise ValueError(f"expected blocks of {FP4_BLOCK_SIZE}, got {blocks.shape[-1]}")

    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    zero_block = absmax == 0
    max_scale = GLOBAL_SCALED_MAX / magnitudes[-1]
    raw_scale = (absmax / magnitudes[-1]).clamp(max=max_scale)
    return _round_e4m3_scale(raw_scale, zero_block).clamp(max=max_scale)


def _quantize_blocks_e4m3(
    blocks: torch.Tensor,
    magnitudes: tuple[float, ...],
) -> torch.Tensor:
    """Quantize globally-scaled fp32 blocks with per-block E4M3 scales."""
    scale = _e4m3_block_scale(blocks, magnitudes)
    codes = _round_magnitude(blocks / scale, magnitudes)
    return codes * scale


def _pad_k_to(x_2d: torch.Tensor, divisor: int) -> tuple[torch.Tensor, int]:
    k = x_2d.shape[-1]
    pad_k = (-k) % divisor
    return (F.pad(x_2d, (0, pad_k)) if pad_k else x_2d), pad_k


def _fixed_format_fake_quant(
    x: torch.Tensor,
    magnitudes: tuple[float, ...],
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    _validate_hardware_clip_ratio(clip_ratio)
    flat, shape = _flatten_operand(x)
    if flat.numel() == 0:
        return x.clone()
    k = flat.shape[-1]
    scaled, s32 = _hardware_global_scale(flat)
    padded, _ = _pad_k_to(scaled, FP4_BLOCK_SIZE)
    blocks = padded.reshape(flat.shape[0], -1, FP4_BLOCK_SIZE)
    quantized = _quantize_blocks_e4m3(blocks, magnitudes) * s32
    return quantized.reshape(flat.shape[0], -1)[:, :k].reshape(shape).to(x.dtype)


def fake_quantize_e2m1(x: torch.Tensor, clip_ratio: float = 1.0) -> torch.Tensor:
    """Hardware-faithful E2M1 with FP32 global and E4M3 block scales."""
    return _fixed_format_fake_quant(x, E2M1_MAGNITUDES, clip_ratio)


def fake_quantize_nvfp4_hw(x: torch.Tensor, clip_ratio: float = 1.0) -> torch.Tensor:
    """Named entry point for the hardware-faithful fixed-E2M1 baseline."""
    return fake_quantize_e2m1(x, clip_ratio)


def fake_quantize_e0m3(x: torch.Tensor, clip_ratio: float = 1.0) -> torch.Tensor:
    """Hardware-faithful E0M3 with FP32 global and E4M3 block scales."""
    return _fixed_format_fake_quant(x, E0M3_MAGNITUDES, clip_ratio)


def fake_quantize_block_mix_oracle(
    x: torch.Tensor,
    clip_ratio: float = 1.0,
    format_stats: FormatSelectionStats | None = None,
) -> torch.Tensor:
    """Choose E2M1/E0M3 independently for every 1x16 activation block."""
    _validate_hardware_clip_ratio(clip_ratio)
    flat, shape = _flatten_operand(x)
    if flat.numel() == 0:
        return x.clone()
    m, k = flat.shape
    scaled, s32 = _hardware_global_scale(flat)
    padded, _ = _pad_k_to(scaled, FP4_BLOCK_SIZE)
    valid = torch.zeros_like(padded, dtype=torch.bool)
    valid[:, :k] = True

    blocks = padded.reshape(m, -1, FP4_BLOCK_SIZE)
    valid_blocks = valid.reshape_as(blocks)
    q_e2 = _quantize_blocks_e4m3(blocks, E2M1_MAGNITUDES)
    q_e0 = _quantize_blocks_e4m3(blocks, E0M3_MAGNITUDES)
    err_e2 = ((blocks - q_e2).square() * valid_blocks).sum(dim=-1)
    err_e0 = ((blocks - q_e0).square() * valid_blocks).sum(dim=-1)
    choose_e0 = err_e0 < err_e2  # strict comparison: ties select E2M1
    if format_stats is not None:
        format_stats.record(choose_e0)
    out = torch.where(choose_e0.unsqueeze(-1), q_e0, q_e2) * s32
    return out.reshape(m, -1)[:, :k].reshape(shape).to(x.dtype)


def fake_quantize_tile_mix_oracle(
    x: torch.Tensor,
    clip_ratio: float = 1.0,
    format_stats: FormatSelectionStats | None = None,
) -> torch.Tensor:
    """Choose one micro-format per 16x64 tile, retaining 1x16 block scales."""
    _validate_hardware_clip_ratio(clip_ratio)
    flat, shape = _flatten_operand(x)
    if flat.numel() == 0:
        return x.clone()
    m, k = flat.shape
    scaled, s32 = _hardware_global_scale(flat)
    pad_m = (-m) % TILE_ROWS
    pad_k = (-k) % TILE_COLS
    padded = F.pad(scaled, (0, pad_k, 0, pad_m)) if (pad_m or pad_k) else scaled
    mp, kp = padded.shape

    valid = torch.zeros_like(padded, dtype=torch.bool)
    valid[:m, :k] = True
    blocks = padded.reshape(mp, kp // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE)
    q_e2 = _quantize_blocks_e4m3(blocks, E2M1_MAGNITUDES).reshape(mp, kp)
    q_e0 = _quantize_blocks_e4m3(blocks, E0M3_MAGNITUDES).reshape(mp, kp)

    mt, kt = mp // TILE_ROWS, kp // TILE_COLS
    err_e2 = ((padded - q_e2).square() * valid).reshape(
        mt, TILE_ROWS, kt, TILE_COLS
    ).sum(dim=(1, 3))
    err_e0 = ((padded - q_e0).square() * valid).reshape(
        mt, TILE_ROWS, kt, TILE_COLS
    ).sum(dim=(1, 3))
    choose_e0 = err_e0 < err_e2  # strict comparison: ties select E2M1
    if format_stats is not None:
        format_stats.record(choose_e0)

    q_e2_tiles = q_e2.reshape(mt, TILE_ROWS, kt, TILE_COLS)
    q_e0_tiles = q_e0.reshape(mt, TILE_ROWS, kt, TILE_COLS)
    out = torch.where(choose_e0[:, None, :, None], q_e0_tiles, q_e2_tiles)
    out = out.reshape(mp, kp)[:m, :k] * s32
    return out.reshape(shape).to(x.dtype)


def fake_quantize_nvfp4_legacy(
    x: torch.Tensor, clip_ratio: float = 1.0, block_size: int = FP4_BLOCK_SIZE
) -> torch.Tensor:
    """Numerically preserve the existing grouped activation NVFP4 path."""
    _check_input(x)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if x.numel() == 0:
        return x.clone()
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    k = flat.shape[-1]
    padded, _ = _pad_k_to(flat, block_size)
    blocks = padded.reshape(flat.shape[0], -1, block_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True) * clip_ratio
    scale = (scale / E2M1_MAGNITUDES[-1]).clamp(min=1e-5)

    key = (x.device, x.dtype)
    if key not in _legacy_e2m1_cache:
        signed = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
                  0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        _legacy_e2m1_cache[key] = torch.tensor(signed, device=x.device, dtype=x.dtype)
    cb = _legacy_e2m1_cache[key]
    midpoints = (cb[:-1] + cb[1:]) * 0.5
    indices = torch.bucketize((blocks / scale).contiguous(), midpoints)
    quantized = cb[indices] * scale
    return quantized.reshape(flat.shape[0], -1)[:, :k].reshape(shape).to(x.dtype)


def fake_quantize_activation(
    x: torch.Tensor,
    activation_format: str,
    clip_ratio: float = 1.0,
    format_stats: FormatSelectionStats | None = None,
) -> torch.Tensor:
    """Dispatch a legacy or hardware-faithful activation fake-quant mode."""
    if activation_format == "nvfp4":
        return fake_quantize_nvfp4_legacy(x, clip_ratio=clip_ratio)
    if activation_format == "nvfp4-hw":
        return fake_quantize_nvfp4_hw(x, clip_ratio=clip_ratio)
    if activation_format == "e0m3":
        return fake_quantize_e0m3(x, clip_ratio=clip_ratio)
    if activation_format == "block-mix-oracle":
        return fake_quantize_block_mix_oracle(
            x, clip_ratio=clip_ratio, format_stats=format_stats
        )
    if activation_format == "tile-mix-oracle":
        return fake_quantize_tile_mix_oracle(
            x, clip_ratio=clip_ratio, format_stats=format_stats
        )
    raise ValueError(f"unsupported activation format: {activation_format}")


__all__ = [
    "E2M1_MAGNITUDES",
    "E0M3_MAGNITUDES",
    "GLOBAL_SCALED_MAX",
    "E2M1_MAX_BLOCK_SCALE",
    "E0M3_MAX_BLOCK_SCALE",
    "FormatSelectionStats",
    "fake_quantize_e2m1",
    "fake_quantize_nvfp4_hw",
    "fake_quantize_e0m3",
    "fake_quantize_block_mix_oracle",
    "fake_quantize_tile_mix_oracle",
    "fake_quantize_nvfp4_legacy",
    "fake_quantize_activation",
]
