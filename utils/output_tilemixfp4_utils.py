"""Local partial-output TileMix fake-quant oracle.

Unlike the activation-SSE TileMix implementation, this selector scores each
16x64 activation tile through the matching slice of the *actual executed*
fake-quantized linear weight.  It is intentionally local in K: errors from
different K tiles are not allowed to cancel.  The selected activation payload
is still shared across every output channel N.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .tilemixfp4_utils import (
    E0M3_MAGNITUDES,
    E2M1_MAGNITUDES,
    FP4_BLOCK_SIZE,
    TILE_COLS,
    TILE_ROWS,
    FormatSelectionStats,
    _flatten_operand,
    _hardware_global_scale,
    _quantize_blocks_e4m3,
    _validate_hardware_clip_ratio,
)


@dataclass
class OutputOracleFormatStats(FormatSelectionStats):
    """Selector counts plus device-side local weighted-output score sums."""

    _weighted_scores: torch.Tensor | None = None  # selected, all-E2, all-E0

    def record_weighted_output(
        self,
        choose_e0: torch.Tensor,
        error_e2: torch.Tensor,
        error_e0: torch.Tensor,
    ) -> None:
        if choose_e0.shape != error_e2.shape or error_e2.shape != error_e0.shape:
            raise ValueError("choice and weighted-output score shapes must match")
        if choose_e0.device != error_e2.device or error_e2.device != error_e0.device:
            raise ValueError("choice and weighted-output scores must share a device")
        selected = torch.where(choose_e0, error_e0, error_e2)
        delta = torch.stack((selected.sum(), error_e2.sum(), error_e0.sum())).double()
        if self._weighted_scores is None:
            self._weighted_scores = torch.zeros(
                3, dtype=torch.float64, device=error_e2.device
            )
        elif self._weighted_scores.device != error_e2.device:
            raise RuntimeError("weighted-output counter device changed during generation")
        self._weighted_scores.add_(delta)

    def snapshot(self) -> dict[str, int | float | str]:
        result = super().snapshot()
        if self._weighted_scores is None:
            selected, all_e2, all_e0 = 0.0, 0.0, 0.0
        else:
            selected, all_e2, all_e0 = self._weighted_scores.detach().cpu().tolist()
        result.update({
            "selector_objective": "local_partial_output_error",
            "weighted_output_error_selected": selected,
            "weighted_output_error_all_e2": all_e2,
            "weighted_output_error_all_e0": all_e0,
        })
        return result


def build_output_weight_grams(weight_low: torch.Tensor) -> torch.Tensor:
    """Build ``G_k = W_k W_k^T`` for every padded 64-column K slice.

    ``weight_low`` uses PyTorch Linear layout ``[N, K]`` and must be the
    already-quantized weight actually consumed by that Linear.  The returned
    tensor is FP32 ``[ceil(K/64), 64, 64]`` on the same device.
    """
    if weight_low.ndim != 2 or not weight_low.is_floating_point():
        raise ValueError("weight_low must be a floating [N, K] tensor")
    n, k = weight_low.shape
    if k == 0:
        return torch.empty(0, TILE_COLS, TILE_COLS, device=weight_low.device)
    pad_k = (-k) % TILE_COLS
    weight = weight_low.float()
    if pad_k:
        weight = F.pad(weight, (0, pad_k))
    blocks = weight.reshape(n, -1, TILE_COLS).permute(1, 2, 0).contiguous()
    return torch.bmm(blocks, blocks.transpose(1, 2))


def weighted_output_tile_scores_direct(
    delta_tiles: torch.Tensor, weight_low: torch.Tensor
) -> torch.Tensor:
    """Reference direct-matmul score for tests: ``||delta @ W_k||_F^2``."""
    if delta_tiles.ndim != 4 or delta_tiles.shape[-2:] != (TILE_ROWS, TILE_COLS):
        raise ValueError("delta_tiles must have shape [M_tiles, K_tiles, 16, 64]")
    mt, kt = delta_tiles.shape[:2]
    needed_k = kt * TILE_COLS
    if weight_low.ndim != 2 or weight_low.shape[1] > needed_k:
        raise ValueError("weight_low shape does not match the number of K tiles")
    padded = F.pad(weight_low.float(), (0, needed_k - weight_low.shape[1]))
    weight_tiles = padded.reshape(padded.shape[0], kt, TILE_COLS).permute(1, 2, 0)
    scores = []
    for tile_index in range(kt):
        output_error = torch.matmul(
            delta_tiles[:, tile_index].float(), weight_tiles[tile_index]
        )
        scores.append(output_error.square().sum(dim=(1, 2)))
    return torch.stack(scores, dim=1) if scores else delta_tiles.new_zeros((mt, 0))


def weighted_output_tile_scores_gram(
    delta_tiles: torch.Tensor, weight_grams: torch.Tensor
) -> torch.Tensor:
    """Score tiles with their per-K-slice Gram matrices, aggregating full N."""
    if delta_tiles.ndim != 4 or delta_tiles.shape[-2:] != (TILE_ROWS, TILE_COLS):
        raise ValueError("delta_tiles must have shape [M_tiles, K_tiles, 16, 64]")
    if weight_grams.shape != (delta_tiles.shape[1], TILE_COLS, TILE_COLS):
        raise ValueError("weight Gram shape does not match activation K tiles")
    delta = delta_tiles.float()
    weighted = torch.matmul(delta, weight_grams.unsqueeze(0))
    # Roundoff in an FP32 quadratic form can produce tiny negative values.
    return (weighted * delta).sum(dim=(-1, -2)).clamp_min_(0.0)


def fake_quantize_tile_mix_output_oracle(
    x: torch.Tensor,
    weight_grams: torch.Tensor,
    clip_ratio: float = 1.0,
    format_stats: OutputOracleFormatStats | None = None,
    m_tile_chunk: int = 64,
) -> torch.Tensor:
    """Choose one E2M1/E0M3 payload per 16x64 tile by local output error."""
    _validate_hardware_clip_ratio(clip_ratio)
    if m_tile_chunk <= 0:
        raise ValueError("m_tile_chunk must be positive")
    flat, shape = _flatten_operand(x)
    if flat.numel() == 0:
        return x.clone()
    if weight_grams.device != x.device:
        raise RuntimeError("weight Gram and activation must be on the same device")

    m, k = flat.shape
    scaled, s32 = _hardware_global_scale(flat)
    pad_m, pad_k = (-m) % TILE_ROWS, (-k) % TILE_COLS
    padded = F.pad(scaled, (0, pad_k, 0, pad_m)) if (pad_m or pad_k) else scaled
    mp, kp = padded.shape
    mt, kt = mp // TILE_ROWS, kp // TILE_COLS
    if weight_grams.shape != (kt, TILE_COLS, TILE_COLS):
        raise ValueError(
            f"expected weight grams {(kt, TILE_COLS, TILE_COLS)}, "
            f"got {tuple(weight_grams.shape)}"
        )

    output = torch.empty_like(padded)
    for start in range(0, mt, m_tile_chunk):
        end = min(start + m_tile_chunk, mt)
        rows = padded[start * TILE_ROWS:end * TILE_ROWS]
        blocks = rows.reshape(
            (end - start) * TILE_ROWS, kt * (TILE_COLS // FP4_BLOCK_SIZE),
            FP4_BLOCK_SIZE,
        )
        q_e2 = _quantize_blocks_e4m3(blocks, E2M1_MAGNITUDES).reshape(
            end - start, TILE_ROWS, kt, TILE_COLS
        ).permute(0, 2, 1, 3)
        q_e0 = _quantize_blocks_e4m3(blocks, E0M3_MAGNITUDES).reshape(
            end - start, TILE_ROWS, kt, TILE_COLS
        ).permute(0, 2, 1, 3)
        original = rows.reshape(end - start, TILE_ROWS, kt, TILE_COLS).permute(
            0, 2, 1, 3
        )

        # Restore s32 before scoring so the objective is in the executed
        # activation/weight units.  The common scalar would not affect choices,
        # but retaining it gives interpretable accumulated output-error scores.
        original_actual = original * s32
        q_e2_actual = q_e2 * s32
        q_e0_actual = q_e0 * s32
        error_e2 = weighted_output_tile_scores_gram(
            original_actual - q_e2_actual, weight_grams
        )
        error_e0 = weighted_output_tile_scores_gram(
            original_actual - q_e0_actual, weight_grams
        )
        choose_e0 = error_e0 < error_e2  # strict: ties select E2M1
        if format_stats is not None:
            format_stats.record(choose_e0)
            format_stats.record_weighted_output(choose_e0, error_e2, error_e0)

        selected = torch.where(
            choose_e0[..., None, None], q_e0_actual, q_e2_actual
        )
        output[start * TILE_ROWS:end * TILE_ROWS] = selected.permute(
            0, 2, 1, 3
        ).reshape((end - start) * TILE_ROWS, kp)

    return output[:m, :k].reshape(shape).to(x.dtype)


__all__ = [
    "OutputOracleFormatStats",
    "build_output_weight_grams",
    "weighted_output_tile_scores_direct",
    "weighted_output_tile_scores_gram",
    "fake_quantize_tile_mix_output_oracle",
]
