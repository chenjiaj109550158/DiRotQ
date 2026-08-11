"""Streaming distribution and temporal diagnostics for SSE TileMix.

This module is an experiment-only sidecar.  It observes the low-precision
operand before the existing ``tile-mix-oracle`` fake quantizer, aggregates
counts/histograms on device, and never changes the activation returned to the
model.  Only the previous timestep's boolean format map and error norm are
retained transiently for temporal alignment; activations are never retained or
written to disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .output_tilemixfp4_utils import (
    build_output_weight_grams,
    weighted_output_tile_scores_gram,
)
from .tilemixfp4_utils import (
    E0M3_MAGNITUDES,
    E2M1_MAGNITUDES,
    FP4_BLOCK_SIZE,
    TILE_COLS,
    TILE_ROWS,
    _e4m3_block_scale,
    _flatten_operand,
    _hardware_global_scale,
    _quantize_blocks_e4m3,
)


CREST_EDGES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
MAG_HIST_BINS = 32
SPEARMAN_CREST_BINS = 64
SPEARMAN_LOG_BINS = 128
SPEARMAN_LOG_RANGE = (-8.0, 8.0)
SCALE_ERROR_EDGES = (
    0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04,
    0.08, 0.16, 0.32, 0.64, float("inf"),
)


_LAYER_FIELDS = (
    "block_count", "nonzero_block_count", "element_count", "zero_element_count",
    "crest_sum", "kurtosis_sum", "exact_e0_win", "hw_e0_win",
    "winner_disagree", "round_e0_to_e2", "round_e2_to_e0",
    "e2_exact_sse", "e0_exact_sse", "e2_hw_sse", "e0_hw_sse",
    "log_error_ratio_sum", "e2_scale_rel_sum", "e0_scale_rel_sum",
    "scale_nonzero_count", "e2_saturation_count", "e0_saturation_count",
    "tile_count", "sse_tile_e0", "output_tile_e0", "selector_agree",
    "sse_e0_score", "sse_e2_score", "weighted_e0_score", "weighted_e2_score",
    "sse_selected_sse", "output_selected_sse",
    "sse_selected_weighted", "output_selected_weighted",
)

_TRANSITION_FIELDS = (
    "tile_count", "flip_count", "e0_to_e2", "e2_to_e0",
    "switched_gain_sum", "switched_gain_count",
    "stable_gain_sum", "stable_gain_count", "error_norm_rel_change_sum",
)

_PROMPT_FIELDS = (
    "nonzero_block_count", "crest_sum", "low_crest_count",
    "exact_e0_win", "hw_e0_win", "tile_count", "sse_e0_score",
    "sse_selected_sse", "selector_disagree", "transition_count",
    "flip_count", "run_length_sum", "run_count",
)


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def _exact_candidate(blocks: torch.Tensor, magnitudes: tuple[float, ...]):
    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    zero = absmax == 0
    scale = torch.where(zero, torch.ones_like(absmax), absmax / magnitudes[-1])
    codebook = torch.tensor(magnitudes, device=blocks.device, dtype=torch.float32)
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5
    indices = torch.bucketize((blocks / scale).abs().contiguous(), midpoints)
    quantized = torch.sign(blocks) * codebook[indices] * scale
    return quantized, scale, indices


def _hardware_candidate(blocks: torch.Tensor, magnitudes: tuple[float, ...]):
    scale = _e4m3_block_scale(blocks, magnitudes)
    codebook = torch.tensor(magnitudes, device=blocks.device, dtype=torch.float32)
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5
    normalized = (blocks / scale).abs()
    indices = torch.bucketize(normalized.contiguous(), midpoints)
    quantized = torch.sign(blocks) * codebook[indices] * scale
    return quantized, scale, indices, normalized


def _weighted_histogram_spearman(hist: np.ndarray) -> float:
    """Approximate Spearman rho from a fixed online bivariate histogram."""
    hist = hist.astype(np.float64, copy=False)
    total = hist.sum()
    if total == 0:
        return math.nan
    rows, cols = hist.sum(1), hist.sum(0)
    row_ranks = np.cumsum(rows) - rows / 2.0
    col_ranks = np.cumsum(cols) - cols / 2.0
    rx = row_ranks[:, None]
    ry = col_ranks[None, :]
    mx = float((hist * rx).sum() / total)
    my = float((hist * ry).sum() / total)
    cov = float((hist * (rx - mx) * (ry - my)).sum() / total)
    vx = float((hist * (rx - mx) ** 2).sum() / total)
    vy = float((hist * (ry - my) ** 2).sum() / total)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else math.nan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DistributionAuditCollector:
    """Device-side aggregate collector attached to quantized linear wrappers."""

    def __init__(
        self,
        model_name: str,
        output_dir: str | Path,
        dataset_path: str | Path,
        quality_csvs: Iterable[str | Path],
        quantized_cache: str | Path,
        bootstrap_samples: int = 2000,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.dataset_path = Path(dataset_path)
        self.quality_csvs = [Path(path) for path in quality_csvs]
        self.quantized_cache = Path(quantized_cache)
        self.bootstrap_samples = bootstrap_samples
        with self.dataset_path.open() as f:
            self.samples = list(json.load(f).items())[:64]
        self.sample_index = {image_id: index for index, (image_id, _) in enumerate(self.samples)}

        self.layer_names: list[str] = []
        self.layer_index: dict[int, int] = {}
        self.pipeline = None
        self._hook = None
        self.device = None
        self.num_steps = None
        self.timestep_values: list[float] = []
        self.layer_acc = {}
        self.transition_acc = {}
        self.prompt_acc = {}
        self.run_sum = None
        self.run_count = None
        self.crest_counts = None
        self.crest_exact_wins = None
        self.crest_hw_wins = None
        self.crest_log_sum = None
        self.norm_hist = None
        self.e2_occupancy = None
        self.e0_occupancy = None
        self.e2_scale_hist = None
        self.e0_scale_hist = None
        self.crest_log_hist = None
        self.weight_grams = {}

        self.current_prompt_indices: list[int] | None = None
        self.current_step = None
        self._called_current = set()
        self._seen_steps: list[int] = []
        self._previous = {}
        self.exclusions: dict[str, int] = {}

    def attach(self, transformer, pipeline) -> None:
        self.pipeline = pipeline
        for name, module in transformer.named_modules():
            quantizer = getattr(module, "quantizer", None)
            if quantizer is None or quantizer.bits >= 16:
                continue
            index = len(self.layer_names)
            self.layer_names.append(name)
            self.layer_index[id(module)] = index
            module.distribution_audit = self
        if not self.layer_names:
            raise RuntimeError("distribution audit found no active activation quantizers")
        self._hook = transformer.register_forward_pre_hook(
            self._transformer_pre_hook, with_kwargs=True
        )

    def start_batch(self, batch) -> None:
        if self.current_prompt_indices is not None:
            raise RuntimeError("distribution audit batch already active")
        self.current_prompt_indices = [self.sample_index[image_id] for image_id, _ in batch]
        self.current_step = None
        self._seen_steps = []
        self._previous = {}

    def end_batch(self) -> None:
        if self.current_prompt_indices is None:
            return
        if self.num_steps is not None and self._seen_steps != list(range(self.num_steps)):
            self._exclude("transformer_timestep_sequence_mismatch")
        self._finalize_open_runs()
        self._previous = {}
        self.current_prompt_indices = None
        self.current_step = None

    def _exclude(self, reason: str, amount: int = 1) -> None:
        self.exclusions[reason] = self.exclusions.get(reason, 0) + amount

    def _transformer_pre_hook(self, module, args, kwargs) -> None:
        if self.current_prompt_indices is None:
            raise RuntimeError("transformer ran without distribution audit batch context")
        if "timestep" not in kwargs:
            raise RuntimeError("transformer forward did not expose a true timestep kwarg")
        passed = kwargs["timestep"].detach().float().flatten()
        if passed.numel() == 0 or not torch.allclose(passed, passed[:1]):
            raise RuntimeError("transformer timestep is empty or differs across CFG batch")
        scale = float(getattr(module.config, "timestep_scale", 1.0))
        true_value = float((passed[0] / scale).cpu())
        schedule = self.pipeline.scheduler.timesteps.detach().float().cpu().numpy()
        index = int(np.argmin(np.abs(schedule - true_value)))
        tolerance = 1e-4 * max(1.0, abs(float(schedule[index])))
        if abs(float(schedule[index]) - true_value) > tolerance:
            raise RuntimeError(
                f"passed timestep {true_value} does not match scheduler trajectory"
            )
        if self.num_steps is None:
            self.num_steps = len(schedule)
            self.timestep_values = [float(v) for v in schedule]
        elif len(schedule) != self.num_steps:
            raise RuntimeError("scheduler timestep count changed during audit")
        self.current_step = index
        self._seen_steps.append(index)
        self._called_current = set()

    def _allocate(self, device: torch.device) -> None:
        if self.device is not None:
            if self.device != device:
                raise RuntimeError("distribution audit device changed; CPU fallback is forbidden")
            return
        if self.num_steps is None:
            raise RuntimeError("distribution audit saw an activation before true timestep")
        self.device = device
        shape = (len(self.layer_names), self.num_steps)
        self.layer_acc = {
            name: torch.zeros(shape, dtype=torch.float64, device=device)
            for name in _LAYER_FIELDS
        }
        self.transition_acc = {
            name: torch.zeros(shape, dtype=torch.float64, device=device)
            for name in _TRANSITION_FIELDS
        }
        self.prompt_acc = {
            name: torch.zeros(len(self.samples), dtype=torch.float64, device=device)
            for name in _PROMPT_FIELDS
        }
        self.run_sum = torch.zeros(len(self.layer_names), dtype=torch.float64, device=device)
        self.run_count = torch.zeros_like(self.run_sum)
        self.crest_counts = torch.zeros(7, dtype=torch.float64, device=device)
        self.crest_exact_wins = torch.zeros_like(self.crest_counts)
        self.crest_hw_wins = torch.zeros_like(self.crest_counts)
        self.crest_log_sum = torch.zeros_like(self.crest_counts)
        self.norm_hist = torch.zeros(MAG_HIST_BINS, dtype=torch.float64, device=device)
        self.e2_occupancy = torch.zeros(8, dtype=torch.float64, device=device)
        self.e0_occupancy = torch.zeros(8, dtype=torch.float64, device=device)
        self.e2_scale_hist = torch.zeros(len(SCALE_ERROR_EDGES) - 1, dtype=torch.float64, device=device)
        self.e0_scale_hist = torch.zeros_like(self.e2_scale_hist)
        self.crest_log_hist = torch.zeros(
            SPEARMAN_CREST_BINS, SPEARMAN_LOG_BINS,
            dtype=torch.float64, device=device,
        )

    @torch.no_grad()
    def observe(self, wrapper, x: torch.Tensor) -> None:
        if x.device.type != "cuda":
            raise RuntimeError("distribution audit requires CUDA; CPU fallback is forbidden")
        if self.current_step is None or self.current_prompt_indices is None:
            raise RuntimeError("distribution audit activation lacks timestep/batch context")
        self._allocate(x.device)
        layer = self.layer_index[id(wrapper)]
        step = self.current_step
        duplicate_call = layer in self._called_current
        self._called_current.add(layer)
        if duplicate_call:
            self._exclude("duplicate_layer_call_same_timestep")

        q_len = x.shape[-1] - wrapper.quantizer.high_bits_length
        if q_len <= 0:
            self._exclude("empty_low_precision_region")
            return
        low = x[..., :q_len]
        flat, _ = _flatten_operand(low)
        m, k = flat.shape
        scaled, s32 = _hardware_global_scale(flat)
        s32_sq = s32.double().square()

        pad_k16 = (-k) % FP4_BLOCK_SIZE
        padded16 = F.pad(scaled, (0, pad_k16)) if pad_k16 else scaled
        valid16 = torch.zeros_like(padded16, dtype=torch.bool)
        valid16[:, :k] = True
        blocks = padded16.reshape(m, -1, FP4_BLOCK_SIZE)
        valid_blocks = valid16.reshape_as(blocks)
        valid_count = valid_blocks.sum(-1).float()

        q2_exact, s2_exact, _ = _exact_candidate(blocks, E2M1_MAGNITUDES)
        q0_exact, s0_exact, _ = _exact_candidate(blocks, E0M3_MAGNITUDES)
        q2_hw, s2_hw, idx2, norm2 = _hardware_candidate(blocks, E2M1_MAGNITUDES)
        q0_hw, s0_hw, idx0, norm0 = _hardware_candidate(blocks, E0M3_MAGNITUDES)

        def block_sse(candidate):
            return ((blocks - candidate).square() * valid_blocks).sum(-1).double() * s32_sq

        e2_exact = block_sse(q2_exact)
        e0_exact = block_sse(q0_exact)
        e2_hw = block_sse(q2_hw)
        e0_hw = block_sse(q0_hw)
        exact_e0 = e0_exact < e2_exact
        hw_e0 = e0_hw < e2_hw
        disagreement = exact_e0 != hw_e0
        log_ratio = torch.log2((e0_hw + 1e-20) / (e2_hw + 1e-20))

        absmax = blocks.abs().amax(-1)
        nonzero = absmax > 0
        rms = torch.sqrt((blocks.square() * valid_blocks).sum(-1) / valid_count.clamp_min(1))
        crest = torch.where(nonzero, absmax / rms.clamp_min(1e-20), torch.zeros_like(absmax))
        mean = (blocks * valid_blocks).sum(-1) / valid_count.clamp_min(1)
        centered = (blocks - mean.unsqueeze(-1)) * valid_blocks
        variance = centered.square().sum(-1) / valid_count.clamp_min(1)
        kurtosis = torch.where(
            variance > 0,
            centered.pow(4).sum(-1) / valid_count.clamp_min(1) / variance.square(),
            torch.zeros_like(variance),
        )

        raw2 = torch.where(nonzero.unsqueeze(-1), absmax.unsqueeze(-1) / 6.0, torch.ones_like(s2_hw))
        raw0 = torch.where(nonzero.unsqueeze(-1), absmax.unsqueeze(-1) / 7.0, torch.ones_like(s0_hw))
        rel2 = ((s2_hw - raw2).abs() / raw2.clamp_min(1e-20)).squeeze(-1)
        rel0 = ((s0_hw - raw0).abs() / raw0.clamp_min(1e-20)).squeeze(-1)

        acc = self.layer_acc
        at = (layer, step)
        acc["block_count"][at] += blocks.shape[0] * blocks.shape[1]
        acc["nonzero_block_count"][at] += nonzero.sum()
        acc["element_count"][at] += valid_blocks.sum()
        acc["zero_element_count"][at] += ((blocks == 0) & valid_blocks).sum()
        acc["crest_sum"][at] += crest[nonzero].double().sum()
        acc["kurtosis_sum"][at] += kurtosis[nonzero].double().sum()
        acc["exact_e0_win"][at] += exact_e0.sum()
        acc["hw_e0_win"][at] += hw_e0.sum()
        acc["winner_disagree"][at] += disagreement.sum()
        acc["round_e0_to_e2"][at] += (exact_e0 & ~hw_e0).sum()
        acc["round_e2_to_e0"][at] += (~exact_e0 & hw_e0).sum()
        acc["e2_exact_sse"][at] += e2_exact.sum()
        acc["e0_exact_sse"][at] += e0_exact.sum()
        acc["e2_hw_sse"][at] += e2_hw.sum()
        acc["e0_hw_sse"][at] += e0_hw.sum()
        acc["log_error_ratio_sum"][at] += log_ratio.sum()
        acc["e2_scale_rel_sum"][at] += rel2[nonzero].double().sum()
        acc["e0_scale_rel_sum"][at] += rel0[nonzero].double().sum()
        acc["scale_nonzero_count"][at] += nonzero.sum()
        acc["e2_saturation_count"][at] += ((norm2 > 6.0) & valid_blocks).sum()
        acc["e0_saturation_count"][at] += ((norm0 > 7.0) & valid_blocks).sum()

        self._update_global_block_histograms(
            blocks, valid_blocks, nonzero, crest, exact_e0, hw_e0, log_ratio,
            idx2, idx0, rel2, rel0,
        )

        pad_k64 = (-k) % TILE_COLS
        pad_m = (-m) % TILE_ROWS
        kp = k + pad_k64
        mp = m + pad_m
        original = F.pad(scaled, (0, pad_k64, 0, pad_m))
        q2 = F.pad(q2_hw.reshape(m, -1)[:, :k], (0, pad_k64, 0, pad_m))
        q0 = F.pad(q0_hw.reshape(m, -1)[:, :k], (0, pad_k64, 0, pad_m))
        mt, kt = mp // TILE_ROWS, kp // TILE_COLS
        original_tiles = original.reshape(mt, TILE_ROWS, kt, TILE_COLS).permute(0, 2, 1, 3)
        q2_tiles = q2.reshape(mt, TILE_ROWS, kt, TILE_COLS).permute(0, 2, 1, 3)
        q0_tiles = q0.reshape(mt, TILE_ROWS, kt, TILE_COLS).permute(0, 2, 1, 3)
        delta2 = (original_tiles - q2_tiles) * s32
        delta0 = (original_tiles - q0_tiles) * s32
        tile_e2 = delta2.double().square().sum((-1, -2))
        tile_e0 = delta0.double().square().sum((-1, -2))
        sse_choice = tile_e0 < tile_e2

        gram_key = (layer, x.device, k)
        grams = self.weight_grams.get(gram_key)
        if grams is None:
            weight = wrapper.module.weight[:, :k]
            if weight.device != x.device:
                raise RuntimeError("audit activation/weight device mismatch; no CPU fallback")
            grams = build_output_weight_grams(weight).detach()
            self.weight_grams[gram_key] = grams
        weighted_e2 = weighted_output_tile_scores_gram(delta2, grams).double()
        weighted_e0 = weighted_output_tile_scores_gram(delta0, grams).double()
        output_choice = weighted_e0 < weighted_e2

        selected_sse = torch.where(sse_choice, tile_e0, tile_e2)
        output_selected_sse = torch.where(output_choice, tile_e0, tile_e2)
        sse_selected_weighted = torch.where(sse_choice, weighted_e0, weighted_e2)
        output_selected_weighted = torch.where(output_choice, weighted_e0, weighted_e2)
        acc["tile_count"][at] += sse_choice.numel()
        acc["sse_tile_e0"][at] += sse_choice.sum()
        acc["output_tile_e0"][at] += output_choice.sum()
        acc["selector_agree"][at] += (sse_choice == output_choice).sum()
        acc["sse_e0_score"][at] += tile_e0.sum()
        acc["sse_e2_score"][at] += tile_e2.sum()
        acc["weighted_e0_score"][at] += weighted_e0.sum()
        acc["weighted_e2_score"][at] += weighted_e2.sum()
        acc["sse_selected_sse"][at] += selected_sse.sum()
        acc["output_selected_sse"][at] += output_selected_sse.sum()
        acc["sse_selected_weighted"][at] += sse_selected_weighted.sum()
        acc["output_selected_weighted"][at] += output_selected_weighted.sum()

        aligned = self._update_prompt_stats(
            x, blocks, nonzero, crest, exact_e0, hw_e0,
            tile_e0, selected_sse, sse_choice, output_choice, kt,
        )
        if aligned and not duplicate_call:
            self._update_temporal(layer, step, sse_choice, selected_sse, tile_e0, kt, m)

    def _update_global_block_histograms(
        self, blocks, valid, nonzero, crest, exact_e0, hw_e0, log_ratio,
        idx2, idx0, rel2, rel0,
    ):
        active_crest = crest[nonzero]
        fixed_edges = torch.tensor(CREST_EDGES[1:-1], device=blocks.device)
        crest_bin = torch.bucketize(active_crest, fixed_edges).clamp(0, 6)
        self.crest_counts += torch.bincount(crest_bin, minlength=7).double()
        self.crest_exact_wins += torch.bincount(
            crest_bin, weights=exact_e0[nonzero].double(), minlength=7
        )
        self.crest_hw_wins += torch.bincount(
            crest_bin, weights=hw_e0[nonzero].double(), minlength=7
        )
        self.crest_log_sum += torch.bincount(
            crest_bin, weights=log_ratio[nonzero].double(), minlength=7
        )

        absmax = blocks.abs().amax(-1, keepdim=True)
        normalized = torch.where(absmax > 0, blocks.abs() / absmax.clamp_min(1e-20), torch.zeros_like(blocks))
        norm_indices = (normalized[valid] * MAG_HIST_BINS).floor().long().clamp(0, MAG_HIST_BINS - 1)
        self.norm_hist += torch.bincount(norm_indices, minlength=MAG_HIST_BINS).double()
        self.e2_occupancy += torch.bincount(idx2[valid], minlength=8).double()
        self.e0_occupancy += torch.bincount(idx0[valid], minlength=8).double()

        scale_edges = torch.tensor(SCALE_ERROR_EDGES[1:-1], device=blocks.device)
        self.e2_scale_hist += torch.bincount(
            torch.bucketize(rel2[nonzero], scale_edges), minlength=len(SCALE_ERROR_EDGES) - 1
        ).double()
        self.e0_scale_hist += torch.bincount(
            torch.bucketize(rel0[nonzero], scale_edges), minlength=len(SCALE_ERROR_EDGES) - 1
        ).double()

        cidx = ((active_crest - 1.0) / 3.0 * SPEARMAN_CREST_BINS).floor().long().clamp(0, SPEARMAN_CREST_BINS - 1)
        lo, hi = SPEARMAN_LOG_RANGE
        lidx = ((log_ratio[nonzero].clamp(lo, hi) - lo) / (hi - lo) * SPEARMAN_LOG_BINS).floor().long().clamp(0, SPEARMAN_LOG_BINS - 1)
        joint = cidx * SPEARMAN_LOG_BINS + lidx
        self.crest_log_hist += torch.bincount(
            joint, minlength=SPEARMAN_CREST_BINS * SPEARMAN_LOG_BINS
        ).reshape(SPEARMAN_CREST_BINS, SPEARMAN_LOG_BINS).double()

    def _sample_layout(self, x, m):
        batch = len(self.current_prompt_indices)
        cfg_batch = 2 * batch
        if x.ndim < 2 or x.shape[0] != cfg_batch:
            self._exclude("cfg_batch_shape_not_alignable")
            return None
        rows_per_sample = m // cfg_batch
        if rows_per_sample * cfg_batch != m or rows_per_sample % TILE_ROWS:
            self._exclude("m_tiles_cross_prompt_boundary")
            return None
        prompt_indices = torch.tensor(
            self.current_prompt_indices * 2, device=x.device, dtype=torch.long
        )
        return cfg_batch, rows_per_sample, prompt_indices

    def _update_prompt_stats(
        self, x, blocks, nonzero, crest, exact_e0, hw_e0,
        tile_e0, selected_sse, sse_choice, output_choice, kt,
    ) -> bool:
        layout = self._sample_layout(x, blocks.shape[0])
        if layout is None:
            return False
        cfg_batch, rows, prompt_indices = layout
        nb = blocks.shape[1]
        prompt = self.prompt_acc
        values = {
            "nonzero_block_count": nonzero.reshape(cfg_batch, rows, nb).sum((1, 2)),
            "crest_sum": crest.reshape(cfg_batch, rows, nb).double().sum((1, 2)),
            "low_crest_count": ((crest < 2.0) & nonzero).reshape(cfg_batch, rows, nb).sum((1, 2)),
            "exact_e0_win": exact_e0.reshape(cfg_batch, rows, nb).sum((1, 2)),
            "hw_e0_win": hw_e0.reshape(cfg_batch, rows, nb).sum((1, 2)),
        }
        mts = rows // TILE_ROWS
        tile_shape = (cfg_batch, mts, kt)
        values.update({
            "tile_count": torch.full((cfg_batch,), mts * kt, device=x.device),
            "sse_e0_score": tile_e0.reshape(tile_shape).sum((1, 2)),
            "sse_selected_sse": selected_sse.reshape(tile_shape).sum((1, 2)),
            "selector_disagree": (sse_choice != output_choice).reshape(tile_shape).sum((1, 2)),
        })
        for name, value in values.items():
            prompt[name].index_add_(0, prompt_indices, value.double())
        return True

    def _update_temporal(self, layer, step, choice, selected_sse, e0_sse, kt, m):
        layout = self._sample_layout_tensor(choice.device, m)
        if layout is None:
            return
        cfg_batch, rows, prompt_indices = layout
        mts = rows // TILE_ROWS
        shape = (cfg_batch, mts, kt)
        current_choice = choice.reshape(shape)
        current_norm = selected_sse.sqrt().reshape(shape)
        current_gain = (e0_sse - selected_sse).reshape(shape)
        previous = self._previous.get(layer)
        if previous is not None:
            prev_step, prev_choice, prev_norm, run_length = previous
            if prev_step != step - 1 or prev_choice.shape != current_choice.shape:
                self._exclude("temporal_shape_or_step_mismatch")
            else:
                switched = current_choice != prev_choice
                e0_to_e2 = prev_choice & ~current_choice
                e2_to_e0 = ~prev_choice & current_choice
                stable = ~switched
                rel_change = (current_norm - prev_norm).abs() / prev_norm.clamp_min(1e-20)
                tr = self.transition_acc
                at = (layer, step)
                tr["tile_count"][at] += switched.numel()
                tr["flip_count"][at] += switched.sum()
                tr["e0_to_e2"][at] += e0_to_e2.sum()
                tr["e2_to_e0"][at] += e2_to_e0.sum()
                tr["switched_gain_sum"][at] += current_gain[switched].sum()
                tr["switched_gain_count"][at] += switched.sum()
                tr["stable_gain_sum"][at] += current_gain[stable].sum()
                tr["stable_gain_count"][at] += stable.sum()
                tr["error_norm_rel_change_sum"][at] += rel_change.sum()

                flat_switched = switched.reshape(cfg_batch, -1)
                p = self.prompt_acc
                p["transition_count"].index_add_(
                    0, prompt_indices,
                    torch.full((cfg_batch,), switched[0].numel(), device=choice.device, dtype=torch.float64),
                )
                p["flip_count"].index_add_(0, prompt_indices, flat_switched.sum(1).double())
                ended = torch.where(switched, run_length, torch.zeros_like(run_length))
                ended_count = switched.sum((1, 2))
                p["run_length_sum"].index_add_(0, prompt_indices, ended.sum((1, 2)).double())
                p["run_count"].index_add_(0, prompt_indices, ended_count.double())
                self.run_sum[layer] += ended.sum()
                self.run_count[layer] += switched.sum()
                run_length = torch.where(switched, torch.ones_like(run_length), run_length + 1)
                self._previous[layer] = (
                    step, current_choice.detach(), current_norm.detach(), run_length.detach()
                )
                return
        self._previous[layer] = (
            step,
            current_choice.detach(),
            current_norm.detach(),
            torch.ones_like(current_choice, dtype=torch.int16),
        )

    def _sample_layout_tensor(self, device, m):
        batch = len(self.current_prompt_indices)
        cfg_batch = 2 * batch
        rows = m // cfg_batch
        if rows * cfg_batch != m or rows % TILE_ROWS:
            return None
        prompt_indices = torch.tensor(self.current_prompt_indices * 2, device=device)
        return cfg_batch, rows, prompt_indices

    def _finalize_open_runs(self):
        if self.device is None:
            return
        prompt_indices = torch.tensor(
            self.current_prompt_indices * 2, device=self.device, dtype=torch.long
        )
        for layer, (_, _, _, run_length) in self._previous.items():
            cfg_batch = run_length.shape[0]
            sums = run_length.sum((1, 2)).double()
            counts = torch.full(
                (cfg_batch,), run_length[0].numel(), device=self.device, dtype=torch.float64
            )
            self.prompt_acc["run_length_sum"].index_add_(0, prompt_indices, sums)
            self.prompt_acc["run_count"].index_add_(0, prompt_indices, counts)
            self.run_sum[layer] += sums.sum()
            self.run_count[layer] += counts.sum()

    def finalize(self) -> dict:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if self.current_prompt_indices is not None:
            self.end_batch()
        if self.device is None:
            raise RuntimeError("distribution audit collected no observations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        layer = {name: tensor.cpu().numpy() for name, tensor in self.layer_acc.items()}
        trans = {name: tensor.cpu().numpy() for name, tensor in self.transition_acc.items()}
        prompt = {name: tensor.cpu().numpy() for name, tensor in self.prompt_acc.items()}
        run_sum = self.run_sum.cpu().numpy()
        run_count = self.run_count.cpu().numpy()
        global_arrays = {
            "crest_counts": self.crest_counts.cpu().numpy(),
            "crest_exact_wins": self.crest_exact_wins.cpu().numpy(),
            "crest_hw_wins": self.crest_hw_wins.cpu().numpy(),
            "crest_log_sum": self.crest_log_sum.cpu().numpy(),
            "norm_hist": self.norm_hist.cpu().numpy(),
            "e2_occupancy": self.e2_occupancy.cpu().numpy(),
            "e0_occupancy": self.e0_occupancy.cpu().numpy(),
            "e2_scale_hist": self.e2_scale_hist.cpu().numpy(),
            "e0_scale_hist": self.e0_scale_hist.cpu().numpy(),
            "crest_log_hist": self.crest_log_hist.cpu().numpy(),
        }
        layer_rows = self._layer_rows(layer)
        scale_rows = self._scale_rows(layer)
        temporal_rows = self._temporal_rows(trans, run_sum, run_count)
        block_rows = self._block_rows(global_arrays)
        prompt_rows = self._prompt_rows(prompt)
        quality_rows = self._quality_correlations(prompt_rows)
        self._write_csv(self.output_dir / "block_distribution.csv", block_rows)
        self._write_csv(self.output_dir / "layer_timestep_summary.csv", layer_rows)
        self._write_csv(self.output_dir / "scale_ablation.csv", scale_rows)
        self._write_csv(self.output_dir / "temporal_flip_summary.csv", temporal_rows)
        self._write_csv(self.output_dir / "prompt_level_summary.csv", prompt_rows)
        self._write_csv(self.output_dir / "quality_correlations.csv", quality_rows)
        self._plots(layer, trans, global_arrays)

        rho = _weighted_histogram_spearman(global_arrays["crest_log_hist"])
        provenance = {
            "model": self.model_name,
            "dataset": str(self.dataset_path),
            "dataset_prefix_count": len(self.samples),
            "activation_format_executed": "tile-mix-oracle",
            "output_oracle_role": "sidecar statistics only; never selected forward activation",
            "residual_rotation": "random",
            "cfg_layout": "unconditional batch followed by conditional batch",
            "temporal_identity": "prompt_id,cfg_branch,layer,M_tile,K_tile",
            "true_scheduler_timesteps": self.timestep_values,
            "crest_bins": list(CREST_EDGES),
            "low_crest_definition": "crest < 2 over nonzero blocks",
            "normalized_magnitude_bins": MAG_HIST_BINS,
            "block_error_ratio_epsilon": 1e-20,
            "crest_log_error_spearman": rho,
            "crest_log_error_spearman_method": (
                f"online {SPEARMAN_CREST_BINS}x{SPEARMAN_LOG_BINS} bivariate-histogram midrank approximation; "
                f"log2 ratio clipped to {SPEARMAN_LOG_RANGE}"
            ),
            "quality_correlation_bootstrap_samples": self.bootstrap_samples,
            "layers": len(self.layer_names),
            "excluded_alignment_events": self.exclusions,
            "quantized_cache": str(self.quantized_cache),
            "quantized_cache_sha256": _sha256(self.quantized_cache),
            "quality_csvs": [str(path) for path in self.quality_csvs],
            "saved_raw_activations": False,
            "saved_selector_maps": False,
        }
        with (self.output_dir / "provenance.json").open("w") as f:
            json.dump(provenance, f, indent=2)
            f.write("\n")
        return provenance

    def _layer_rows(self, a):
        rows = []
        for li, name in enumerate(self.layer_names):
            for si, timestep in enumerate(self.timestep_values):
                blocks = a["block_count"][li, si]
                if not blocks:
                    continue
                nz = a["nonzero_block_count"][li, si]
                elems = a["element_count"][li, si]
                tiles = a["tile_count"][li, si]
                rows.append({
                    "model": self.model_name, "layer": name, "timestep_index": si,
                    "timestep": timestep, "block_count": int(blocks),
                    "mean_crest": _safe_div(a["crest_sum"][li, si], nz),
                    "mean_kurtosis": _safe_div(a["kurtosis_sum"][li, si], nz),
                    "zero_rate": _safe_div(a["zero_element_count"][li, si], elems),
                    "e2_saturation_rate": _safe_div(a["e2_saturation_count"][li, si], elems),
                    "e0_saturation_rate": _safe_div(a["e0_saturation_count"][li, si], elems),
                    "mean_e2_scale_relative_error": _safe_div(a["e2_scale_rel_sum"][li, si], nz),
                    "mean_e0_scale_relative_error": _safe_div(a["e0_scale_rel_sum"][li, si], nz),
                    "mean_log2_e0_over_e2_hw_error": _safe_div(a["log_error_ratio_sum"][li, si], blocks),
                    "sse_tile_e0_ratio": _safe_div(a["sse_tile_e0"][li, si], tiles),
                    "output_tile_e0_ratio": _safe_div(a["output_tile_e0"][li, si], tiles),
                    "selector_agreement_rate": _safe_div(a["selector_agree"][li, si], tiles),
                    "sse_selector_sse_gain_vs_e0": a["sse_e0_score"][li, si] - a["sse_selected_sse"][li, si],
                    "output_selector_sse_gain_vs_e0": a["sse_e0_score"][li, si] - a["output_selected_sse"][li, si],
                    "sse_selector_weighted_gain_vs_e0": a["weighted_e0_score"][li, si] - a["sse_selected_weighted"][li, si],
                    "output_selector_weighted_gain_vs_e0": a["weighted_e0_score"][li, si] - a["output_selected_weighted"][li, si],
                })
        return rows

    def _scale_rows(self, a):
        rows = []
        for li, name in enumerate(self.layer_names):
            for si, timestep in enumerate(self.timestep_values):
                count = a["block_count"][li, si]
                if not count:
                    continue
                exact_adv = a["e2_exact_sse"][li, si] - a["e0_exact_sse"][li, si]
                hw_adv = a["e2_hw_sse"][li, si] - a["e0_hw_sse"][li, si]
                rounding = hw_adv - exact_adv
                rows.append({
                    "model": self.model_name, "layer": name, "timestep_index": si,
                    "timestep": timestep, "block_count": int(count),
                    "exact_e0_win_rate": a["exact_e0_win"][li, si] / count,
                    "rounded_e0_win_rate": a["hw_e0_win"][li, si] / count,
                    "winner_disagreement_rate": a["winner_disagree"][li, si] / count,
                    "rounding_e0_to_e2_rate": a["round_e0_to_e2"][li, si] / count,
                    "rounding_e2_to_e0_rate": a["round_e2_to_e0"][li, si] / count,
                    "e2_exact_sse": a["e2_exact_sse"][li, si],
                    "e0_exact_sse": a["e0_exact_sse"][li, si],
                    "e2_rounded_sse": a["e2_hw_sse"][li, si],
                    "e0_rounded_sse": a["e0_hw_sse"][li, si],
                    "exact_e0_error_advantage": exact_adv,
                    "rounded_e0_error_advantage": hw_adv,
                    "rounding_contribution_to_e0_advantage": rounding,
                    "rounding_contribution_fraction": _safe_div(rounding, hw_adv),
                })
        return rows

    def _temporal_rows(self, t, run_sum, run_count):
        rows = []
        for li, name in enumerate(self.layer_names):
            mean_run = _safe_div(run_sum[li], run_count[li])
            for si in range(1, self.num_steps):
                count = t["tile_count"][li, si]
                if not count:
                    continue
                rows.append({
                    "model": self.model_name, "layer": name,
                    "from_timestep_index": si - 1, "to_timestep_index": si,
                    "from_timestep": self.timestep_values[si - 1],
                    "to_timestep": self.timestep_values[si],
                    "tile_count": int(count),
                    "flip_rate": t["flip_count"][li, si] / count,
                    "e0_to_e2_rate": t["e0_to_e2"][li, si] / count,
                    "e2_to_e0_rate": t["e2_to_e0"][li, si] / count,
                    "mean_run_length_layer": mean_run,
                    "switched_tile_mean_sse_gain_vs_e0": _safe_div(
                        t["switched_gain_sum"][li, si], t["switched_gain_count"][li, si]
                    ),
                    "stable_tile_mean_sse_gain_vs_e0": _safe_div(
                        t["stable_gain_sum"][li, si], t["stable_gain_count"][li, si]
                    ),
                    "mean_error_norm_relative_change": _safe_div(
                        t["error_norm_rel_change_sum"][li, si], count
                    ),
                })
        return rows

    def _block_rows(self, g):
        rows = []
        total_crest = g["crest_counts"].sum()
        for i in range(7):
            count = g["crest_counts"][i]
            rows.append({
                "model": self.model_name, "distribution": "crest_factor",
                "bin_index": i, "bin_lower": CREST_EDGES[i], "bin_upper": CREST_EDGES[i + 1],
                "level": "", "count": int(count), "rate": _safe_div(count, total_crest),
                "exact_e0_win_rate": _safe_div(g["crest_exact_wins"][i], count),
                "rounded_e0_win_rate": _safe_div(g["crest_hw_wins"][i], count),
                "mean_log2_e0_over_e2_hw_error": _safe_div(g["crest_log_sum"][i], count),
            })
        total_norm = g["norm_hist"].sum()
        for i, count in enumerate(g["norm_hist"]):
            rows.append({
                "model": self.model_name, "distribution": "normalized_magnitude",
                "bin_index": i, "bin_lower": i / MAG_HIST_BINS,
                "bin_upper": (i + 1) / MAG_HIST_BINS, "level": "",
                "count": int(count), "rate": _safe_div(count, total_norm),
                "exact_e0_win_rate": "", "rounded_e0_win_rate": "",
                "mean_log2_e0_over_e2_hw_error": "",
            })
        for name, values, levels in (
            ("e2_level_occupancy", g["e2_occupancy"], E2M1_MAGNITUDES),
            ("e0_level_occupancy", g["e0_occupancy"], E0M3_MAGNITUDES),
        ):
            total = values.sum()
            for i, (count, level) in enumerate(zip(values, levels)):
                rows.append({
                    "model": self.model_name, "distribution": name,
                    "bin_index": i, "bin_lower": "", "bin_upper": "", "level": level,
                    "count": int(count), "rate": _safe_div(count, total),
                    "exact_e0_win_rate": "", "rounded_e0_win_rate": "",
                    "mean_log2_e0_over_e2_hw_error": "",
                })
        for name, values in (
            ("e2_scale_relative_error", g["e2_scale_hist"]),
            ("e0_scale_relative_error", g["e0_scale_hist"]),
        ):
            total = values.sum()
            for i, count in enumerate(values):
                rows.append({
                    "model": self.model_name, "distribution": name,
                    "bin_index": i, "bin_lower": SCALE_ERROR_EDGES[i],
                    "bin_upper": SCALE_ERROR_EDGES[i + 1], "level": "",
                    "count": int(count), "rate": _safe_div(count, total),
                    "exact_e0_win_rate": "", "rounded_e0_win_rate": "",
                    "mean_log2_e0_over_e2_hw_error": "",
                })
        return rows

    def _prompt_rows(self, p):
        rows = []
        for index, (image_id, info) in enumerate(self.samples):
            blocks = p["nonzero_block_count"][index]
            tiles = p["tile_count"][index]
            transitions = p["transition_count"][index]
            e0_sse = p["sse_e0_score"][index]
            rows.append({
                "prompt_index": index, "image_id": image_id, "category": info["category"],
                "mean_crest_factor": _safe_div(p["crest_sum"][index], blocks),
                "low_crest_block_ratio": _safe_div(p["low_crest_count"][index], blocks),
                "exact_e0_win_rate": _safe_div(p["exact_e0_win"][index], blocks),
                "rounded_e0_win_rate": _safe_div(p["hw_e0_win"][index], blocks),
                "format_flip_rate": _safe_div(p["flip_count"][index], transitions),
                "mean_run_length": _safe_div(p["run_length_sum"][index], p["run_count"][index]),
                "sse_tile_relative_activation_sse_gain_vs_e0": _safe_div(
                    e0_sse - p["sse_selected_sse"][index], e0_sse
                ),
                "sse_output_selector_disagreement_rate": _safe_div(
                    p["selector_disagree"][index], tiles
                ),
            })
        return rows

    def _quality_correlations(self, prompt_rows):
        from scipy.stats import spearmanr

        wanted = {"e0m3": {}, "tile-mix-oracle": {}}
        expected_ids = set(self.sample_index)
        for path in self.quality_csvs:
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row["config"] in wanted and row["image_id"] in expected_ids:
                        wanted[row["config"]][row["image_id"]] = row
        for config, by_id in wanted.items():
            if set(by_id) != expected_ids:
                raise RuntimeError(f"quality CSVs do not contain all {config} first64 rows")
        features = [
            "mean_crest_factor", "low_crest_block_ratio", "exact_e0_win_rate",
            "rounded_e0_win_rate", "format_flip_rate",
            "sse_tile_relative_activation_sse_gain_vs_e0",
            "sse_output_selector_disagreement_rate",
        ]
        metrics = ("lpips", "psnr", "ssim", "clip_score")
        rng = np.random.default_rng(20260811)
        rows = []
        ids = [row["image_id"] for row in prompt_rows]
        for feature in features:
            x = np.asarray([row[feature] for row in prompt_rows], dtype=np.float64)
            for metric in metrics:
                y = np.asarray([
                    float(wanted["tile-mix-oracle"][image_id][metric])
                    - float(wanted["e0m3"][image_id][metric])
                    for image_id in ids
                ])
                rho = float(spearmanr(x, y).statistic)
                boot = []
                for _ in range(self.bootstrap_samples):
                    idx = rng.integers(0, len(x), len(x))
                    value = spearmanr(x[idx], y[idx]).statistic
                    if math.isfinite(value):
                        boot.append(value)
                low, high = np.quantile(boot, (0.025, 0.975)) if boot else (math.nan, math.nan)
                rows.append({
                    "model": self.model_name, "feature": feature,
                    "quality_delta": f"tile-mix-oracle_minus_e0m3_{metric}",
                    "n": len(x), "spearman_rho": rho,
                    "bootstrap_ci95_low": float(low), "bootstrap_ci95_high": float(high),
                    "bootstrap_samples": self.bootstrap_samples,
                    "interpretation": "exploratory association; not causal",
                })
        return rows

    def _plots(self, layer, trans, g):
        from PIL import Image, ImageDraw

        centers = [(CREST_EDGES[i] + CREST_EDGES[i + 1]) / 2 for i in range(7)]
        labels = [f"[{CREST_EDGES[i]:g},{CREST_EDGES[i+1]:g}{']' if i == 6 else ')'}" for i in range(7)]
        counts = g["crest_counts"]
        exact = np.divide(
            g["crest_exact_wins"], counts,
            out=np.zeros_like(counts), where=counts > 0,
        )
        rounded = np.divide(
            g["crest_hw_wins"], counts,
            out=np.zeros_like(counts), where=counts > 0,
        )
        width, height = 1200, 700
        left, right, top, bottom = 110, 40, 70, 120
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((left, 20), f"{self.model_name}: crest factor vs E0 win", fill="black")
        draw.line((left, top, left, height - bottom), fill="black", width=2)
        draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
        for tick in range(6):
            y = height - bottom - tick * (height - top - bottom) / 5
            draw.line((left - 5, y, width - right, y), fill=(220, 220, 220), width=1)
            draw.text((35, y - 7), f"{tick / 5:.1f}", fill="black")

        xs = np.linspace(left + 20, width - right - 20, len(centers))
        def points(values):
            return [
                (float(x), float(height - bottom - value * (height - top - bottom)))
                for x, value in zip(xs, values)
            ]
        for values, color, label, offset in (
            (exact, (30, 100, 210), "exact scale", 0),
            (rounded, (220, 80, 40), "E4M3 rounded", 25),
        ):
            pts = points(values)
            draw.line(pts, fill=color, width=4)
            for x, y in pts:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            draw.line((width - 270, 35 + offset, width - 230, 35 + offset), fill=color, width=4)
            draw.text((width - 220, 28 + offset), label, fill="black")
        for x, label in zip(xs, labels):
            draw.text((x - 35, height - bottom + 20), label, fill="black")
        draw.text((10, top - 20), "P(E0 wins)", fill="black")
        image.save(self.output_dir / "crest_vs_e0_win.png")

        e0_ratio = np.divide(
            layer["sse_tile_e0"], layer["tile_count"],
            out=np.full_like(layer["tile_count"], np.nan), where=layer["tile_count"] > 0,
        )
        flip = np.divide(
            trans["flip_count"], trans["tile_count"],
            out=np.full_like(trans["tile_count"], np.nan), where=trans["tile_count"] > 0,
        )
        self._heatmap(e0_ratio, "SSE TileMix E0M3 ratio", "layer_timestep_e0_ratio_heatmap.png", 0, 1)
        self._heatmap(flip[:, 1:], "Adjacent-timestep SSE TileMix flip rate", "layer_timestep_flip_rate_heatmap.png", 0, np.nanpercentile(flip[:, 1:], 99))

    def _heatmap(self, values, title, filename, vmin, vmax):
        from PIL import Image, ImageDraw

        finite = np.isfinite(values)
        vmax = float(vmax) if math.isfinite(float(vmax)) and vmax > vmin else vmin + 1.0
        normalized = np.clip((values - vmin) / (vmax - vmin), 0, 1)
        normalized = np.where(finite, normalized, 0.0)
        # Compact blue→cyan→yellow diagnostic palette; missing cells are gray.
        red = np.clip(2.0 * normalized - 0.3, 0, 1)
        green = np.clip(1.7 * normalized, 0, 1)
        blue = np.clip(1.2 - 1.5 * normalized, 0, 1)
        rgb = (np.stack((red, green, blue), -1) * 255).astype(np.uint8)
        rgb[~finite] = (180, 180, 180)
        heat = Image.fromarray(rgb, mode="RGB")
        cell_width = 34
        cell_height = max(3, min(7, 1200 // len(self.layer_names)))
        heat = heat.resize(
            (values.shape[1] * cell_width, values.shape[0] * cell_height),
            Image.Resampling.NEAREST,
        )
        margin_left, margin_top, margin_bottom = 360, 60, 40
        canvas = Image.new(
            "RGB",
            (margin_left + heat.width + 30, margin_top + heat.height + margin_bottom),
            "white",
        )
        canvas.paste(heat, (margin_left, margin_top))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 15), f"{self.model_name}: {title}; range=[{vmin:.4g},{vmax:.4g}]", fill="black")
        stride = max(1, len(self.layer_names) // 35)
        for index in range(0, len(self.layer_names), stride):
            y = margin_top + index * cell_height
            draw.text((5, y), self.layer_names[index], fill="black")
        for index in range(values.shape[1]):
            x = margin_left + index * cell_width
            draw.text((x + 8, margin_top + heat.height + 8), str(index), fill="black")
        canvas.save(self.output_dir / filename)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]):
        if not rows:
            raise RuntimeError(f"refusing to write empty audit CSV: {path}")
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


__all__ = ["DistributionAuditCollector", "CREST_EDGES"]
