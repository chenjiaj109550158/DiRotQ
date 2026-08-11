"""Paper-faithful Four Over Six activation fake quantization.

The adaptive modes in this file follow Section 3/3.1 of Four Over Six:
one FP32 tensor scale maps the low-precision operand maximum to ``6 * 256``,
then each 1x16 block compares ordinary E2M1 candidates whose E4M3 scale maps
the block maximum to either 4 or 6.  The selected E4M3 scale fully encodes the
choice; no M=4/M=6 payload metadata or decoder branch is required.

This module is activation-only.  It does not alter weight quantization, GPTQ,
PCA, residual rotation, or the high-precision activation tail.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .tilemixfp4_utils import (
    E0M3_MAGNITUDES,
    E2M1_MAGNITUDES,
    FP4_BLOCK_SIZE,
    TILE_COLS,
    TILE_ROWS,
    _flatten_operand,
    _get_magnitudes,
    _round_e4m3_scale,
)


FOUR_OVER_SIX_GLOBAL_MAX = 6.0 * 256.0  # paper alpha = amax / 1536
STANDARD_NVFP4_GLOBAL_MAX = 6.0 * 448.0  # hardware NVFP4 alpha = amax / 2688
M6_MAX_BLOCK_SCALE = 256.0
M4_MAX_BLOCK_SCALE = 384.0

FOUR_OVER_SIX_FORMATS = {
    "nvfp4-4over6",
    "e0m3-gscale1536",
    "tile-mix-e0-e2-4over6",
}

_STAT_SCALARS = (
    "m4_count",
    "m6_count",
    "block_count",
    "m4_sse",
    "m6_sse",
    "adaptive_sse",
    "m6_gscale2688_sse",
    "m6_gscale1536_sse",
    "e0_sse",
    "e2_tile_sse",
    "e0_tile_sse",
    "tile_selected_sse",
    "e0_tile_count",
    "e2_tile_count",
    "tile_count",
    "m4_saturation_count",
    "m6_saturation_count",
    "selected_saturation_count",
    "valid_value_count",
    "signal_energy",
    "reconstruction_sse",
)


@dataclass
class FourOverSixBlockCandidates:
    """Inspectable candidates in the original activation domain."""

    m4: torch.Tensor
    m6: torch.Tensor
    selected: torch.Tensor
    choose_m4: torch.Tensor
    scale_m4: torch.Tensor
    scale_m6: torch.Tensor
    global_scale: torch.Tensor


class FourOverSixStats:
    """Streaming device-side sidecar for the three paper-faithful modes.

    Quantizers receive cheap layer-scoped views of one root collector.  Scalar
    and occupancy reductions remain on the activation device; host transfer
    happens once in :meth:`snapshot`.  A transformer pre-hook supplies the
    real scheduler timestep once per denoising step, rather than inferring it
    from linear-wrapper call counts.
    """

    def __init__(self, selection_unit: str = "block", *, _root=None, _layer=None):
        if _root is None:
            self._root = self
            self.selection_unit = selection_unit
            self._layer = None
            self._global: dict[str, torch.Tensor] | None = None
            self._layers: dict[str, dict[str, torch.Tensor]] = {}
            self._layer_timesteps: dict[tuple[str, int, float], dict[str, torch.Tensor]] = {}
            self._current_timestep: tuple[int, float] | None = None
            self._hook = None
            self._pipeline = None
            self._device = None
        else:
            self._root = _root
            self.selection_unit = _root.selection_unit
            self._layer = _layer

    def for_layer(self, layer_name: str) -> "FourOverSixStats":
        return FourOverSixStats(_root=self._root, _layer=layer_name)

    @staticmethod
    def _empty(device: torch.device) -> dict[str, torch.Tensor]:
        values = {
            name: torch.zeros((), dtype=torch.float64, device=device)
            for name in _STAT_SCALARS
        }
        values["m4_occupancy"] = torch.zeros(8, dtype=torch.float64, device=device)
        values["m6_occupancy"] = torch.zeros(8, dtype=torch.float64, device=device)
        return values

    def _check_device(self, device: torch.device) -> None:
        root = self._root
        if root._device is None:
            root._device = device
        elif root._device != device:
            raise RuntimeError("Four Over Six stats device changed; CPU fallback is forbidden")

    @staticmethod
    def _add(target: dict[str, torch.Tensor], values: dict[str, torch.Tensor]) -> None:
        for name, value in values.items():
            target[name].add_(value.detach().to(device=target[name].device, dtype=torch.float64))

    def record_sidecar(self, values: dict[str, torch.Tensor]) -> None:
        if not values:
            return
        device = next(iter(values.values())).device
        root = self._root
        root._check_device(device)
        if root._global is None:
            root._global = root._empty(device)
        root._add(root._global, values)
        if self._layer is None:
            return
        layer_acc = root._layers.setdefault(self._layer, root._empty(device))
        root._add(layer_acc, values)
        if root._current_timestep is not None:
            step, timestep = root._current_timestep
            key = (self._layer, step, timestep)
            step_acc = root._layer_timesteps.setdefault(key, root._empty(device))
            root._add(step_acc, values)

    def record_reconstruction(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> None:
        if original.shape != reconstructed.shape or original.device != reconstructed.device:
            raise ValueError("reconstruction stats require matching shape/device")
        self.record_sidecar({
            "signal_energy": original.float().square().sum(),
            "reconstruction_sse": (original.float() - reconstructed.float()).square().sum(),
        })

    def attach_timestep_source(self, transformer, pipeline) -> None:
        root = self._root
        if root._hook is not None:
            raise RuntimeError("Four Over Six timestep source is already attached")
        root._pipeline = pipeline

        def _pre_hook(module, args, kwargs):
            if "timestep" not in kwargs:
                raise RuntimeError("transformer forward did not expose a true timestep kwarg")
            passed = kwargs["timestep"].detach().float().flatten()
            if passed.numel() == 0 or not torch.allclose(passed, passed[:1]):
                raise RuntimeError("transformer timestep is empty or differs across CFG batch")
            scale = float(getattr(module.config, "timestep_scale", 1.0))
            true_value = float((passed[0] / scale).cpu())
            schedule = pipeline.scheduler.timesteps.detach().float().cpu()
            distances = (schedule - true_value).abs()
            step = int(distances.argmin())
            tolerance = 1e-4 * max(1.0, abs(float(schedule[step])))
            if float(distances[step]) > tolerance:
                raise RuntimeError(
                    f"passed timestep {true_value} does not match scheduler trajectory"
                )
            root._current_timestep = (step, float(schedule[step]))

        root._hook = transformer.register_forward_pre_hook(_pre_hook, with_kwargs=True)

    @staticmethod
    def _derived(values: dict[str, float | list[float]]) -> dict[str, float | list[float]]:
        out = dict(values)
        blocks = float(out["block_count"])
        tiles = float(out["tile_count"])
        valid = float(out["valid_value_count"])
        signal = float(out["signal_energy"])
        reconstruction = float(out["reconstruction_sse"])
        out["m4_ratio"] = float(out["m4_count"]) / blocks if blocks else 0.0
        out["e0_tile_ratio"] = float(out["e0_tile_count"]) / tiles if tiles else 0.0
        out["m4_saturation_rate"] = (
            float(out["m4_saturation_count"]) / valid if valid else 0.0
        )
        out["m6_saturation_rate"] = (
            float(out["m6_saturation_count"]) / valid if valid else 0.0
        )
        out["selected_saturation_rate"] = (
            float(out["selected_saturation_count"]) / valid if valid else 0.0
        )
        out["adaptive_gain_vs_m6_gscale1536"] = (
            float(out["m6_gscale1536_sse"]) - float(out["adaptive_sse"])
        )
        out["gscale1536_gain_vs_standard_m6"] = (
            float(out["m6_gscale2688_sse"]) - float(out["m6_gscale1536_sse"])
        )
        out["tile_gain_vs_fair_fixed_e0"] = (
            float(out["e0_tile_sse"]) - float(out["tile_selected_sse"])
        )
        if reconstruction == 0.0:
            out["qsnr_db"] = math.inf if signal > 0.0 else 0.0
        elif signal == 0.0:
            out["qsnr_db"] = -math.inf
        else:
            out["qsnr_db"] = 10.0 * math.log10(signal / reconstruction)
        return out

    @classmethod
    def _host(cls, values: dict[str, torch.Tensor]) -> dict[str, float | list[float]]:
        host = {
            name: (value.detach().cpu().tolist() if value.ndim else float(value.detach().cpu()))
            for name, value in values.items()
        }
        return cls._derived(host)

    def snapshot(self) -> dict:
        root = self._root
        if root._global is None:
            empty = root._empty(torch.device("cpu"))
            global_stats = self._host(empty)
        else:
            global_stats = self._host(root._global)
        per_layer = {name: self._host(values) for name, values in sorted(root._layers.items())}
        per_layer_timestep = []
        for (layer, step, timestep), values in sorted(
            root._layer_timesteps.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            per_layer_timestep.append({
                "layer": layer,
                "timestep_index": step,
                "timestep": timestep,
                **self._host(values),
            })
        return {
            "selection_unit": root.selection_unit,
            **global_stats,
            "per_layer": per_layer,
            "per_layer_timestep": per_layer_timestep,
        }


def _validate_clip_ratio(clip_ratio: float) -> None:
    if clip_ratio != 1.0:
        raise ValueError("Four Over Six modes require clip_ratio=1.0")


def _global_scale(flat: torch.Tensor, denominator: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale exactly the unpadded low-precision operand with one FP32 scalar."""
    if flat.numel() == 0:
        return flat, torch.ones((), dtype=torch.float32, device=flat.device)
    amax = flat.abs().amax()
    alpha = torch.where(amax == 0, torch.ones_like(amax), amax / denominator)
    return flat / alpha, alpha


def _pad_2d(flat: torch.Tensor, *, tile: bool) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = flat.shape
    row_multiple = TILE_ROWS if tile else 1
    col_multiple = TILE_COLS if tile else FP4_BLOCK_SIZE
    pad_m = (-m) % row_multiple
    pad_k = (-k) % col_multiple
    padded = F.pad(flat, (0, pad_k, 0, pad_m)) if (pad_m or pad_k) else flat
    valid = torch.zeros_like(padded, dtype=torch.bool)
    valid[:m, :k] = True
    return padded, valid


def _e2_candidate(
    blocks: torch.Tensor, maximum: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reconstruction, scale, E2M1 payload values, and level indices."""
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = _round_e4m3_scale(amax / maximum, amax == 0)
    normalized = blocks / scale
    codebook = _get_magnitudes(E2M1_MAGNITUDES, blocks.device)
    indices = torch.bucketize(
        normalized.abs().contiguous(), (codebook[:-1] + codebook[1:]) * 0.5
    )
    codes = normalized.sign() * codebook[indices]
    return codes * scale, scale, codes, indices


def _e0_candidate(
    blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = _round_e4m3_scale(amax / 7.0, amax == 0)
    normalized = blocks / scale
    codebook = _get_magnitudes(E0M3_MAGNITUDES, blocks.device)
    indices = torch.bucketize(
        normalized.abs().contiguous(), (codebook[:-1] + codebook[1:]) * 0.5
    )
    codes = normalized.sign() * codebook[indices]
    return codes * scale, scale, codes, indices


def _occupancy(indices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # Bin 8 receives padding and is discarded.  This replaces sixteen full
    # tensor equality reductions (eight per candidate) with one device pass.
    histogram_input = indices.masked_fill(~valid, 8).reshape(-1)
    return torch.bincount(histogram_input, minlength=9)[:8]


def _build_block_candidates(
    flat: torch.Tensor, *, tile_padding: bool
) -> dict[str, torch.Tensor]:
    scaled, alpha = _global_scale(flat, FOUR_OVER_SIX_GLOBAL_MAX)
    padded, valid = _pad_2d(scaled, tile=tile_padding)
    blocks = padded.reshape(padded.shape[0], -1, FP4_BLOCK_SIZE)
    valid_blocks = valid.reshape_as(blocks)
    q4, scale4, codes4, indices4 = _e2_candidate(blocks, 4.0)
    q6, scale6, codes6, indices6 = _e2_candidate(blocks, 6.0)
    qe0, scale_e0, codes_e0, indices_e0 = _e0_candidate(blocks)
    err4 = ((blocks - q4).square() * valid_blocks).sum(dim=-1)
    err6 = ((blocks - q6).square() * valid_blocks).sum(dim=-1)
    choose4 = err4 < err6  # paper pseudo-code: strict comparison, ties use M=6
    q_e2 = torch.where(choose4.unsqueeze(-1), q4, q6)
    codes_e2 = torch.where(choose4.unsqueeze(-1), codes4, codes6)
    scale_e2 = torch.where(choose4.unsqueeze(-1), scale4, scale6)
    return {
        "scaled": scaled,
        "alpha": alpha,
        "padded": padded,
        "valid": valid,
        "blocks": blocks,
        "valid_blocks": valid_blocks,
        "q4": q4,
        "q6": q6,
        "qe0": qe0,
        "codes4": codes4,
        "codes6": codes6,
        "codes_e0": codes_e0,
        "indices4": indices4,
        "indices6": indices6,
        "indices_e0": indices_e0,
        "scale4": scale4,
        "scale6": scale6,
        "scale_e0": scale_e0,
        "err4": err4,
        "err6": err6,
        "choose4": choose4,
        "q_e2": q_e2,
        "codes_e2": codes_e2,
        "scale_e2": scale_e2,
    }


def _to_native_finite(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct in the activation dtype without producing Inf on cast.

    With denominator 1536, an E0 block scale can round upward (for example
    1536/7 to 224), so an input exactly at the FP16/BF16 finite limit can
    reconstruct a few percent above that dtype's range.  Saturating only this
    final dtype boundary preserves the paper scale/codebook decision while
    meeting the fake-quantizer contract that finite inputs stay finite.
    """
    limit = torch.finfo(dtype).max
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(
        min=-limit, max=limit
    ).to(dtype)


def four_over_six_block_candidates(x: torch.Tensor) -> FourOverSixBlockCandidates:
    """Expose paper M=4/M=6 candidates for tests and diagnostics."""
    flat, shape = _flatten_operand(x)
    if flat.numel() == 0:
        empty = x.clone()
        return FourOverSixBlockCandidates(
            empty, empty, empty, torch.empty(0, dtype=torch.bool, device=x.device),
            torch.empty(0, device=x.device), torch.empty(0, device=x.device),
            torch.ones((), dtype=torch.float32, device=x.device),
        )
    m, k = flat.shape
    data = _build_block_candidates(flat, tile_padding=False)
    def restore(blocks):
        restored = (blocks.reshape(m, -1)[:, :k] * data["alpha"]).reshape(shape)
        return _to_native_finite(restored, x.dtype)
    return FourOverSixBlockCandidates(
        m4=restore(data["q4"]),
        m6=restore(data["q6"]),
        selected=restore(data["q_e2"]),
        choose_m4=data["choose4"],
        scale_m4=data["scale4"],
        scale_m6=data["scale6"],
        global_scale=data["alpha"],
    )


def _candidate_sse(
    source: torch.Tensor, reconstructed: torch.Tensor, valid: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    return ((source - reconstructed).square() * valid).sum() * alpha.square()


def _record_diagnostics(
    flat: torch.Tensor,
    data: dict[str, torch.Tensor],
    choose_e0: torch.Tensor,
    tile_err_e0: torch.Tensor,
    tile_err_e2: torch.Tensor,
    format_stats: FourOverSixStats | None,
) -> None:
    if format_stats is None:
        return
    m, k = flat.shape
    alpha = data["alpha"]
    blocks = data["blocks"]
    valid_blocks = data["valid_blocks"]
    selected = data["q_e2"]

    # Sidecar-only standard M=6 at denominator 2688, computed from this exact
    # unpadded low operand.  It never changes the forward trajectory.
    scaled_std, alpha_std = _global_scale(flat, STANDARD_NVFP4_GLOBAL_MAX)
    padded_std, valid_std = _pad_2d(scaled_std, tile=(data["padded"].shape[0] != m or data["padded"].shape[1] != ((k + 15) // 16) * 16))
    blocks_std = padded_std.reshape(padded_std.shape[0], -1, FP4_BLOCK_SIZE)
    valid_blocks_std = valid_std.reshape_as(blocks_std)
    q6_std, _, _, _ = _e2_candidate(blocks_std, 6.0)

    valid_block = valid_blocks.any(dim=-1)
    m6_saturation_map = (
        (blocks.abs() / data["scale6"] > 6.0) & valid_blocks
    )
    m6_saturation = m6_saturation_map.sum(dtype=torch.int64)
    selected_saturation = (
        m6_saturation_map & (~data["choose4"]).unsqueeze(-1)
    ).sum(dtype=torch.int64)

    tile_selected = torch.where(choose_e0, tile_err_e0, tile_err_e2)
    values = {
        "m4_count": (data["choose4"] & valid_block).sum(dtype=torch.int64),
        "m6_count": ((~data["choose4"]) & valid_block).sum(dtype=torch.int64),
        "block_count": valid_block.sum(dtype=torch.int64),
        "m4_sse": data["err4"].sum() * alpha.square(),
        "m6_sse": data["err6"].sum() * alpha.square(),
        "adaptive_sse": torch.minimum(data["err4"], data["err6"]).sum()
                        * alpha.square(),
        "m6_gscale2688_sse": _candidate_sse(
            blocks_std, q6_std, valid_blocks_std, alpha_std
        ),
        "m6_gscale1536_sse": data["err6"].sum() * alpha.square(),
        "e0_sse": tile_err_e0.sum() * alpha.square(),
        "e2_tile_sse": tile_err_e2.sum() * alpha.square(),
        "e0_tile_sse": tile_err_e0.sum() * alpha.square(),
        "tile_selected_sse": tile_selected.sum() * alpha.square(),
        "e0_tile_count": choose_e0.sum(dtype=torch.int64),
        "e2_tile_count": (~choose_e0).sum(dtype=torch.int64),
        "tile_count": torch.tensor(choose_e0.numel(), device=flat.device),
        # M=4 maps amax to 4; nearest E4M3 rounding cannot push its normalized
        # maximum past the E2M1 payload limit 6.  Keep the explicit zero field
        # so any future semantic change is visible in the schema/tests.
        "m4_saturation_count": torch.zeros((), device=flat.device),
        "m6_saturation_count": m6_saturation,
        "selected_saturation_count": selected_saturation,
        "valid_value_count": valid_blocks.sum(dtype=torch.int64),
        "m4_occupancy": _occupancy(data["indices4"], valid_blocks),
        "m6_occupancy": _occupancy(data["indices6"], valid_blocks),
    }
    format_stats.record_sidecar(values)


def _quantize(
    x: torch.Tensor,
    mode: str,
    *,
    clip_ratio: float = 1.0,
    format_stats: FourOverSixStats | None = None,
) -> torch.Tensor:
    _validate_clip_ratio(clip_ratio)
    flat, shape = _flatten_operand(x)
    if flat.numel() == 0:
        return x.clone()
    m, k = flat.shape
    data = _build_block_candidates(flat, tile_padding=True)
    padded = data["padded"]
    valid = data["valid"]
    mp, kp = padded.shape
    q_e2 = data["q_e2"].reshape(mp, kp)
    q_e0 = data["qe0"].reshape(mp, kp)
    mt, kt = mp // TILE_ROWS, kp // TILE_COLS
    tile_err_e2 = ((padded - q_e2).square() * valid).reshape(
        mt, TILE_ROWS, kt, TILE_COLS
    ).sum(dim=(1, 3))
    tile_err_e0 = ((padded - q_e0).square() * valid).reshape(
        mt, TILE_ROWS, kt, TILE_COLS
    ).sum(dim=(1, 3))
    choose_e0 = tile_err_e0 < tile_err_e2  # tile tie selects adaptive E2M1

    _record_diagnostics(
        flat, data, choose_e0, tile_err_e0, tile_err_e2, format_stats
    )

    if mode == "nvfp4-4over6":
        selected = q_e2
    elif mode == "e0m3-gscale1536":
        selected = q_e0
    elif mode == "tile-mix-e0-e2-4over6":
        selected = torch.where(
            choose_e0[:, None, :, None],
            q_e0.reshape(mt, TILE_ROWS, kt, TILE_COLS),
            q_e2.reshape(mt, TILE_ROWS, kt, TILE_COLS),
        ).reshape(mp, kp)
    else:
        raise ValueError(f"unsupported Four Over Six activation format: {mode}")

    out = (selected[:m, :k] * data["alpha"]).reshape(shape)
    return _to_native_finite(out, x.dtype)


def fake_quantize_nvfp4_4over6(
    x: torch.Tensor,
    clip_ratio: float = 1.0,
    format_stats: FourOverSixStats | None = None,
) -> torch.Tensor:
    return _quantize(
        x, "nvfp4-4over6", clip_ratio=clip_ratio, format_stats=format_stats
    )


def fake_quantize_e0m3_gscale1536(
    x: torch.Tensor,
    clip_ratio: float = 1.0,
    format_stats: FourOverSixStats | None = None,
) -> torch.Tensor:
    return _quantize(
        x, "e0m3-gscale1536", clip_ratio=clip_ratio, format_stats=format_stats
    )


def fake_quantize_tile_mix_e0_e2_4over6(
    x: torch.Tensor,
    clip_ratio: float = 1.0,
    format_stats: FourOverSixStats | None = None,
) -> torch.Tensor:
    return _quantize(
        x,
        "tile-mix-e0-e2-4over6",
        clip_ratio=clip_ratio,
        format_stats=format_stats,
    )


def fake_quantize_four_over_six_activation(
    x: torch.Tensor,
    activation_format: str,
    clip_ratio: float = 1.0,
    format_stats: FourOverSixStats | None = None,
) -> torch.Tensor:
    return _quantize(
        x, activation_format, clip_ratio=clip_ratio, format_stats=format_stats
    )


__all__ = [
    "FOUR_OVER_SIX_FORMATS",
    "FOUR_OVER_SIX_GLOBAL_MAX",
    "M4_MAX_BLOCK_SCALE",
    "M6_MAX_BLOCK_SCALE",
    "FourOverSixBlockCandidates",
    "FourOverSixStats",
    "four_over_six_block_candidates",
    "fake_quantize_nvfp4_4over6",
    "fake_quantize_e0m3_gscale1536",
    "fake_quantize_tile_mix_e0_e2_4over6",
    "fake_quantize_four_over_six_activation",
]
