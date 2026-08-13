"""Streaming trajectory statistics for the existing SSE TileMix selector.

This module is deliberately observational: it consumes the selector decision
and the two candidate errors already computed by ``tilemixfp4_utils``.  It
never recomputes a format choice and never changes a fake-quantized tensor.
Reductions stay on the activation device until :meth:`snapshot` is called.
"""

from __future__ import annotations

import math
from typing import Any

import torch


_FIELDS = (
    "e0m3_count",
    "e2m1_count",
    "all_e0_sse",
    "all_e2_sse",
    "selected_sse",
    "flip_count",
    "flip_total",
    "e0_to_e2",
    "e2_to_e0",
)

ASYMMETRIC_FIXED_E0_WEIGHT_ACTIVATIONS = frozenset(
    {"e0m3", "nvfp4-hw", "tile-mix-oracle"}
)


def validate_fixed_e0_weight_activation_format(activation_format: str) -> None:
    """Restrict the matched fixed-E0-weight control matrix.

    The weight cache kind remains an explicit, activation-independent runtime
    argument.  This validator only prevents accidentally routing legacy,
    Four-Over-Six, BlockMix, or output-aware activation experiments through
    the Pilot32 command matrix.
    """
    if activation_format not in ASYMMETRIC_FIXED_E0_WEIGHT_ACTIVATIONS:
        allowed = ", ".join(sorted(ASYMMETRIC_FIXED_E0_WEIGHT_ACTIVATIONS))
        raise ValueError(
            f"hardware fixed-weight runtime activation must be one of: {allowed}"
        )


def _new_counter(device: torch.device) -> torch.Tensor:
    return torch.zeros(len(_FIELDS), dtype=torch.float64, device=device)


class _LayerStats:
    def __init__(self, root: "TileMixTrajectoryStats", layer_name: str):
        self._root = root
        self.layer_name = layer_name

    def record_tilemix(self, *args, **kwargs) -> None:
        self._root._record_tilemix(self.layer_name, *args, **kwargs)

    def record(self, choose_e0: torch.Tensor) -> None:
        # Kept for compatibility with generic format-stat plumbing.  The
        # TileMix quantizer calls record_tilemix when it is available.
        self._root._record_choice_only(self.layer_name, choose_e0)

    def record_reconstruction(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> None:
        self._root._record_reconstruction(original, reconstructed)


class TileMixTrajectoryStats:
    """Online aggregate/per-layer/per-timestep statistics for SSE TileMix.

    ``require_cuda`` protects real runs from an accidental host fallback.  It
    is disabled only by small CPU unit tests.  The one retained selector map is
    the preceding timestep for the current batch/layer, which is the minimum
    state needed to compute the requested format flip rate.
    """

    selection_unit = "tile"

    def __init__(self, selection_unit: str = "tile", *, require_cuda: bool = True):
        if selection_unit != "tile":
            raise ValueError("TileMixTrajectoryStats requires tile selection units")
        self.require_cuda = require_cuda
        self._device: torch.device | None = None
        self._global: torch.Tensor | None = None
        self._layers: dict[str, torch.Tensor] = {}
        self._timesteps: dict[int, torch.Tensor] = {}
        self._layer_timesteps: dict[tuple[str, int], torch.Tensor] = {}
        self._prompts: dict[str, torch.Tensor] = {}
        self._current_prompt_ids: list[str] | None = None
        self._current_timestep: int | None = None
        self._previous: dict[str, tuple[int, torch.Tensor]] = {}
        self._signal_error: torch.Tensor | None = None
        self._hook = None

    def for_layer(self, layer_name: str) -> _LayerStats:
        return _LayerStats(self, layer_name)

    def attach_timestep_source(self, transformer, pipeline) -> None:
        """Read the true scheduler timestep passed to the transformer."""
        if self._hook is not None:
            raise RuntimeError("timestep source is already attached")

        def _capture_timestep(_module, args, kwargs):
            timestep = kwargs.get("timestep")
            if timestep is None and len(args) > 1:
                timestep = args[1]
            if timestep is None:
                raise RuntimeError("transformer call did not expose a true timestep")
            if isinstance(timestep, torch.Tensor):
                if timestep.numel() == 0:
                    raise RuntimeError("empty transformer timestep")
                value = int(timestep.detach().reshape(-1)[0].cpu())
            else:
                value = int(timestep)
            self._current_timestep = value

        self._hook = transformer.register_forward_pre_hook(
            _capture_timestep, with_kwargs=True
        )

    def start_batch(self, batch) -> None:
        if self._current_prompt_ids is not None:
            raise RuntimeError("a statistics batch is already active")
        self._current_prompt_ids = [str(image_id) for image_id, _ in batch]
        self._previous.clear()
        self._current_timestep = None

    def end_batch(self) -> None:
        self._current_prompt_ids = None
        self._previous.clear()
        self._current_timestep = None

    def _ensure_device(self, device: torch.device) -> None:
        if self.require_cuda and device.type != "cuda":
            raise RuntimeError("TileMix trajectory statistics refuse CPU fallback")
        if self._device is None:
            self._device = device
            self._global = _new_counter(device)
            self._signal_error = torch.zeros(2, dtype=torch.float64, device=device)
        elif self._device != device:
            raise RuntimeError("TileMix trajectory-stat device changed during generation")

    def _counter(self, table: dict, key: Any) -> torch.Tensor:
        counter = table.get(key)
        if counter is None:
            assert self._device is not None
            counter = _new_counter(self._device)
            table[key] = counter
        return counter

    def _record_choice_only(self, layer_name: str, choose_e0: torch.Tensor) -> None:
        raise RuntimeError(
            f"{layer_name}: TileMix stats require candidate errors; "
            "record_tilemix was not called"
        )

    def _record_reconstruction(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> None:
        if original.shape != reconstructed.shape or original.device != reconstructed.device:
            raise ValueError("reconstruction tensors must have matching shape/device")
        self._ensure_device(original.device)
        delta = torch.stack(
            (original.float().square().sum(),
             (original - reconstructed).float().square().sum())
        ).to(torch.float64)
        assert self._signal_error is not None
        self._signal_error.add_(delta)

    @staticmethod
    def _delta(
        choose_e0: torch.Tensor,
        err_e0: torch.Tensor,
        err_e2: torch.Tensor,
        previous: torch.Tensor | None,
    ) -> torch.Tensor:
        selected = torch.where(choose_e0, err_e0, err_e2)
        e0_count = choose_e0.sum(dtype=torch.int64)
        e2_count = choose_e0.numel() - e0_count
        values = [
            e0_count,
            e2_count,
            err_e0.sum(),
            err_e2.sum(),
            selected.sum(),
        ]
        if previous is None:
            values.extend((0, 0, 0, 0))
        else:
            values.extend((
                (choose_e0 != previous).sum(dtype=torch.int64),
                choose_e0.numel(),
                (previous & ~choose_e0).sum(dtype=torch.int64),
                (~previous & choose_e0).sum(dtype=torch.int64),
            ))
        return torch.stack([torch.as_tensor(v, device=choose_e0.device) for v in values]).to(
            torch.float64
        )

    def _record_tilemix(
        self,
        layer_name: str,
        choose_e0: torch.Tensor,
        err_e0: torch.Tensor,
        err_e2: torch.Tensor,
        *,
        original_shape,
    ) -> None:
        if choose_e0.dtype != torch.bool:
            raise TypeError("choose_e0 must be boolean")
        if choose_e0.shape != err_e0.shape or choose_e0.shape != err_e2.shape:
            raise ValueError("selector and candidate error tensors must match")
        if choose_e0.device != err_e0.device or choose_e0.device != err_e2.device:
            raise ValueError("selector and candidate errors must share a device")
        if self._current_timestep is None:
            raise RuntimeError("true scheduler timestep is unavailable")
        self._ensure_device(choose_e0.device)
        step = self._current_timestep
        prior_entry = self._previous.get(layer_name)
        previous = None
        if prior_entry is not None:
            previous_step, previous_map = prior_entry
            if previous_step == step:
                raise RuntimeError(f"duplicate TileMix call for {layer_name} at timestep {step}")
            if previous_map.shape == choose_e0.shape:
                previous = previous_map
        delta = self._delta(choose_e0, err_e0, err_e2, previous)
        assert self._global is not None
        self._global.add_(delta)
        self._counter(self._layers, layer_name).add_(delta)
        self._counter(self._timesteps, step).add_(delta)
        self._counter(self._layer_timesteps, (layer_name, step)).add_(delta)
        self._previous[layer_name] = (step, choose_e0.detach().clone())
        self._record_prompts(choose_e0, err_e0, err_e2, original_shape)

    def _record_prompts(
        self,
        choose_e0: torch.Tensor,
        err_e0: torch.Tensor,
        err_e2: torch.Tensor,
        original_shape,
    ) -> None:
        prompt_ids = self._current_prompt_ids
        if not prompt_ids:
            return
        shape = tuple(int(v) for v in original_shape)
        if len(shape) < 2 or shape[0] % len(prompt_ids):
            raise RuntimeError(
                f"cannot map activation shape {shape} to {len(prompt_ids)} prompts"
            )
        rows_per_item = math.prod(shape[1:-1]) if len(shape) > 2 else 1
        if rows_per_item % 16:
            raise RuntimeError(
                f"per-item activation row count {rows_per_item} is not tile aligned"
            )
        row_tiles = rows_per_item // 16
        if choose_e0.shape[0] != shape[0] * row_tiles:
            raise RuntimeError("padded M tiles cannot be mapped losslessly to prompts")
        kt = choose_e0.shape[1]
        choices = choose_e0.reshape(shape[0], row_tiles, kt)
        e0_errors = err_e0.reshape_as(choices)
        e2_errors = err_e2.reshape_as(choices)
        repeats = shape[0] // len(prompt_ids)
        for prompt_index, prompt_id in enumerate(prompt_ids):
            indices = torch.arange(
                prompt_index, shape[0], len(prompt_ids), device=choose_e0.device
            )
            if indices.numel() != repeats:
                raise RuntimeError("unexpected CFG/prompt batch layout")
            c = choices.index_select(0, indices)
            e0 = e0_errors.index_select(0, indices)
            e2 = e2_errors.index_select(0, indices)
            prompt_delta = self._delta(c, e0, e2, None)
            # Prompt flip fields are intentionally zero: prompt-level output asks
            # only for ratio and SSE gain, while flips are aggregated exactly via
            # the layer-aligned maps above.
            self._counter(self._prompts, prompt_id).add_(prompt_delta)

    @staticmethod
    def _to_record(counter: torch.Tensor) -> dict[str, int | float]:
        values = counter.detach().cpu().tolist()
        record = dict(zip(_FIELDS, values))
        for key in ("e0m3_count", "e2m1_count", "flip_count", "flip_total",
                    "e0_to_e2", "e2_to_e0"):
            record[key] = int(round(record[key]))
        total = record["e0m3_count"] + record["e2m1_count"]
        record["total_count"] = total
        record["e0m3_ratio"] = record["e0m3_count"] / total if total else 0.0
        record["flip_rate"] = (
            record["flip_count"] / record["flip_total"]
            if record["flip_total"] else 0.0
        )
        e0_sse = record["all_e0_sse"]
        e2_sse = record["all_e2_sse"]
        selected = record["selected_sse"]
        record["selected_reduction_vs_e0"] = (
            (e0_sse - selected) / e0_sse if e0_sse > 0 else 0.0
        )
        record["selected_reduction_vs_e2"] = (
            (e2_sse - selected) / e2_sse if e2_sse > 0 else 0.0
        )
        return record

    def snapshot(self) -> dict[str, Any]:
        if self._global is None:
            global_record = self._to_record(torch.zeros(len(_FIELDS)))
        else:
            global_record = self._to_record(self._global)
        if self._signal_error is None:
            signal, reconstruction = 0.0, 0.0
        else:
            signal, reconstruction = self._signal_error.detach().cpu().tolist()
        qsnr = (
            10.0 * math.log10(signal / reconstruction)
            if signal > 0 and reconstruction > 0
            else (math.inf if signal > 0 else 0.0)
        )
        return {
            "selection_unit": "tile",
            **global_record,
            "signal_energy": signal,
            "reconstruction_sse": reconstruction,
            "qsnr_db": qsnr,
            "per_layer": {
                key: self._to_record(value) for key, value in sorted(self._layers.items())
            },
            "per_timestep": {
                str(key): self._to_record(value)
                for key, value in sorted(self._timesteps.items())
            },
            "per_layer_timestep": {
                f"{layer}|{step}": self._to_record(value)
                for (layer, step), value in sorted(self._layer_timesteps.items())
            },
            "per_prompt": {
                key: self._to_record(value) for key, value in sorted(self._prompts.items())
            },
        }


__all__ = [
    "ASYMMETRIC_FIXED_E0_WEIGHT_ACTIVATIONS",
    "TileMixTrajectoryStats",
    "validate_fixed_e0_weight_activation_format",
]
