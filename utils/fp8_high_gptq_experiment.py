"""Formal plumbing for fit-only high-weight FP8 GPTQ on SANA rank-3r.

The existing rank-3r low-E0 cache is immutable.  New artifacts contain only
high Hessians and legal FP8 payload/scale sidecars.  Evaluation materializes a
temporary BF16 fake-quant state in host memory and never writes a duplicate
full transformer cache.
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

from .e0joint_gptq import extract_fused_low_weight, sha256_file
from .fp8_high_e0_low import tensor_sha256
from .fp8_high_e0_low_experiment import (
    REQUIRED_ACTIVE_LAYERS,
    evaluate_teacher_cache,
    stitch_low_high,
    unquantized_split_parity,
)
from .fp8_high_gptq import (
    E4_SCALE_MULTIPLIERS,
    choose_e4_per_channel_scales,
    choose_mx_scale_bytes,
    decode_high_weight_record,
    hessian_weighted_error,
    make_high_weight_record,
    quantize_e4_per_channel_gptq,
    quantize_e4_per_channel_rtn,
    quantize_mx_weight_gptq,
    quantize_mx_weight_rtn,
    serialized_high_record_bytes,
)
from .quant_utils import ActQuantWrapper, _rotate_and_split_W, find_qlayers


HIGH_HESSIAN_SCHEMA = "dirotq.fp8_high_hessian.v1"
HIGH_SIDECAR_SCHEMA = "dirotq.fp8_high_weight_collection.v1"
METHOD_FREEZE_SCHEMA = "dirotq.fp8_high_gptq_method_freeze.v1"
FORMAL_RECIPES = (
    "e4-pc-rtn", "e4-pc-gptq", "mx-best-rtn", "mx-neighbor-gptq",
)
MX_RTN_CANDIDATES = ("current", "nosat", "neighbor")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _active_layers(transformer, names: Iterable[str] | None = None) -> dict[str, ActQuantWrapper]:
    active = {
        name: layer for name, layer in find_qlayers(
            transformer, layers=[ActQuantWrapper]
        ).items() if layer.quantizer.bits < 16
    }
    if len(active) != REQUIRED_ACTIVE_LAYERS:
        raise RuntimeError(f"expected 120 active wrapped layers, found {len(active)}")
    if names is None:
        return active
    requested = list(names)
    missing = sorted(set(requested) - set(active))
    if missing:
        raise KeyError(f"unknown active layer(s): {missing}")
    return {name: active[name] for name in requested}


def project_high_activation(layer: ActQuantWrapper, x: torch.Tensor) -> torch.Tensor:
    """Project pre-wrapper input to the exact contiguous high runtime layout."""
    if x.device != layer.module.weight.device:
        raise RuntimeError("high-Hessian projection forbids CPU/device fallback")
    dtype = layer.module.weight.dtype
    batch = x.shape[0]
    if layer.rotation is not None:
        rotation = layer.rotation.to(device=x.device, dtype=dtype)
        flat = x.to(dtype).reshape(-1, x.shape[-1])
        rotated = flat @ rotation
        high = int(layer.quantizer.high_bits_length)
        if high <= 0:
            raise ValueError("rank-3r high projection has no protected channels")
        return rotated[:, -high:].reshape(batch, -1, high)
    if layer.rotation_per_head is not None:
        heads, head_dim = int(layer.num_heads), int(layer.head_dim)
        high = int(layer.quantizer.high_bits_length)
        if high <= 0 or high >= head_dim:
            raise ValueError("invalid per-head high rank")
        rotation = layer.rotation_per_head.to(device=x.device, dtype=dtype)
        shaped = x.to(dtype).reshape(batch, -1, heads, head_dim)
        rotated = torch.einsum("bmhd,hde->bmhe", shaped, rotation)
        # Concatenation is per-head and never mixes head bases.
        return rotated[..., -high:].reshape(batch, -1, heads * high)
    raise ValueError("high-Hessian collection requires PCA/random residual rotation")


def high_target_hashes(transformer, names: Iterable[str] | None = None) -> dict[str, str]:
    output = {}
    for name, layer in _active_layers(transformer, names).items():
        _, high, _, _ = _rotate_and_split_W(layer, layer.module.weight.detach())
        output[name] = tensor_sha256(high.to(layer.module.weight.dtype))
    return output


def high_hessian_metadata(
    *, source_commit: str, fit_manifest: Path, basis_path: Path,
    rotation_path: Path, target_hashes: dict[str, str], damping: float,
) -> dict:
    return {
        "schema": HIGH_HESSIAN_SCHEMA,
        "source_commit": source_commit,
        "fit_manifest_sha256": sha256_file(fit_manifest),
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
        "high_weight_target_hashes": target_hashes,
        "normalization": "2/sum_rows * sum(A_high.T @ A_high)",
        "projection_dtype": "model-native-bfloat16",
        "damping_pct": float(damping),
        "residual_rotation": "matched-rank-3r-random",
    }


def _move_fit_batch(batch: list[dict], device: str, dtype: torch.dtype):
    latents = torch.cat([item["input_args"][0] for item in batch], dim=0).to(
        device=device, dtype=dtype
    )
    kwargs = {}
    for key in batch[0]["input_kwargs"]:
        values = [item["input_kwargs"][key] for item in batch]
        first = values[0]
        if isinstance(first, torch.Tensor):
            value = torch.cat(values, dim=0) if first.ndim and first.shape[0] == 1 else first
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
            kwargs[key] = value
        else:
            kwargs[key] = first
    return latents, kwargs


@torch.inference_mode()
def collect_high_activation_hessians(
    transformer,
    fit_dir: Path,
    *,
    batch_size: int = 4,
    layer_names: Iterable[str] | None = None,
    device: str = "cuda",
) -> tuple[dict[str, torch.Tensor], dict]:
    """Replay frozen fit calls and accumulate high Hessians online."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal high-Hessian collection requires CUDA")
    layers = _active_layers(transformer, layer_names)
    dtype = next(transformer.parameters()).dtype
    hessians = {}
    rows = {name: 0 for name in layers}
    chunks = {name: 0 for name in layers}
    for name, layer in layers.items():
        if layer.rotation is not None:
            high = int(layer.quantizer.high_bits_length)
        else:
            high = int(layer.num_heads) * int(layer.quantizer.high_bits_length)
        hessians[name] = torch.zeros(high, high, device=device, dtype=torch.float32)

    hooks = []
    for name, layer in layers.items():
        def make_hook(layer_name, qlayer):
            def hook(_module, inputs, _output):
                high = project_high_activation(qlayer, inputs[0].detach())
                flat = high.reshape(-1, high.shape[-1]).float()
                if not torch.isfinite(flat).all():
                    raise RuntimeError(f"{layer_name}: non-finite projected high activation")
                hessians[layer_name].addmm_(flat.T, flat)
                rows[layer_name] += flat.shape[0]
                chunks[layer_name] += high.shape[0]
            return hook
        hooks.append(layer.register_forward_hook(make_hook(name, layer)))

    files = sorted(fit_dir.glob("*.pt"))
    if not files:
        raise RuntimeError(f"no fit calls in {fit_dir}")
    batches = [files[index:index + batch_size] for index in range(0, len(files), batch_size)]

    def load_batch(paths):
        return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            queue = deque()
            iterator = iter(batches)
            for _ in range(2):
                try:
                    queue.append(pool.submit(load_batch, next(iterator)))
                except StopIteration:
                    break
            for _ in tqdm(range(len(batches)), desc="high activation Hessian calibration"):
                try:
                    queue.append(pool.submit(load_batch, next(iterator)))
                except StopIteration:
                    pass
                batch = queue.popleft().result()
                latents, kwargs = _move_fit_batch(batch, device, dtype)
                transformer(latents, **kwargs)
                del latents, kwargs, batch
    finally:
        for hook in hooks:
            hook.remove()

    output = {}
    for name in layers:
        if chunks[name] != len(files) or rows[name] <= 0:
            raise RuntimeError(
                f"{name}: incomplete high Hessian chunks={chunks[name]}, rows={rows[name]}"
            )
        hessians[name].mul_(2.0 / rows[name])
        if not torch.isfinite(hessians[name]).all():
            raise RuntimeError(f"{name}: non-finite high Hessian")
        output[name] = hessians[name].cpu()
    return output, {
        "fit_calls": len(files), "batches": len(batches), "rows_by_layer": rows,
        "chunks_by_layer": chunks, "seconds": time.perf_counter() - started,
        "hessian_bytes": sum(value.numel() * value.element_size() for value in output.values()),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def write_high_hessian_cache(
    path: Path, hessians: dict[str, torch.Tensor], metadata: dict,
) -> dict:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite high Hessian cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale incomplete Hessian cache: {temporary}")
    torch.save({"schema": HIGH_HESSIAN_SCHEMA, "metadata": metadata,
                "hessians": hessians}, temporary)
    temporary.rename(path)
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def load_high_hessian_cache(path: Path, expected_metadata: dict) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != HIGH_HESSIAN_SCHEMA:
        raise RuntimeError("invalid high Hessian cache schema")
    observed = payload.get("metadata", {})
    for key, value in expected_metadata.items():
        if observed.get(key) != value:
            raise RuntimeError(
                f"high Hessian provenance mismatch for {key}: {observed.get(key)!r} != {value!r}"
            )
    return payload["hessians"]


def _result_diagnostics(source, result, hessian) -> dict:
    reconstructed = result.reconstructed.float()
    raw = (source.float() - reconstructed).double().square().sum()
    weighted = hessian_weighted_error(source, reconstructed, hessian)
    payload_elements = source.numel()
    scale_tensor = result.scales if result.scales is not None else result.scale_bytes
    payload_histogram = torch.bincount(
        result.payload.reshape(-1).long().cpu(), minlength=256
    ).tolist()
    diagnostics = {
        "raw_sse": float(raw),
        "relative_sse": float(raw / source.double().square().sum().clamp_min(1e-30)),
        "hessian_weighted_error": float(weighted),
        "saturation_count": int(result.saturation_count),
        "saturation_rate": int(result.saturation_count) / payload_elements,
        "payload_sha256": tensor_sha256(result.payload),
        "scale_sha256": tensor_sha256(scale_tensor),
        "payload_mismatch_vs_rtn": int(result.payload_mismatch_vs_rtn),
        "gptq_status": result.gptq_status,
        "gptq_attempts": result.attempts,
        "payload_occupancy": {
            str(index): int(count) for index, count in enumerate(payload_histogram) if count
        },
    }
    if result.scales is not None:
        scales = result.scales.float()
        diagnostics.update({
            "scale_min": float(scales.min()), "scale_max": float(scales.max()),
            "scale_mean": float(scales.mean()), "scale_exponent_histogram": None,
        })
    else:
        exponents = result.scale_bytes.to(torch.int16) - 127
        histogram = torch.bincount((exponents + 127).reshape(-1).long().cpu(), minlength=255)
        diagnostics.update({
            "scale_min": float(torch.pow(torch.tensor(2.0), exponents.float()).min()),
            "scale_max": float(torch.pow(torch.tensor(2.0), exponents.float()).max()),
            "scale_mean": float(torch.pow(torch.tensor(2.0), exponents.float()).mean()),
            "scale_exponent_histogram": {
                str(index - 127): int(count)
                for index, count in enumerate(histogram.tolist()) if count
            },
        })
    return diagnostics


@torch.inference_mode()
def build_high_weight_sidecars(
    transformer,
    hessians: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    common_metadata: dict,
    layer_names: Iterable[str] | None = None,
    damp_pct: float = .01,
    require_cuda: bool = True,
) -> dict:
    """Build legal FP8 sidecars; low-E0 state is never read or modified."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite high-sidecar directory: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale incomplete high-sidecar directory: {temporary}")
    temporary.mkdir(parents=True)
    layers = _active_layers(transformer, layer_names)
    expected_layers = len(layers)
    records = {recipe: {} for recipe in FORMAL_RECIPES}
    rows = []
    aggregate = defaultdict(lambda: defaultdict(float))
    candidate_hashers = {recipe: hashlib.sha256() for recipe in MX_RTN_CANDIDATES}
    started = time.perf_counter()
    try:
        for index, (name, layer) in enumerate(layers.items(), 1):
            if name not in hessians:
                raise KeyError(f"missing high Hessian for {name}")
            parity = unquantized_split_parity(layer, layer.module.weight.detach())
            if parity > 1e-3:
                raise RuntimeError(
                    f"{name}: unquantized high/low orientation parity failed ({parity})"
                )
            _, high, _, _ = _rotate_and_split_W(layer, layer.module.weight.detach())
            source = high.to("cuda" if require_cuda else high.device, dtype=torch.float32)
            hessian = hessians[name].to(source.device, dtype=torch.float32)
            if hessian.shape != (source.shape[1], source.shape[1]):
                raise RuntimeError(f"{name}: high Hessian/weight orientation mismatch")

            scales, _, scale_selection = choose_e4_per_channel_scales(
                source, hessian=hessian, multipliers=E4_SCALE_MULTIPLIERS
            )
            e4_rtn, _ = quantize_e4_per_channel_rtn(source, scales=scales)
            e4_gptq = quantize_e4_per_channel_gptq(
                source, hessian, scales, damp_pct=damp_pct, require_cuda=require_cuda
            )
            if e4_gptq.gptq_status != "gptq":
                raise RuntimeError(f"{name}: E4 GPTQ failed closed: {e4_gptq.failure}")
            e4_rtn_loss = hessian_weighted_error(source, e4_rtn.reconstructed, hessian)
            e4_gptq_loss = hessian_weighted_error(source, e4_gptq.reconstructed, hessian)
            tolerance = max(1e-5 * float(e4_rtn_loss), 1e-6)
            if float(e4_gptq_loss) > float(e4_rtn_loss) + tolerance:
                raise RuntimeError(
                    f"{name}: E4 GPTQ objective {float(e4_gptq_loss)} exceeds "
                    f"matched RTN {float(e4_rtn_loss)}"
                )

            mx_results = {}
            mx_meta = {}
            for recipe in MX_RTN_CANDIDATES:
                result, meta = quantize_mx_weight_rtn(source, recipe=recipe)
                mx_results[recipe], mx_meta[recipe] = result, meta
                candidate_hashers[recipe].update(result.payload.cpu().numpy().tobytes())
            mx_best_recipe = min(
                MX_RTN_CANDIDATES,
                key=lambda recipe: float(hessian_weighted_error(
                    source, mx_results[recipe].reconstructed, hessian
                )),
            )
            mx_best = mx_results[mx_best_recipe]
            neighbor_scales, neighbor_meta = choose_mx_scale_bytes(source, "neighbor")
            mx_gptq = quantize_mx_weight_gptq(
                source, hessian, neighbor_scales,
                damp_pct=damp_pct, require_cuda=require_cuda,
            )
            if mx_gptq.gptq_status != "gptq":
                raise RuntimeError(f"{name}: MX GPTQ failed closed: {mx_gptq.failure}")
            mx_neighbor_loss = hessian_weighted_error(
                source, mx_results["neighbor"].reconstructed, hessian
            )
            mx_gptq_loss = hessian_weighted_error(source, mx_gptq.reconstructed, hessian)
            tolerance = max(1e-5 * float(mx_neighbor_loss), 1e-6)
            if float(mx_gptq_loss) > float(mx_neighbor_loss) + tolerance:
                raise RuntimeError(
                    f"{name}: MX GPTQ objective {float(mx_gptq_loss)} exceeds "
                    f"matched neighbor RTN {float(mx_neighbor_loss)}"
                )

            selected = {
                "e4-pc-rtn": (e4_rtn, "e4m3-per-channel", "neighbor-fit-rtn"),
                "e4-pc-gptq": (e4_gptq, "e4m3-per-channel", "neighbor-fit-gptq"),
                "mx-best-rtn": (mx_best, "mxfp8-e4m3-k32", f"{mx_best_recipe}-rtn"),
                "mx-neighbor-gptq": (mx_gptq, "mxfp8-e4m3-k32", "neighbor-gptq"),
            }
            layer_meta = {
                **common_metadata, "layer": name,
                "high_target_sha256": tensor_sha256(high.to(layer.module.weight.dtype)),
                "hessian_sha256": tensor_sha256(hessians[name]),
                "damp_pct": float(damp_pct), "cpu_fallback": False,
                "rtn_fallback": False, "unquantized_parity_max_abs": parity,
            }
            row = {
                "layer": name, "out_features": source.shape[0], "high_k": source.shape[1],
                "unquantized_parity_max_abs": parity,
                "e4_scale_selection": json.dumps(scale_selection, sort_keys=True),
                "mx_best_rtn_recipe": mx_best_recipe,
                "mx_neighbor_selection": json.dumps(neighbor_meta, sort_keys=True),
            }
            for recipe, (result, fmt, method) in selected.items():
                records[recipe][name] = make_high_weight_record(
                    result, fmt=fmt, recipe=method, metadata=layer_meta
                )
                diagnostics = _result_diagnostics(source, result, hessian)
                for key, value in diagnostics.items():
                    row[f"{recipe}_{key}"] = value
                aggregate[recipe]["raw_sse"] += diagnostics["raw_sse"]
                aggregate[recipe]["hessian_weighted_error"] += diagnostics[
                    "hessian_weighted_error"
                ]
                aggregate[recipe]["saturation_count"] += diagnostics["saturation_count"]
                aggregate[recipe]["payload_elements"] += source.numel()
            for recipe in MX_RTN_CANDIDATES:
                diagnostic = _result_diagnostics(source, mx_results[recipe], hessian)
                row[f"mx-{recipe}-rtn_hessian_weighted_error"] = diagnostic[
                    "hessian_weighted_error"
                ]
                aggregate[f"mx-{recipe}-rtn"]["hessian_weighted_error"] += diagnostic[
                    "hessian_weighted_error"
                ]
                aggregate[f"mx-{recipe}-rtn"]["raw_sse"] += diagnostic["raw_sse"]
            rows.append(row)
            del source, hessian, high, e4_rtn, e4_gptq, mx_results, mx_gptq
            if require_cuda and index % 10 == 0:
                torch.cuda.empty_cache()

        if len(rows) != expected_layers:
            raise RuntimeError("high-sidecar build has incomplete layer coverage")
        sidecars = {}
        for recipe in FORMAL_RECIPES:
            path = temporary / f"{recipe}.packing.pt"
            payload = {
                "schema": HIGH_SIDECAR_SCHEMA,
                "recipe": recipe,
                "metadata": {**common_metadata, "damp_pct": float(damp_pct),
                             "layer_count": expected_layers},
                "layers": records[recipe],
            }
            torch.save(payload, path)
            sidecars[recipe] = {
                "path": str(output_dir / path.name), "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "serialized_tensor_bytes": {
                    key: sum(serialized_high_record_bytes(record)[key]
                             for record in records[recipe].values())
                    for key in ("payload", "scales", "total")
                },
            }
        with (temporary / "layer_objectives.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary = {
            "schema": HIGH_SIDECAR_SCHEMA,
            "metadata": common_metadata,
            "layer_count": expected_layers,
            "gptq_coverage": expected_layers,
            "rtn_fallbacks": 0, "cpu_fallbacks": 0,
            "aggregate": {key: dict(value) for key, value in aggregate.items()},
            "mx_rejected_candidate_payload_hashes": {
                key: value.hexdigest() for key, value in candidate_hashers.items()
            },
            "sidecars": sidecars,
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if require_cuda else 0.0
            ),
            "peak_cuda_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024**3 if require_cuda else 0.0
            ),
        }
        _write_json(temporary / "build_summary.json", summary)
        temporary.rename(output_dir)
        return summary
    except Exception:
        # Preserve the visibly incomplete directory for diagnosis; it cannot
        # be mistaken for a complete cache because atomic rename never occurs.
        raise


def validate_high_sidecar(path: Path, expected_metadata: dict, layer_count: int = 120) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != HIGH_SIDECAR_SCHEMA:
        raise RuntimeError("invalid high-sidecar schema")
    if len(payload.get("layers", {})) != layer_count:
        raise RuntimeError("high-sidecar layer coverage mismatch")
    for key, value in expected_metadata.items():
        if payload.get("metadata", {}).get(key) != value:
            raise RuntimeError(f"high-sidecar provenance mismatch for {key}")
    for name, record in payload["layers"].items():
        decoded = decode_high_weight_record(record)
        if tuple(decoded.shape) != tuple(record["stored_shape"]):
            raise RuntimeError(f"{name}: decoded high-sidecar shape mismatch")
    return payload


@torch.inference_mode()
def materialize_high_sidecar_into_state(
    transformer,
    base_state: dict[str, torch.Tensor],
    sidecar: dict,
) -> dict[str, torch.Tensor]:
    """Return an in-memory state with identical low cache and replaced high."""
    layers = _active_layers(transformer)
    if set(sidecar["layers"]) != set(layers):
        raise RuntimeError("high-sidecar names do not match active wrappers")
    state = dict(base_state)
    for name, layer in layers.items():
        key = f"{name}.module.weight"
        if key not in base_state:
            raise KeyError(f"base rank-3r cache is missing {key}")
        low, _ = extract_fused_low_weight(layer, base_state[key])
        high = decode_high_weight_record(
            sidecar["layers"][name], dtype=base_state[key].dtype
        )
        fused = stitch_low_high(layer, low, high)
        if fused.shape != base_state[key].shape:
            raise RuntimeError(f"{name}: materialized fused weight shape mismatch")
        state[key] = fused
        weight_key = f"{name}.weight"
        if weight_key in state:
            state[weight_key] = fused
    return state


def summarize_dev_weight_gate(
    summaries: dict[str, dict],
    *,
    b0_persistent_bytes: int,
    arm_persistent_bytes: dict[str, int],
) -> dict:
    """Apply the frozen W8A16 gate using raw, equal-prompt and time groups."""
    b0 = summaries["B0"]
    b1 = summaries["B1"]
    denominator = b0["raw_sse"] - b1["raw_sse"]
    groups = {"early": range(0, 7), "mid": range(7, 14), "late": range(14, 20)}
    results = {}
    for arm, candidate in summaries.items():
        if arm in {"B0", "B1", "old-E4-W", "old-MX-W"}:
            continue
        recovery = ((b0["raw_sse"] - candidate["raw_sse"]) / denominator
                    if denominator > 0 else None)
        prompt_keys = sorted(b0["per_prompt"])
        per_prompt_fraction = [
            (b0["per_prompt"][key] - candidate["per_prompt"][key])
            / b0["per_prompt"][key]
            for key in prompt_keys
        ]
        wins = sum(value > 0 for value in per_prompt_fraction)
        group_changes = {}
        for group, steps in groups.items():
            base = sum(b0["per_timestep"].get(str(step), b0["per_timestep"].get(step, 0.0))
                       for step in steps)
            value = sum(candidate["per_timestep"].get(
                str(step), candidate["per_timestep"].get(step, 0.0)
            ) for step in steps)
            group_changes[group] = (value - base) / base
        byte_ratio = arm_persistent_bytes[arm] / b0_persistent_bytes
        passed = (
            recovery is not None and recovery >= .5 and wins >= 24
            and max(group_changes.values()) <= .02 and byte_ratio <= 1.01
        )
        results[arm] = {
            "raw_aggregate_gain_fraction": (
                b0["raw_sse"] - candidate["raw_sse"]
            ) / b0["raw_sse"],
            "rank_headroom_recovery": recovery,
            "equal_prompt_mean_gain_fraction": float(np.mean(per_prompt_fraction)),
            "equal_prompt_median_gain_fraction": float(np.median(per_prompt_fraction)),
            "prompt_wins": wins,
            "timestep_group_relative_changes": group_changes,
            "persistent_weight_bytes": arm_persistent_bytes[arm],
            "persistent_weight_ratio_vs_B0": byte_ratio,
            "passed": passed,
        }
    return {"arms": results, "continue_to_A8": any(row["passed"] for row in results.values())}
