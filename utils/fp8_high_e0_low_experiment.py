"""Formal SANA experiment plumbing for FP8-high / hardware-E0-low DiRotQ.

Artifacts produced here are intentionally external to Git.  The module keeps
the experiment fail-closed: exact model revision, frozen splits, rank-matched
rotation, candidate-specific E0 Hessian/GPTQ, and cache metadata must agree
before an arm can run.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch
import yaml

from .e0joint_gptq import sha256_file
from .fp8_high_e0_low import (
    HighFormatStats,
    RankContract,
    generate_matched_residual_rotations,
    quantize_mxfp8_e4m3,
    quantize_plain_e4m3,
    serialized_weight_bytes,
    tensor_sha256,
    validate_residual_rotation,
)
from .hardware_weight_fp4 import (
    decode_packing_record,
    gptq_quantize_hardware_fixed,
    make_packing_record,
)
from .quant_utils import ActQuantWrapper, _rotate_and_split_W, add_actquant, find_qlayers
from .tilemixfp4_utils import FormatSelectionStats
from .weight_mixfp4 import collect_e0_activation_hessians, hessian_trace_loss


MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
MODEL_REVISION = "e2b3c0cbffebcd09d83805e88b9f5f106afc74ac"
DATASET_SHA256 = "07ce5ef172dc0454c0267ad7a68a16e21ae2e695356651a51a9f303a166b120e"
REQUIRED_ACTIVE_LAYERS = 120
FIT_STEP_INDICES = (0, 5, 10, 15, 19)
ARMS = {
    "B0": {"rank": "r", "weight": "bf16", "activation": "bf16"},
    "B1": {"rank": "3r", "weight": "bf16", "activation": "bf16"},
    "E4-W": {"rank": "3r", "weight": "e4m3", "activation": "bf16"},
    "E4-AW": {"rank": "3r", "weight": "e4m3", "activation": "e4m3"},
    "MX-W": {"rank": "3r", "weight": "mxfp8", "activation": "bf16"},
    "MX-AW": {"rank": "3r", "weight": "mxfp8", "activation": "mxfp8"},
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalized_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def prompt_hashes(prompt: str) -> tuple[str, str]:
    exact = hashlib.sha256(prompt.encode()).hexdigest()
    normalized = hashlib.sha256(normalized_prompt(prompt).encode()).hexdigest()
    return exact, normalized


def hash_str_to_int(value: str) -> int:
    modulus = 10**9 + 7
    output = 0
    for character in value:
        output = (output * 31 + ord(character)) % modulus
    return output


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def visible_historical_usage(repo: Path, dataset: dict) -> dict:
    """Conservative audit over visible IDs and parseable small manifests."""
    dataset_ids = set(dataset)
    exact_lookup = {}
    normalized_lookup = {}
    for image_id, info in dataset.items():
        exact, normalized = prompt_hashes(info["prompt"])
        exact_lookup[exact] = image_id
        normalized_lookup[normalized] = image_id
    used_ids: set[str] = set()
    sources: set[str] = set()
    models = repo / "models"
    for path in models.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        for candidate in re.findall(r"[0-9a-f]{40}", path.name):
            if candidate in dataset_ids:
                used_ids.add(candidate)
                sources.add(str(path.relative_to(repo)))
        if path.suffix.lower() not in {".json", ".csv", ".yaml", ".yml"}:
            continue
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        try:
            if path.suffix.lower() == ".json":
                strings = _walk_strings(json.loads(path.read_text()))
            elif path.suffix.lower() in {".yaml", ".yml"}:
                strings = _walk_strings(yaml.safe_load(path.read_text()))
            else:
                with path.open(newline="") as handle:
                    strings = (cell for row in csv.reader(handle) for cell in row)
            hit = False
            for string in strings:
                if string in dataset_ids:
                    used_ids.add(string)
                    hit = True
                exact, normalized = prompt_hashes(string)
                image_id = exact_lookup.get(exact) or normalized_lookup.get(normalized)
                if image_id is not None:
                    used_ids.add(image_id)
                    hit = True
            if hit:
                sources.add(str(path.relative_to(repo)))
        except Exception:
            # An opaque/invalid historical file is retained and explicitly
            # prevents any claim stronger than "no overlap with visible manifests".
            continue
    return {
        "used_ids": sorted(used_ids),
        "visible_sources": sorted(sources),
        "claim_scope": "no overlap with visible parseable manifests and image filenames",
    }


def freeze_splits(repo: Path, output: Path) -> dict:
    dataset_path = repo / "datasets/mjhq_5000_samples.json"
    if sha256_file(dataset_path) != DATASET_SHA256:
        raise RuntimeError("MJHQ dataset SHA-256 mismatch")
    dataset = json.loads(dataset_path.read_text())
    historical = visible_historical_usage(repo, dataset)
    used = set(historical["used_ids"])
    requested = (("fit", 64), ("dev", 32), ("final", 64), ("pilot", 64))
    splits = {}
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    iterator = iter(dataset.items())
    for split, count in requested:
        rows = []
        while len(rows) < count:
            try:
                image_id, info = next(iterator)
            except StopIteration as error:
                raise RuntimeError("not enough leakage-free MJHQ prompts") from error
            exact, normalized = prompt_hashes(info["prompt"])
            if image_id in used or exact in seen_exact or normalized in seen_normalized:
                continue
            rows.append({
                "image_id": image_id, "prompt": info["prompt"],
                "category": info["category"], "seed": hash_str_to_int(image_id),
                "exact_prompt_sha256": exact,
                "normalized_prompt_sha256": normalized,
            })
            seen_exact.add(exact)
            seen_normalized.add(normalized)
        splits[split] = rows
    manifest = {
        "schema": "dirotq.fp8_high_e0_low.splits.v1",
        "dataset": str(dataset_path), "dataset_sha256": DATASET_SHA256,
        "fit_step_indices": list(FIT_STEP_INDICES),
        "historical_audit": historical, "splits": splits,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json(output, manifest)
    return manifest


def _tree_map(fn, value):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: _tree_map(fn, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        mapped = [_tree_map(fn, item) for item in value]
        return type(value)(mapped)
    return value


def _tree_split(value):
    if isinstance(value, dict):
        keys = list(value)
        values = [_tree_split(value[key]) for key in keys]
        length = max(map(len, values))
        return [
            {key: values[j][min(index, len(values[j]) - 1)] for j, key in enumerate(keys)}
            for index in range(length)
        ]
    if isinstance(value, (list, tuple)):
        values = [_tree_split(item) for item in value]
        length = max(map(len, values))
        return [type(value)(item[min(index, len(item) - 1)] for item in values)
                for index in range(length)]
    if isinstance(value, torch.Tensor):
        return [value[index:index + 1] for index in range(value.shape[0])]
    return [value]


class TeacherCapture:
    def __init__(self):
        self.calls = []

    def __call__(self, module, args, kwargs, output):
        signature = inspect.signature(module.forward)
        bound = signature.bind(*args, **kwargs)
        arguments = dict(bound.arguments)
        hidden = arguments.pop("hidden_states")
        cache = _tree_map(
            lambda tensor: tensor.detach().cpu(),
            {"input_args": [hidden], "input_kwargs": arguments, "outputs": output},
        )
        self.calls.extend(_tree_split(cache))


@torch.inference_mode()
def collect_teacher_cache(
    pipeline, rows: list[dict], output_dir: Path, *, selected_steps: tuple[int, ...],
    num_steps: int = 20, guidance_scale: float = 4.5,
) -> dict:
    """Collect stock-BF16 teacher calls without VAE decoding or images."""
    output_dir.mkdir(parents=True, exist_ok=False)
    capture = TeacherCapture()
    hook = pipeline.transformer.register_forward_hook(capture, with_kwargs=True)
    manifest_rows = []
    started = time.perf_counter()
    try:
        pipeline.set_progress_bar_config(disable=True)
        for row in rows:
            generator = torch.Generator(device=pipeline.device).manual_seed(row["seed"])
            pipeline(
                row["prompt"], generator=generator,
                num_inference_steps=num_steps, guidance_scale=guidance_scale,
                height=1024, width=1024, output_type="latent",
            )
            expected = num_steps * 2
            if len(capture.calls) != expected:
                raise RuntimeError(
                    f"{row['image_id']}: expected {expected} CFG calls, got {len(capture.calls)}"
                )
            for step in selected_steps:
                for guidance in range(2):
                    index = step * 2 + guidance
                    cache = capture.calls[index]
                    cache.update({
                        "filename": row["image_id"], "step": step,
                        "guidance": guidance, "prompt_sha256": row["exact_prompt_sha256"],
                    })
                    name = f"{row['image_id']}-{step:05d}-{guidance}.pt"
                    path = output_dir / name
                    torch.save(cache, path)
                    manifest_rows.append({
                        "file": name, "sha256": sha256_file(path),
                        "size": path.stat().st_size, "image_id": row["image_id"],
                        "step": step, "guidance": guidance,
                    })
            capture.calls.clear()
    finally:
        hook.remove()
    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dtype": "bfloat16", "num_steps": num_steps,
        "guidance_scale": guidance_scale, "height": 1024, "width": 1024,
        "selected_steps": list(selected_steps), "prompt_count": len(rows),
        "cache_count": len(manifest_rows), "elapsed_seconds": time.perf_counter() - started,
        "files": manifest_rows,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_model_utils(repo: Path):
    path = repo / "models/sana-1.6b/model_utils.py"
    spec = importlib.util.spec_from_file_location("sana_fp8_experiment_model_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def setup_wrapped_transformer(
    transformer, *, repo: Path, basis: dict, rotation: dict,
    high_activation_format: str = "bf16", collect_high_stats: bool = False,
    collect_low_stats: bool = False,
) -> dict:
    cfg = yaml.safe_load((repo / "models/sana-1.6b/config.yaml").read_text())
    model_utils = _load_model_utils(repo)
    preprocess = getattr(model_utils, "preprocess_transformer", None)
    if preprocess is not None:
        preprocess(transformer, cfg)
    skip = cfg["nvfp4"]["skip_layers"]
    add_actquant(transformer, skip_names=skip)
    model_utils.assign_online_rotations(
        transformer, basis, rotation, cfg, residual_rotation="random"
    )
    model_utils.configure_quantizers_by_name(
        transformer,
        rotation["high_len_hidden"], rotation["high_len_head"], cfg,
        nvfp4=True, activation_format="e0m3",
        high_quant_format=high_activation_format,
    )
    active = {
        name: layer for name, layer in find_qlayers(
            transformer, layers=[ActQuantWrapper]
        ).items() if layer.quantizer.bits < 16
    }
    if len(active) != REQUIRED_ACTIVE_LAYERS:
        raise RuntimeError(f"expected 120 active SANA layers, found {len(active)}")
    transformer.eval().requires_grad_(False)
    high_stats = {}
    low_stats = {}
    if collect_high_stats:
        for name, layer in active.items():
            stats = HighFormatStats()
            layer.quantizer.high_format_stats = stats
            high_stats[name] = stats
    if collect_low_stats:
        for name, layer in active.items():
            stats = FormatSelectionStats(selection_unit="fixed-e0")
            layer.quantizer.format_stats = stats
            low_stats[name] = stats
    return {
        "active_layers": active, "config": cfg, "skip_layers": skip,
        "high_stats": high_stats,
        "low_stats": low_stats,
    }


def stitch_low_high(layer: ActQuantWrapper, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    if layer.rotation is not None:
        return torch.cat((low, high.to(low.device)), dim=1)
    if layer.rotation_per_head is not None:
        heads, head_dim = int(layer.num_heads), int(layer.head_dim)
        high_per_head = int(layer.quantizer.high_bits_length)
        low_per_head = head_dim - high_per_head
        out = low.shape[0]
        return torch.cat((
            low.reshape(out, heads, low_per_head),
            high.to(low.device).reshape(out, heads, high_per_head),
        ), dim=2).reshape(out, heads * head_dim)
    raise RuntimeError("FP8-high experiment requires a PCA/random-R transformed layer")


def unquantized_split_parity(layer: ActQuantWrapper, source_weight: torch.Tensor) -> float:
    low, high, _, _ = _rotate_and_split_W(layer, source_weight)
    fused = stitch_low_high(layer, low, high)
    generator = torch.Generator(device=source_weight.device).manual_seed(99)
    x = torch.randn(2, source_weight.shape[1], generator=generator,
                    device=source_weight.device, dtype=torch.float32)
    if layer.rotation is not None:
        rotated = x @ layer.rotation.to(x.device).float()
    else:
        heads, head_dim = int(layer.num_heads), int(layer.head_dim)
        shaped = x.reshape(2, heads, head_dim)
        rotated = torch.einsum(
            "bhd,hde->bhe", shaped, layer.rotation_per_head.to(x.device).float()
        ).reshape_as(x)
    direct = x @ source_weight.float().T
    split = rotated @ fused.float().T
    return float((direct - split).abs().max())


def _base_state(transformer) -> dict:
    transient = (".quantizer.scale", ".quantizer.zero")
    return {
        key: value.detach().cpu()
        for key, value in transformer.state_dict().items()
        if not key.endswith(transient)
    }


def _unique_tensor_bytes(values) -> int:
    seen = set()
    total = 0
    for value in values:
        storage = value.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


@torch.inference_mode()
def build_rank_caches(
    transformer, *, fit_dir: Path, output_dir: Path, rank_label: str,
    formats: tuple[str, ...], provenance: dict, batch_size: int = 4,
    damp_pct: float = .01,
) -> dict:
    """Build one matched low-E0 GPTQ state and requested high variants."""
    output_dir.mkdir(parents=True, exist_ok=False)
    active = {
        name: layer for name, layer in find_qlayers(
            transformer, layers=[ActQuantWrapper]
        ).items() if layer.quantizer.bits < 16
    }
    fit_files = sorted(fit_dir.glob("*.pt"))
    transformer.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    hessians, hmeta = collect_e0_activation_hessians(
        transformer, fit_dir, len(fit_files), batch_size, device="cuda"
    )
    hessian_path = output_dir / "e0_hessians.pt"
    torch.save(hessians, hessian_path)
    base_state = _base_state(transformer)
    weights = {fmt: {} for fmt in formats}
    high_records = {fmt: {} for fmt in formats if fmt != "bf16"}
    low_records = {}
    layer_rows = []
    failures = []
    aggregate_loss = 0.0
    parity_max = 0.0
    serialized_active = {fmt: defaultdict(int) for fmt in formats}
    started = time.perf_counter()
    for index, (name, layer) in enumerate(active.items(), 1):
        original_weight = layer.module.weight.detach()
        parity = unquantized_split_parity(layer, original_weight)
        parity_max = max(parity_max, parity)
        low, high, _, _ = _rotate_and_split_W(layer, original_weight)
        source = low.to("cuda", dtype=torch.float32)
        hessian = hessians[name].to("cuda", dtype=torch.float32)
        quantized, stats, frozen = gptq_quantize_hardware_fixed(
            source, hessian, "hardware-fixed-e0", damp_pct=damp_pct
        )
        if quantized is None:
            failures.append({"layer": name, "reason": stats["failure"]})
            continue
        loss = float(hessian_trace_loss(source, quantized, hessian))
        aggregate_loss += loss
        original_high = high.to("cuda", dtype=torch.float32)
        low_record = make_packing_record(
            quantized, "hardware-fixed-e0", frozen,
            high_branch_hash=tensor_sha256(high.to(layer.module.weight.dtype)),
        )
        decoded_low = decode_packing_record(low_record, dtype=torch.float32, device="cuda")
        low_records[name] = low_record
        high_diagnostics = {}
        for fmt in formats:
            if fmt == "bf16":
                reconstructed_high = original_high.to(layer.module.weight.dtype)
            elif fmt == "e4m3":
                result = quantize_plain_e4m3(original_high)
                reconstructed_high = result.reconstructed.to(layer.module.weight.dtype)
                high_records[fmt][name] = {
                    "format": fmt, "payload": result.payload.cpu(),
                    "global_scale": result.scale.cpu(),
                    "stored_shape": list(original_high.shape),
                    "saturation_count": result.saturation_count,
                }
            elif fmt == "mxfp8":
                result = quantize_mxfp8_e4m3(original_high)
                reconstructed_high = result.reconstructed.to(layer.module.weight.dtype)
                high_records[fmt][name] = {
                    "format": fmt, "payload": result.payload.cpu(),
                    "scale_bytes": result.scale_bytes.cpu(),
                    "stored_shape": list(original_high.shape),
                    "padded_k": result.padded_k,
                    "saturation_count": result.saturation_count,
                }
            else:
                raise ValueError(f"unsupported high format {fmt!r}")
            fused = stitch_low_high(
                layer, decoded_low.to(layer.module.weight.dtype), reconstructed_high
            ).cpu()
            weights[fmt][name] = fused
            high_error = (original_high - reconstructed_high.float()).double().square().sum()
            high_energy = original_high.double().square().sum().clamp_min(1e-30)
            high_diagnostics[f"high_{fmt}_weight_sse"] = float(high_error)
            high_diagnostics[f"high_{fmt}_weight_relative_sse"] = float(
                high_error / high_energy
            )
        layer_rows.append({
            "layer": name, "low_n": source.shape[0], "low_k": source.shape[1],
            "high_k": original_high.shape[1], "gptq_status": stats["gptq_status"],
            "attempts": stats["attempts"], "loss": loss,
            "low_weight_sse": float((source - quantized).double().square().sum()),
            "low_weight_relative_sse": float(
                (source - quantized).double().square().sum()
                / source.double().square().sum().clamp_min(1e-30)
            ),
            "global_scale": stats["global_scale"], "saturation_rate": stats["saturation_rate"],
            "unquantized_parity_max_abs": parity,
            **high_diagnostics,
        })
        for fmt in formats:
            byte_row = serialized_weight_bytes(
                out_features=source.shape[0], low_rank=source.shape[1],
                high_rank=original_high.shape[1], high_format=fmt,
            )
            for key, value in byte_row.items():
                serialized_active[fmt][key] += value
        del source, hessian, quantized, decoded_low, original_high
        if index % 10 == 0:
            torch.cuda.empty_cache()
    if failures or len(layer_rows) != REQUIRED_ACTIVE_LAYERS:
        _write_json(output_dir / "failures.json", failures)
        raise RuntimeError("rank cache build did not achieve 120/120 GPTQ coverage")
    low_sidecar = output_dir / "low_e0_packing.pt"
    torch.save({"format": "hardware-fixed-e0", "layers": low_records}, low_sidecar)
    cache_records = {}
    for fmt in formats:
        state = dict(base_state)
        for name, fused in weights[fmt].items():
            state[f"{name}.weight"] = fused
            state[f"{name}.module.weight"] = fused
        cache_path = output_dir / f"rank-{rank_label}_high-{fmt}_low-e0.pt"
        torch.save(state, cache_path)
        high_sidecar = None
        if fmt != "bf16":
            high_sidecar = output_dir / f"rank-{rank_label}_high-{fmt}.packing.pt"
            torch.save({"format": fmt, "layers": high_records[fmt]}, high_sidecar)
        cache_records[fmt] = {
            "path": str(cache_path), "sha256": sha256_file(cache_path),
            "size": cache_path.stat().st_size,
            "high_sidecar": str(high_sidecar) if high_sidecar else None,
            "high_sidecar_sha256": sha256_file(high_sidecar) if high_sidecar else None,
        }
        del state
    with (output_dir / "layer_objectives.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader(); writer.writerows(layer_rows)
    summary = {
        **provenance, "rank_label": rank_label, "fit_cache_count": len(fit_files),
        "hessian": {"path": str(hessian_path), "sha256": sha256_file(hessian_path)},
        "hessian_collection": hmeta, "gptq_layers": len(layer_rows),
        "rtn_fallbacks": 0, "cpu_fallbacks": 0, "aggregate_hessian_loss": aggregate_loss,
        "unquantized_parity_max_abs": parity_max,
        "low_sidecar": str(low_sidecar), "low_sidecar_sha256": sha256_file(low_sidecar),
        "cache_records": cache_records, "build_seconds": time.perf_counter() - started,
        "serialized_active_weight_bytes": {
            fmt: dict(values) for fmt, values in serialized_active.items()
        },
        "full_bf16_transformer_state_bytes": _unique_tensor_bytes(base_state.values()),
        "active_original_weight_bytes": sum(
            layer.module.weight.numel() * layer.module.weight.element_size()
            for layer in active.values()
        ),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }
    _write_json(output_dir / "build_summary.json", summary)
    return summary


def _move_cache_batch(batch: list[dict], device: str, dtype: torch.dtype):
    latents = torch.cat([item["input_args"][0] for item in batch], dim=0).to(device, dtype=dtype)
    keys = batch[0]["input_kwargs"]
    kwargs = {}
    for key in keys:
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
    teachers = torch.cat([item["outputs"][0] for item in batch], dim=0).to(device)
    return latents, kwargs, teachers


@torch.inference_mode()
def evaluate_teacher_cache(
    transformer, cache_dir: Path, output: Path, *, arm: str, batch_size: int = 4,
) -> dict:
    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise RuntimeError(f"no teacher cache files in {cache_dir}")
    dtype = next(transformer.parameters()).dtype
    transformer.to("cuda").eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    started = time.perf_counter()
    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        batch = [torch.load(path, map_location="cpu", weights_only=False) for path in batch_files]
        latents, kwargs, teachers = _move_cache_batch(batch, "cuda", dtype)
        outputs = transformer(latents, **kwargs)[0]
        for index, (path, item) in enumerate(zip(batch_files, batch)):
            error = outputs[index].float() - teachers[index].float()
            teacher = teachers[index].float()
            raw = float(error.double().square().sum())
            norm = float(teacher.double().square().sum())
            cosine = float(torch.nn.functional.cosine_similarity(
                outputs[index].float().reshape(1, -1), teacher.reshape(1, -1)
            ))
            rows.append({
                "arm": arm, "file": path.name, "image_id": item["filename"],
                "step": int(item["step"]), "guidance": int(item["guidance"]),
                "raw_sse": raw, "teacher_square_sum": norm,
                "relative_mse": raw / norm if norm else 0.0, "cosine": cosine,
            })
        del latents, kwargs, teachers, outputs, batch
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    per_prompt = defaultdict(float)
    per_step = defaultdict(float)
    for row in rows:
        per_prompt[row["image_id"]] += row["raw_sse"]
        per_step[row["step"]] += row["raw_sse"]
    summary = {
        "arm": arm, "cache_count": len(rows),
        "raw_sse": sum(row["raw_sse"] for row in rows),
        "teacher_square_sum": sum(row["teacher_square_sum"] for row in rows),
        "relative_mse": sum(row["raw_sse"] for row in rows) /
                        sum(row["teacher_square_sum"] for row in rows),
        "mean_cosine": float(np.mean([row["cosine"] for row in rows])),
        "per_prompt": dict(per_prompt), "per_timestep": dict(per_step),
        "seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }
    _write_json(output.with_suffix(".summary.json"), summary)
    return summary


def dev_gate(summaries: dict[str, dict]) -> dict:
    baseline = summaries["B0"]
    results = {}
    groups = {"early": range(0, 7), "mid": range(7, 14), "late": range(14, 20)}
    for arm in ("E4-AW", "MX-AW"):
        candidate = summaries[arm]
        aggregate_gain = (baseline["raw_sse"] - candidate["raw_sse"]) / baseline["raw_sse"]
        prompt_wins = sum(
            candidate["per_prompt"][key] < value
            for key, value in baseline["per_prompt"].items()
        )
        group_changes = {}
        for name, steps in groups.items():
            base = sum(baseline["per_timestep"].get(str(step), baseline["per_timestep"].get(step, 0))
                       for step in steps)
            cand = sum(candidate["per_timestep"].get(str(step), candidate["per_timestep"].get(step, 0))
                       for step in steps)
            group_changes[name] = (cand - base) / base
        passed = aggregate_gain >= -.01 and prompt_wins >= 16 and max(group_changes.values()) <= .02
        results[arm] = {
            "aggregate_gain": aggregate_gain, "prompt_wins": prompt_wins,
            "timestep_group_relative_changes": group_changes, "passed": passed,
        }
    return {"arms": results, "continue": any(item["passed"] for item in results.values())}


def load_pipeline(*, revision: str = MODEL_REVISION):
    from diffusers import SanaPipeline
    return SanaPipeline.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=torch.bfloat16,
        use_safetensors=True, local_files_only=True,
    )
