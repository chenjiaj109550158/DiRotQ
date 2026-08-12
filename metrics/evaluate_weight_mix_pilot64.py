#!/usr/bin/env python3
"""Matched SANA WeightMix Pilot64 evaluation and immutable tile-count audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPScore

from evaluate_pilot128 import (
    _evaluate_config,
    _load_samples,
    _validate_configs,
    _write_csv,
)


REFERENCE = "bf16-reference"
STANDARD = "existing-standard-e2"
NEW_CONFIGS = (
    "reoptimized-fixed-e2", "reoptimized-fixed-e0", "weight-tilemix",
)
ALL_CONFIGS = (REFERENCE, STANDARD, *NEW_CONFIGS)
METRICS = ("psnr", "lpips", "ssim", "clip_score")
COMPARISONS = (
    ("weight-tilemix", "reoptimized-fixed-e0"),
    ("reoptimized-fixed-e0", "reoptimized-fixed-e2"),
    ("weight-tilemix", "reoptimized-fixed-e2"),
    ("weight-tilemix", STANDARD),
    ("reoptimized-fixed-e2", STANDARD),
)
SEGMENTS = (("first16", 0, 16), ("new48", 16, 64), ("full64", 0, 64))


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(float(x) for x in np.quantile(values[indices].mean(1), (.025, .975)))


def _load_existing(path: Path, ids: list[str]) -> dict[str, dict[str, dict]]:
    mapping = {"fp16": REFERENCE, "e0m3": STANDARD}
    expected = set(ids)
    output = {config: {} for config in (REFERENCE, STANDARD)}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            config = mapping.get(row["config"])
            if config is not None and row["image_id"] in expected:
                if row["image_id"] in output[config]:
                    raise RuntimeError(f"duplicate metric row: {config}/{row['image_id']}")
                copied = dict(row)
                copied["config"] = config
                output[config][row["image_id"]] = copied
    for config, rows in output.items():
        if set(rows) != expected:
            raise RuntimeError(f"{config}: existing Pilot64 metric IDs mismatch")
    return output


def _load_w16(path: Path, ids: list[str]) -> dict[str, dict]:
    expected = set(ids)
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["config"] == "e0a-w16-residual" and row["image_id"] in expected:
                output[row["image_id"]] = row
    if set(output) != expected:
        raise RuntimeError("W16 metrics do not match the first32 IDs")
    return output


def _values(rows, config: str, ids: list[str], metric: str) -> np.ndarray:
    return np.asarray([float(rows[config][image_id][metric]) for image_id in ids])


def _family(layer: str) -> str:
    suffix = layer.split(".", 2)[-1]
    if ".attn1.to_out" in layer:
        return "self-attention-output"
    if ".attn1." in layer:
        return "self-attention-qkv"
    if ".attn2.to_out" in layer:
        return "cross-attention-output"
    if ".attn2." in layer:
        return "cross-attention-query"
    if ".ff." in layer:
        return "mlp"
    return suffix


def _tile_audit(layer_csv: Path, output_dir: Path) -> dict:
    rows = [row for row in csv.DictReader(layer_csv.open()) if row["mode"] == "tilemix"]
    if len(rows) != 120:
        raise RuntimeError(f"expected 120 TileMix layer rows, found {len(rows)}")
    enriched = []
    family_counts = {}
    e2_total = e0_total = valid_weights = padded_slots = 0
    lower_e2_valid = upper_e2_valid = 0
    for row in rows:
        e2, e0 = int(row["e2_count"]), int(row["e0_count"])
        n, k = int(row["n"]), int(row["k"])
        total = e2 + e0
        expected_tiles = math.ceil(n / 8) * math.ceil(k / 64)
        if total != expected_tiles:
            raise RuntimeError(f"{row['layer']}: tile count/mapping mismatch")
        family = _family(row["layer"])
        family_counts.setdefault(family, {"e2": 0, "e0": 0})
        family_counts[family]["e2"] += e2
        family_counts[family]["e0"] += e0
        e2_total += e2
        e0_total += e0
        valid_weights += n * k
        padded_slots += total * 8 * 64

        # The build report has counts but no map. Bound valid E2 elements by
        # placing E2 choices first/last in the partial-K tiles.
        full_tiles = (n // 8) * (k // 64)
        partial_tiles = (n // 8) if k % 64 else 0
        partial_size = 8 * (k % 64)
        lower_e2_valid += min(e2, partial_tiles) * partial_size
        lower_e2_valid += max(0, e2 - partial_tiles) * 8 * 64
        upper_e2_valid += min(e2, full_tiles) * 8 * 64
        upper_e2_valid += max(0, e2 - full_tiles) * partial_size
        enriched.append({
            "layer": row["layer"], "family": family, "n": n, "k": k,
            "e2_tiles": e2, "e0_tiles": e0, "total_tiles": total,
            "e2_ratio": e2 / total,
        })
    enriched.sort(key=lambda row: (-row["e2_ratio"], row["layer"]))
    _write_csv(output_dir / "weight_tilemix_layer_format_counts.csv", enriched)
    family_output = {}
    for family, counts in sorted(family_counts.items()):
        total = counts["e2"] + counts["e0"]
        family_output[family] = {**counts, "total": total, "e2_ratio": counts["e2"] / total}
    attention_e2 = sum(v["e2"] for key, v in family_counts.items() if key != "mlp")
    mlp_e2 = family_counts.get("mlp", {}).get("e2", 0)
    output = {
        "source": str(layer_csv), "layers": len(rows),
        "e2_tiles": e2_total, "e0_tiles": e0_total,
        "total_tiles": e2_total + e0_total,
        "e2_tile_ratio": e2_total / (e2_total + e0_total),
        "nominal_e2_payload_element_ratio_including_padding": e2_total / (e2_total + e0_total),
        "valid_weight_elements": valid_weights,
        "padded_tile_slots": padded_slots,
        "valid_e2_weight_element_ratio_bounds_without_choice_map": [
            lower_e2_valid / valid_weights, upper_e2_valid / valid_weights,
        ],
        "exact_valid_element_ratio_note": (
            "Unavailable: the immutable cache/build report retains per-layer counts but no "
            "per-tile map. Reconstructing it would require prohibited GPTQ rerun."
        ),
        "e2_family_share": {
            "attention": attention_e2 / e2_total if e2_total else 0.0,
            "mlp": mlp_e2 / e2_total if e2_total else 0.0,
        },
        "families": family_output,
        "top20_layers_by_e2_ratio": enriched[:20],
    }
    (output_dir / "weight_tilemix_format_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--existing-per-prompt", type=Path, required=True)
    parser.add_argument("--headroom32-per-prompt", type=Path, required=True)
    parser.add_argument("--reoptimized-fixed-e2-dir", type=Path, required=True)
    parser.add_argument("--reoptimized-fixed-e0-dir", type=Path, required=True)
    parser.add_argument("--weight-tilemix-dir", type=Path, required=True)
    parser.add_argument("--layer-objectives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    ids = [image_id for image_id, _ in samples]
    new_dirs = {
        "reoptimized-fixed-e2": args.reoptimized_fixed_e2_dir,
        "reoptimized-fixed-e0": args.reoptimized_fixed_e0_dir,
        "weight-tilemix": args.weight_tilemix_dir,
    }
    _validate_configs(list(new_dirs.items()), samples)
    by_config = _load_existing(args.existing_per_prompt, ids)

    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction="none").to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()
    new_rows = []
    for config, directory in new_dirs.items():
        print(f"Evaluating {config}: {directory}", flush=True)
        rows = _evaluate_config(
            config, directory, args.reference_dir, samples, args.batch_size,
            device, lpips, ssim, clip,
        )
        by_config[config] = {row["image_id"]: row for row in rows}
        new_rows.extend(rows)

    config_summary = []
    for segment, start, end in SEGMENTS:
        segment_ids = ids[start:end]
        for config in ALL_CONFIGS:
            for metric in METRICS:
                values = _values(by_config, config, segment_ids, metric)
                config_summary.append({
                    "row_type": "config", "segment": segment,
                    "config": config, "reference": "", "metric": metric,
                    "n": len(values), "mean": float(values.mean()),
                    "median": float(np.median(values)), "paired_mean_delta": "",
                    "paired_median_delta": "", "win_rate": "", "ci95_low": "",
                    "ci95_high": "", "delta_p25": "", "delta_p50": "",
                    "delta_p75": "", "trim_each_count": "", "trimmed_mean_delta": "",
                })

    comparisons = []
    for comparison_index, (candidate, baseline) in enumerate(COMPARISONS):
        for segment_index, (segment, start, end) in enumerate(SEGMENTS):
            segment_ids = ids[start:end]
            for metric_index, metric in enumerate(METRICS):
                candidate_values = _values(by_config, candidate, segment_ids, metric)
                baseline_values = _values(by_config, baseline, segment_ids, metric)
                delta = candidate_values - baseline_values
                low, high = _bootstrap(
                    delta, args.bootstrap_samples,
                    20260812 + comparison_index * 100 + segment_index * 10 + metric_index,
                )
                wins = delta < 0 if metric == "lpips" else delta > 0
                trim_count = max(1, round(.05 * len(delta)))
                ordered = np.sort(delta)
                trimmed = ordered[trim_count:-trim_count]
                q25, q50, q75 = np.quantile(delta, (.25, .5, .75))
                comparisons.append({
                    "row_type": "paired", "segment": segment,
                    "config": candidate, "reference": baseline, "metric": metric,
                    "n": len(delta), "mean": "", "median": "",
                    "paired_mean_delta": float(delta.mean()),
                    "paired_median_delta": float(np.median(delta)),
                    "win_rate": float(wins.mean()), "ci95_low": low, "ci95_high": high,
                    "delta_p25": float(q25), "delta_p50": float(q50),
                    "delta_p75": float(q75), "trim_each_count": trim_count,
                    "trimmed_mean_delta": float(trimmed.mean()),
                })

    w16_ids = ids[:32]
    w16 = _load_w16(args.headroom32_per_prompt, w16_ids)
    recovery = {"n": 32, "scope": "matched first32 only", "configs": {}}
    for config in NEW_CONFIGS:
        recovery["configs"][config] = {}
        for metric in ("psnr", "lpips"):
            standard = _values(by_config, STANDARD, w16_ids, metric).mean()
            candidate = _values(by_config, config, w16_ids, metric).mean()
            ceiling = np.asarray([float(w16[i][metric]) for i in w16_ids]).mean()
            if metric == "psnr":
                numerator, denominator = candidate - standard, ceiling - standard
            else:
                numerator, denominator = standard - candidate, standard - ceiling
            recovery["configs"][config][metric] = (
                float(numerator / denominator) if denominator > 0 else None
            )
            recovery["configs"][config][f"{metric}_numerator"] = float(numerator)
            recovery["configs"][config][f"{metric}_denominator"] = float(denominator)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_prompt = []
    for config in ALL_CONFIGS:
        per_prompt.extend(by_config[config][image_id] for image_id in ids)
    _write_csv(args.output_dir / "weight_mix_pilot64_per_prompt.csv", per_prompt)
    _write_csv(
        args.output_dir / "weight_mix_pilot64_summary.csv",
        config_summary + comparisons,
    )
    (args.output_dir / "weight_mix_first32_recovery.json").write_text(
        json.dumps(recovery, indent=2) + "\n"
    )
    audit = _tile_audit(args.layer_objectives, args.output_dir)
    print(json.dumps({"headroom_recovery": recovery, "tile_audit": audit}, indent=2))


if __name__ == "__main__":
    main()
