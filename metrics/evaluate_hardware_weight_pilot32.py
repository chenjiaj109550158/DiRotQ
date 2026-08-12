#!/usr/bin/env python3
"""Matched SANA Pilot32 evaluation for hardware-faithful fixed E2/E0 weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPScore

from evaluate_pilot128 import (
    _bootstrap_ci,
    _evaluate_config,
    _load_samples,
    _validate_configs,
    _write_csv,
)


REFERENCE = "bf16-reference"
STANDARD = "existing-standard-e2"
LEGACY_E2 = "legacy-reoptimized-e2"
LEGACY_E0 = "legacy-reoptimized-e0"
W16 = "w16-ceiling"
HW_E2 = "hardware-fixed-e2"
HW_E0 = "hardware-fixed-e0"
METRICS = ("psnr", "lpips", "ssim", "clip_score")
COMPARISONS = (
    (HW_E0, HW_E2),
    (HW_E2, LEGACY_E2),
    (HW_E0, LEGACY_E0),
    (HW_E0, STANDARD),
    (HW_E0, W16),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_reused(weight_mix_csv: Path, headroom_csv: Path, ids: list[str]):
    expected = set(ids)
    mapping = {
        "bf16-reference": REFERENCE,
        "existing-standard-e2": STANDARD,
        "reoptimized-fixed-e2": LEGACY_E2,
        "reoptimized-fixed-e0": LEGACY_E0,
    }
    output = {name: {} for name in (*mapping.values(), W16)}
    with weight_mix_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = mapping.get(row["config"])
            if name and row["image_id"] in expected:
                copied = dict(row)
                copied["config"] = name
                output[name][row["image_id"]] = copied
    with headroom_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["config"] == "e0a-w16-residual" and row["image_id"] in expected:
                copied = dict(row)
                copied["config"] = W16
                output[W16][row["image_id"]] = copied
    for name, rows in output.items():
        if set(rows) != expected:
            raise RuntimeError(f"{name}: reused metric IDs do not match first32")
    return output


def _array(by_config, config, ids, metric):
    return np.asarray([float(by_config[config][image_id][metric]) for image_id in ids])


def _integrity_manifest(samples, directories: dict[str, Path]) -> list[dict]:
    rows = []
    _validate_configs(list(directories.items()), samples)
    for config, root in directories.items():
        for index, (image_id, info) in enumerate(samples):
            path = root / info["category"] / f"{image_id}.png"
            with Image.open(path) as image:
                image.load()
                extrema = image.getextrema()
            rows.append({
                "prompt_index": index, "image_id": image_id,
                "category": info["category"], "config": config,
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256(path), "mode": "RGB", "width": 1024,
                "height": 1024, "channel_extrema": repr(extrema), "flat": False,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--weight-mix-per-prompt", type=Path, required=True)
    parser.add_argument("--headroom-per-prompt", type=Path, required=True)
    parser.add_argument("--hardware-e2-dir", type=Path, required=True)
    parser.add_argument("--hardware-e0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    ids = [image_id for image_id, _ in samples]
    new_dirs = {HW_E2: args.hardware_e2_dir, HW_E0: args.hardware_e0_dir}
    integrity = _integrity_manifest(samples, new_dirs)
    by_config = _read_reused(
        args.weight_mix_per_prompt, args.headroom_per_prompt, ids
    )

    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction="none").to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()
    new_rows = []
    for name, path in new_dirs.items():
        print(f"Evaluating {name}: {path}", flush=True)
        rows = _evaluate_config(
            name, path, args.reference_dir, samples, args.batch_size,
            device, lpips, ssim, clip,
        )
        by_config[name] = {row["image_id"]: row for row in rows}
        new_rows.extend(rows)

    configs = (REFERENCE, STANDARD, LEGACY_E2, LEGACY_E0, W16, HW_E2, HW_E0)
    summary = []
    for config in configs:
        for metric in METRICS:
            values = _array(by_config, config, ids, metric)
            summary.append({
                "row_type": "config", "config": config, "reference": "",
                "metric": metric, "n": len(ids), "mean": float(values.mean()),
                "median": float(np.median(values)), "paired_mean_delta": "",
                "win_rate": "", "ci95_low": "", "ci95_high": "",
                "exploratory_ci": "", "headroom_recovery": "", "note": "",
            })

    paired = {}
    for ci, (candidate, baseline) in enumerate(COMPARISONS):
        paired[f"{candidate}_minus_{baseline}"] = {}
        for mi, metric in enumerate(METRICS):
            candidate_values = _array(by_config, candidate, ids, metric)
            baseline_values = _array(by_config, baseline, ids, metric)
            delta = candidate_values - baseline_values
            low, high = _bootstrap_ci(
                delta, args.bootstrap_samples, 20260812 + ci * 19 + mi
            )
            wins = candidate_values < baseline_values if metric == "lpips" else candidate_values > baseline_values
            record = {
                "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                "win_rate": float(wins.mean()), "ci95": [low, high],
            }
            paired[f"{candidate}_minus_{baseline}"][metric] = record
            summary.append({
                "row_type": "paired", "config": candidate, "reference": baseline,
                "metric": metric, "n": len(ids), "mean": "", "median": "",
                "paired_mean_delta": record["mean_delta"],
                "win_rate": record["win_rate"], "ci95_low": low,
                "ci95_high": high, "exploratory_ci": True,
                "headroom_recovery": "", "note": "candidate minus reference",
            })

    recovery = {}
    for metric in ("psnr", "lpips"):
        e2 = _array(by_config, HW_E2, ids, metric).mean()
        e0 = _array(by_config, HW_E0, ids, metric).mean()
        w16 = _array(by_config, W16, ids, metric).mean()
        if metric == "psnr":
            numerator, denominator = e0 - e2, w16 - e2
            formula = "(P_hwE0-P_hwE2)/(P_W16-P_hwE2)"
        else:
            numerator, denominator = e2 - e0, e2 - w16
            formula = "(L_hwE2-L_hwE0)/(L_hwE2-L_W16)"
        value = float(numerator / denominator) if denominator > 0 else None
        recovery[metric] = {
            "value": value, "numerator": float(numerator),
            "denominator": float(denominator), "formula": formula,
        }
        summary.append({
            "row_type": "headroom_recovery", "config": HW_E0,
            "reference": f"{HW_E2}->{W16}", "metric": metric,
            "n": len(ids), "mean": "", "median": "", "paired_mean_delta": "",
            "win_rate": "", "ci95_low": "", "ci95_high": "",
            "exploratory_ci": "", "headroom_recovery": "" if value is None else value,
            "note": formula + ("" if value is not None else "; undefined denominator"),
        })

    all_rows = []
    for config in configs[:-2]:
        all_rows.extend(by_config[config][image_id] for image_id in ids)
    all_rows.extend(new_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "hardware_weight_pilot32_per_prompt.csv", all_rows)
    _write_csv(args.output_dir / "hardware_weight_pilot32_summary.csv", summary)
    _write_csv(args.output_dir / "hardware_weight_pilot32_image_manifest.csv", integrity)
    (args.output_dir / "hardware_weight_pilot32_results.json").write_text(
        json.dumps({
            "n": len(ids), "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_intervals_are_exploratory": True,
            "paired_comparisons": paired,
            "hardware_e0_vs_e2_headroom_recovery": recovery,
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
