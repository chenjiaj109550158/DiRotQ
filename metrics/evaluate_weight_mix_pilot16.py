#!/usr/bin/env python3
"""Paired SANA Pilot16 evaluation for E0-activation weight formats.

The BF16 reference, existing standard-E2 result, and W16 ceiling metrics are
read from the already completed matched pilots.  Only the three newly created
weight-cache configurations are evaluated here.  Bootstrap intervals are
exploratory because n=16 is not used as a significance gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
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
W16 = "w16-ceiling"
NEW_CONFIGS = (
    "reoptimized-fixed-e2",
    "reoptimized-fixed-e0",
    "weight-tilemix",
)
W4_CONFIGS = (STANDARD, *NEW_CONFIGS)
METRICS = ("psnr", "lpips", "ssim", "clip_score")
COMPARISONS = (
    ("reoptimized-fixed-e2", STANDARD),
    ("reoptimized-fixed-e0", "reoptimized-fixed-e2"),
    ("weight-tilemix", "reoptimized-fixed-e0"),
    ("weight-tilemix", "reoptimized-fixed-e2"),
    ("weight-tilemix", STANDARD),
    *((config, W16) for config in W4_CONFIGS),
)


def _read_existing(
    pilot64_path: Path, headroom32_path: Path, expected_ids: set[str]
) -> dict[str, dict[str, dict]]:
    wanted = {"fp16": REFERENCE, "e0m3": STANDARD}
    output = {name: {} for name in (REFERENCE, STANDARD, W16)}
    with pilot64_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = wanted.get(row["config"])
            if name is not None and row["image_id"] in expected_ids:
                copied = dict(row)
                copied["config"] = name
                output[name][row["image_id"]] = copied
    with headroom32_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["config"] == "e0a-w16-residual" and row["image_id"] in expected_ids:
                copied = dict(row)
                copied["config"] = W16
                output[W16][row["image_id"]] = copied
    for name, rows in output.items():
        if set(rows) != expected_ids:
            raise RuntimeError(f"{name}: reused metric IDs do not match Pilot16")
    return output


def _arrays(by_config: dict[str, dict[str, dict]], config: str, ids: list[str], metric: str):
    return np.asarray([float(by_config[config][image_id][metric]) for image_id in ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--pilot64-per-prompt", type=Path, required=True)
    parser.add_argument("--headroom32-per-prompt", type=Path, required=True)
    parser.add_argument("--reoptimized-fixed-e2-dir", type=Path, required=True)
    parser.add_argument("--reoptimized-fixed-e0-dir", type=Path, required=True)
    parser.add_argument("--weight-tilemix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    ids = [image_id for image_id, _ in samples]
    expected_ids = set(ids)
    new_dirs = {
        "reoptimized-fixed-e2": args.reoptimized_fixed_e2_dir,
        "reoptimized-fixed-e0": args.reoptimized_fixed_e0_dir,
        "weight-tilemix": args.weight_tilemix_dir,
    }
    _validate_configs(list(new_dirs.items()), samples)
    by_config = _read_existing(
        args.pilot64_per_prompt, args.headroom32_per_prompt, expected_ids
    )

    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()

    new_rows = []
    for name, path in new_dirs.items():
        print(f"Evaluating {name}: {path}", flush=True)
        rows = _evaluate_config(
            name, path, args.reference_dir, samples, args.batch_size, device,
            lpips, ssim, clip,
        )
        by_config[name] = {row["image_id"]: row for row in rows}
        new_rows.extend(rows)

    config_rows = []
    for config in (REFERENCE, *W4_CONFIGS, W16):
        for metric in METRICS:
            values = _arrays(by_config, config, ids, metric)
            config_rows.append({
                "row_type": "config", "config": config, "reference": "",
                "metric": metric, "n": len(ids), "mean": float(values.mean()),
                "median": float(np.median(values)), "paired_mean_delta": "",
                "win_rate": "", "ci95_low": "", "ci95_high": "",
                "exploratory_ci": "", "headroom_recovery": "",
                "note": "",
            })

    comparison_rows = []
    for comparison_index, (candidate, baseline) in enumerate(COMPARISONS):
        for metric_index, metric in enumerate(METRICS):
            candidate_values = _arrays(by_config, candidate, ids, metric)
            baseline_values = _arrays(by_config, baseline, ids, metric)
            delta = candidate_values - baseline_values
            low, high = _bootstrap_ci(
                delta, args.bootstrap_samples,
                seed=20260812 + 17 * comparison_index + metric_index,
            )
            wins = candidate_values < baseline_values if metric == "lpips" else candidate_values > baseline_values
            comparison_rows.append({
                "row_type": "paired", "config": candidate,
                "reference": baseline, "metric": metric, "n": len(ids),
                "mean": "", "median": "",
                "paired_mean_delta": float(delta.mean()),
                "win_rate": float(wins.mean()), "ci95_low": low,
                "ci95_high": high, "exploratory_ci": True,
                "headroom_recovery": "", "note": "candidate minus reference",
            })

    recovery_rows = []
    recovery = {}
    for config in W4_CONFIGS:
        recovery[config] = {}
        for metric in ("psnr", "lpips"):
            standard = _arrays(by_config, STANDARD, ids, metric).mean()
            candidate = _arrays(by_config, config, ids, metric).mean()
            ceiling = _arrays(by_config, W16, ids, metric).mean()
            if metric == "psnr":
                numerator, denominator = candidate - standard, ceiling - standard
                formula = "(P_q-P_stdE2)/(P_W16-P_stdE2)"
            else:
                numerator, denominator = standard - candidate, standard - ceiling
                formula = "(L_stdE2-L_q)/(L_stdE2-L_W16)"
            value = float(numerator / denominator) if denominator > 0 else None
            recovery[config][metric] = value
            recovery_rows.append({
                "row_type": "headroom_recovery", "config": config,
                "reference": f"{STANDARD}->{W16}", "metric": metric,
                "n": len(ids), "mean": "", "median": "",
                "paired_mean_delta": "", "win_rate": "", "ci95_low": "",
                "ci95_high": "", "exploratory_ci": "",
                "headroom_recovery": "" if value is None else value,
                "note": formula + ("" if value is not None else "; undefined denominator"),
            })

    all_rows = []
    for config in (REFERENCE, STANDARD, W16):
        all_rows.extend(by_config[config][image_id] for image_id in ids)
    all_rows.extend(new_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "weight_mix_pilot16_per_prompt.csv", all_rows)
    _write_csv(
        args.output_dir / "weight_mix_pilot16_summary.csv",
        config_rows + comparison_rows + recovery_rows,
    )
    (args.output_dir / "weight_mix_pilot16_recovery.json").write_text(
        json.dumps({
            "n": len(ids), "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_intervals_are_exploratory": True,
            "weight_headroom_recovery": recovery,
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
