#!/usr/bin/env python3
"""Incremental paired evaluation for paper-faithful Four Over Six Pilot32."""

from __future__ import annotations

import argparse
import csv
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


NEW_CONFIGS = (
    "nvfp4-4over6",
    "e0m3-gscale1536",
    "tile-mix-e0-e2-4over6",
)
EXISTING_CONFIGS = (
    "fp16", "nvfp4-hw", "e0m3", "tile-mix-oracle", "a16w4-residual",
)
CONFIGS = EXISTING_CONFIGS + NEW_CONFIGS
METRICS = ("psnr", "lpips", "ssim", "clip_score")
COMPARISONS = (
    ("nvfp4-4over6", "nvfp4-hw"),
    ("nvfp4-4over6", "e0m3"),
    ("e0m3-gscale1536", "e0m3"),
    ("tile-mix-e0-e2-4over6", "e0m3-gscale1536"),
    ("tile-mix-e0-e2-4over6", "tile-mix-oracle"),
    ("tile-mix-e0-e2-4over6", "a16w4-residual"),
)


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(float(x) for x in np.quantile(values[indices].mean(1), (0.025, 0.975)))


def _load_existing(paths: list[Path], samples) -> dict[str, dict[str, dict]]:
    expected_ids = {image_id for image_id, _ in samples}
    rows = {config: {} for config in EXISTING_CONFIGS}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                config, image_id = row["config"], row["image_id"]
                if config not in rows or image_id not in expected_ids:
                    continue
                previous = rows[config].get(image_id)
                if previous is not None:
                    # Existing evaluator outputs intentionally overlap for
                    # fp16/HW/E0.  Require exact metric identity before reuse.
                    for metric in METRICS:
                        if float(previous[metric]) != float(row[metric]):
                            raise RuntimeError(
                                f"conflicting existing metric {config}/{image_id}/{metric}"
                            )
                    continue
                rows[config][image_id] = row
    for config, by_id in rows.items():
        if set(by_id) != expected_ids:
            missing = expected_ids - set(by_id)
            raise RuntimeError(f"{config}: missing {len(missing)} first32 metric rows")
    return rows


def _summary(indexed: dict[str, dict[str, dict]], ordered_ids, bootstrap_samples: int):
    output = []
    for config in CONFIGS:
        for metric in METRICS:
            values = np.asarray(
                [float(indexed[config][image_id][metric]) for image_id in ordered_ids],
                dtype=np.float64,
            )
            output.append({
                "row_type": "config_summary", "config": config, "reference": "",
                "metric": metric, "n": len(values), "mean": float(values.mean()),
                "median": float(np.median(values)), "paired_mean_delta": "",
                "ci95_low": "", "ci95_high": "", "prompt_win_rate": "",
            })
    for comparison_index, (candidate, reference) in enumerate(COMPARISONS):
        for metric_index, metric in enumerate(METRICS):
            candidate_values = np.asarray([
                float(indexed[candidate][image_id][metric]) for image_id in ordered_ids
            ])
            reference_values = np.asarray([
                float(indexed[reference][image_id][metric]) for image_id in ordered_ids
            ])
            delta = candidate_values - reference_values
            low, high = _bootstrap(
                delta, bootstrap_samples,
                20260812 + comparison_index * len(METRICS) + metric_index,
            )
            wins = candidate_values < reference_values if metric == "lpips" else (
                candidate_values > reference_values
            )
            output.append({
                "row_type": "paired_comparison", "config": candidate,
                "reference": reference, "metric": metric, "n": len(delta),
                "mean": "", "median": "", "paired_mean_delta": float(delta.mean()),
                "ci95_low": low, "ci95_high": high,
                "prompt_win_rate": float(wins.mean()),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--existing-per-prompt", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    roots = {config: args.new_root / config for config in NEW_CONFIGS}
    _validate_configs(list(roots.items()), samples)
    existing = _load_existing(args.existing_per_prompt, samples)
    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()

    indexed = dict(existing)
    new_rows = []
    for config in NEW_CONFIGS:
        rows = _evaluate_config(
            config, roots[config], args.reference_dir, samples,
            args.batch_size, device, lpips, ssim, clip,
        )
        indexed[config] = {row["image_id"]: row for row in rows}
        new_rows.extend(rows)

    combined = []
    for config in EXISTING_CONFIGS:
        combined.extend(existing[config].values())
    combined.extend(new_rows)
    ordered_ids = [image_id for image_id, _ in samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_prompt = args.output_dir / "fouroversix_pilot32_per_prompt.csv"
    summary = args.output_dir / "fouroversix_pilot32_summary.csv"
    _write_csv(per_prompt, combined)
    _write_csv(summary, _summary(indexed, ordered_ids, args.bootstrap_samples))
    print(f"Wrote {per_prompt}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
