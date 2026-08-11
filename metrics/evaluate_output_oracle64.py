#!/usr/bin/env python3
"""Incremental paired evaluation for local partial-output TileMix Pilot64.

Only the new output-aware images are decoded by the neural metrics.  The four
comparison configurations are loaded from their existing per-prompt CSVs, so
no prior generation or metric job is repeated.
"""

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


CANDIDATE = "tile-mix-output-oracle"
REFERENCES = ("e0m3", "tile-mix-oracle", "nvfp4-hw", "block-mix-oracle")
CONFIGS = REFERENCES + (CANDIDATE,)
METRICS = ("psnr", "lpips", "ssim", "clip_score")


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _load_existing_rows(
    paths: list[Path], samples: list[tuple[str, dict]]
) -> dict[str, dict[str, dict]]:
    expected_ids = {image_id for image_id, _ in samples}
    by_config: dict[str, dict[str, dict]] = {}
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                config, image_id = row["config"], row["image_id"]
                if config not in REFERENCES or image_id not in expected_ids:
                    continue
                if image_id in by_config.setdefault(config, {}):
                    raise RuntimeError(f"duplicate existing row for {config}/{image_id}")
                by_config[config][image_id] = row
    if set(by_config) != set(REFERENCES):
        raise RuntimeError(
            f"existing CSVs have {sorted(by_config)}, expected {sorted(REFERENCES)}"
        )
    for config, by_id in by_config.items():
        if set(by_id) != expected_ids:
            raise RuntimeError(
                f"{config}: existing per-prompt IDs do not match dataset prefix"
            )
    return by_config


def _summary_rows(
    candidate_rows: list[dict],
    existing: dict[str, dict[str, dict]],
    bootstrap_samples: int,
) -> list[dict]:
    candidate_by_id = {row["image_id"]: row for row in candidate_rows}
    ordered_ids = [row["image_id"] for row in candidate_rows]
    all_rows = {
        **existing,
        CANDIDATE: candidate_by_id,
    }
    output = []
    for config in CONFIGS:
        for metric in METRICS:
            values = np.asarray(
                [float(all_rows[config][image_id][metric]) for image_id in ordered_ids],
                dtype=np.float64,
            )
            output.append({
                "row_type": "config_summary",
                "config": config,
                "reference": "",
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "paired_mean_delta": "",
                "ci95_low": "",
                "ci95_high": "",
                "prompt_win_rate": "",
            })

    for reference_index, reference in enumerate(REFERENCES):
        for metric_index, metric in enumerate(METRICS):
            candidate_values = np.asarray(
                [float(candidate_by_id[image_id][metric]) for image_id in ordered_ids]
            )
            reference_values = np.asarray(
                [float(existing[reference][image_id][metric]) for image_id in ordered_ids]
            )
            delta = candidate_values - reference_values
            low, high = _bootstrap_ci(
                delta,
                bootstrap_samples,
                seed=20260811 + reference_index * len(METRICS) + metric_index,
            )
            wins = (
                candidate_values < reference_values
                if metric == "lpips"
                else candidate_values > reference_values
            )
            output.append({
                "row_type": "paired_comparison",
                "config": CANDIDATE,
                "reference": reference,
                "metric": metric,
                "n": len(delta),
                "mean": "",
                "median": "",
                "paired_mean_delta": float(delta.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "prompt_win_rate": float(wins.mean()),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--existing-per-prompt", action="append", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    _validate_configs([(CANDIDATE, args.candidate_dir)], samples)
    existing = _load_existing_rows(args.existing_per_prompt, samples)

    device = torch.device(args.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip_metric = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip_metric.model.eval()

    candidate_rows = _evaluate_config(
        CANDIDATE,
        args.candidate_dir,
        args.reference_dir,
        samples,
        args.batch_size,
        device,
        lpips_metric,
        ssim_metric,
        clip_metric,
    )
    combined_rows = []
    for config in REFERENCES:
        combined_rows.extend(existing[config].values())
    combined_rows.extend(candidate_rows)

    per_prompt_path = args.output_dir / "output_oracle64_per_prompt.csv"
    summary_path = args.output_dir / "output_oracle64_summary.csv"
    _write_csv(per_prompt_path, combined_rows)
    _write_csv(
        summary_path,
        _summary_rows(candidate_rows, existing, args.bootstrap_samples),
    )
    print(f"Wrote {per_prompt_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
