#!/usr/bin/env python3
"""Incremental paired evaluation for the SANA Pilot64 BlockMix upper bound."""

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


EXISTING_CONFIGS = ("fp16", "nvfp4-hw", "e0m3", "tile-mix-oracle")
REFERENCES = ("nvfp4-hw", "e0m3", "tile-mix-oracle")
METRICS = ("psnr", "lpips", "ssim", "clip_score")


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _load_existing_rows(path: Path, samples: list[tuple[str, dict]]) -> dict[str, dict[str, dict]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected_ids = {image_id for image_id, _ in samples}
    by_config: dict[str, dict[str, dict]] = {}
    for row in rows:
        config = row["config"]
        if config not in EXISTING_CONFIGS:
            raise RuntimeError(f"unexpected existing config: {config}")
        by_config.setdefault(config, {})[row["image_id"]] = row
    if set(by_config) != set(EXISTING_CONFIGS):
        raise RuntimeError("existing per-prompt CSV does not contain all Pilot64 configs")
    for config, by_id in by_config.items():
        if set(by_id) != expected_ids:
            raise RuntimeError(f"{config}: existing per-prompt IDs do not match dataset first 64")
    return by_config


def _summary_rows(
    block_rows: list[dict],
    existing: dict[str, dict[str, dict]],
    bootstrap_samples: int,
) -> list[dict]:
    block_by_id = {row["image_id"]: row for row in block_rows}
    output = []
    for metric in METRICS:
        values = np.asarray([row[metric] for row in block_rows], dtype=np.float64)
        output.append({
            "row_type": "config_summary",
            "config": "block-mix-oracle",
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

    image_ids = [image_id for image_id, _ in block_by_id.items()]
    for reference_index, reference in enumerate(REFERENCES):
        reference_by_id = existing[reference]
        for metric_index, metric in enumerate(METRICS):
            candidate_values = np.asarray(
                [float(block_by_id[image_id][metric]) for image_id in image_ids]
            )
            reference_values = np.asarray(
                [float(reference_by_id[image_id][metric]) for image_id in image_ids]
            )
            delta = candidate_values - reference_values
            low, high = _bootstrap_ci(
                delta,
                bootstrap_samples,
                seed=20260812 + reference_index * len(METRICS) + metric_index,
            )
            lower_is_better = metric == "lpips"
            wins = candidate_values < reference_values if lower_is_better else (
                candidate_values > reference_values
            )
            output.append({
                "row_type": "paired_comparison",
                "config": "block-mix-oracle",
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
    parser.add_argument("--blockmix-dir", type=Path, required=True)
    parser.add_argument("--existing-per-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    _validate_configs([("block-mix-oracle", args.blockmix_dir)], samples)
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

    block_rows = _evaluate_config(
        "block-mix-oracle",
        args.blockmix_dir,
        args.reference_dir,
        samples,
        args.batch_size,
        device,
        lpips_metric,
        ssim_metric,
        clip_metric,
    )
    per_prompt_path = args.output_dir / "pilot64_blockmix_per_prompt.csv"
    summary_path = args.output_dir / "pilot64_blockmix_summary.csv"
    _write_csv(per_prompt_path, block_rows)
    _write_csv(summary_path, _summary_rows(block_rows, existing, args.bootstrap_samples))
    print(f"Wrote {per_prompt_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
