#!/usr/bin/env python3
"""Paired FP16-reference evaluation for the fixed SANA-1.6B pilot64 study."""

from __future__ import annotations

import argparse
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
    _parse_config,
    _validate_configs,
    _write_csv,
)


CONFIG_ORDER = ("fp16", "nvfp4-hw", "e0m3", "tile-mix-oracle")
METRICS = ("psnr", "lpips", "ssim", "clip_score")
PAIRS = (
    ("e0m3", "nvfp4-hw"),
    ("tile-mix-oracle", "nvfp4-hw"),
    ("tile-mix-oracle", "e0m3"),
)


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _summary_rows(rows: list[dict], bootstrap_samples: int) -> list[dict]:
    by_config: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_config.setdefault(row["config"], {})[row["image_id"]] = row

    output = []
    for config in CONFIG_ORDER:
        config_rows = list(by_config[config].values())
        for metric in METRICS:
            values = np.asarray([row[metric] for row in config_rows], dtype=np.float64)
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

    for pair_index, (candidate, reference) in enumerate(PAIRS):
        candidate_by_id = by_config[candidate]
        reference_by_id = by_config[reference]
        image_ids = list(candidate_by_id)
        if set(image_ids) != set(reference_by_id):
            raise RuntimeError(f"unaligned prompt IDs for {candidate} vs {reference}")
        for metric_index, metric in enumerate(METRICS):
            candidate_values = np.asarray(
                [candidate_by_id[image_id][metric] for image_id in image_ids],
                dtype=np.float64,
            )
            reference_values = np.asarray(
                [reference_by_id[image_id][metric] for image_id in image_ids],
                dtype=np.float64,
            )
            delta = candidate_values - reference_values
            low, high = _bootstrap_ci(
                delta,
                bootstrap_samples,
                seed=20260811 + pair_index * len(METRICS) + metric_index,
            )
            lower_is_better = metric == "lpips"
            wins = candidate_values < reference_values if lower_is_better else (
                candidate_values > reference_values
            )
            output.append({
                "row_type": "paired_comparison",
                "config": candidate,
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
    parser.add_argument("--config", action="append", type=_parse_config, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configs = args.config
    names = tuple(name for name, _ in configs)
    if names != CONFIG_ORDER:
        raise RuntimeError(f"configs must be supplied in this order: {CONFIG_ORDER}")
    samples = _load_samples(args.dataset, args.count)
    _validate_configs(configs, samples)

    device = torch.device(args.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip_metric = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip_metric.model.eval()

    all_rows = []
    for name, root in configs:
        print(f"Evaluating {name}: {root}", flush=True)
        all_rows.extend(_evaluate_config(
            name,
            root,
            args.reference_dir,
            samples,
            args.batch_size,
            device,
            lpips_metric,
            ssim_metric,
            clip_metric,
        ))

    per_prompt_path = args.output_dir / "pilot64_per_prompt.csv"
    summary_path = args.output_dir / "pilot64_summary.csv"
    _write_csv(per_prompt_path, all_rows)
    _write_csv(summary_path, _summary_rows(all_rows, args.bootstrap_samples))
    print(f"Wrote {per_prompt_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
