#!/usr/bin/env python3
"""Paired BF16 image metrics for rank-64 shared-basis legacy NVFP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPScore

from evaluate_shared_pca_audit import (
    METRICS,
    bootstrap_ci,
    build_grid,
    evaluate_one,
    load_samples,
    validate_images,
    write_csv,
)


CONFIGS = ("shared-width", "shared-operator")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    samples = load_samples(args.dataset, 32)
    configs = [(name, args.pilot_root / name) for name in CONFIGS]
    integrity = {
        "bf16": validate_images([("bf16", args.reference_dir)], samples),
        "nvfp4": validate_images(configs, samples),
    }
    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()

    rows = []
    summaries = []
    with torch.inference_mode():
        for name, root in configs:
            arm = evaluate_one(
                name, root, args.reference_dir, samples, args.batch_size,
                device, lpips, ssim, clip,
            )
            rows.extend(arm)
            summary = {"config": name, "n": len(arm)}
            for metric in METRICS:
                values = np.asarray([row[metric] for row in arm], dtype=np.float64)
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_median"] = float(np.median(values))
            summaries.append(summary)

        indexed = {
            name: {row["image_id"]: row for row in rows if row["config"] == name}
            for name in CONFIGS
        }
        comparison = {"comparison": "shared-operator-minus-shared-width", "n": 32}
        for metric in METRICS:
            delta = np.asarray([
                indexed["shared-operator"][image_id][metric]
                - indexed["shared-width"][image_id][metric]
                for image_id, _ in samples
            ], dtype=np.float64)
            comparison[f"{metric}_delta_mean"] = float(delta.mean())
            comparison[f"{metric}_delta_median"] = float(np.median(delta))
            comparison[f"{metric}_win_rate"] = float(
                np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
            )
            lo, hi = bootstrap_ci(delta, samples=5000, seed=20260820)
            comparison[f"{metric}_ci95_low"] = lo
            comparison[f"{metric}_ci95_high"] = hi

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "nvfp4_per_prompt.csv", rows)
    write_csv(args.output_dir / "nvfp4_summary.csv", summaries)
    write_csv(args.output_dir / "nvfp4_paired_comparison.csv", [comparison])
    (args.output_dir / "image_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    build_grid(
        args.output_dir / "nvfp4_comparison_grid.png",
        [("BF16", args.reference_dir), *configs], samples,
    )
    print(json.dumps({"summary": summaries, "paired": comparison}, indent=2))


if __name__ == "__main__":
    main()
