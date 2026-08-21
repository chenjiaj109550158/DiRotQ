#!/usr/bin/env python3
"""Evaluate rank-64/BF16-scale Scheme A against matched rank-384 shared-width."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--rank384-dir", type=Path, required=True)
    parser.add_argument("--scheme-a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.count)
    configs = [
        ("shared-width-r384-fp32-scales", args.rank384_dir),
        ("scheme-a-r64-bf16-scales", args.scheme_a_dir),
    ]
    integrity = validate_images(configs, samples)
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
    with torch.inference_mode():
        for name, root in configs:
            rows.extend(evaluate_one(
                name, root, args.reference_dir, samples, args.batch_size,
                device, lpips, ssim, clip,
            ))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["config"], {})[row["image_id"]] = row
    baseline = grouped[configs[0][0]]
    summaries = []
    for name, _ in configs:
        aligned = [grouped[name][image_id] for image_id, _ in samples]
        summary = {"config": name, "n": len(aligned)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in aligned], dtype=np.float64)
            control = np.asarray(
                [baseline[row["image_id"]][metric] for row in aligned],
                dtype=np.float64,
            )
            delta = values - control
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_delta_vs_rank384_mean"] = float(delta.mean())
            summary[f"{metric}_delta_vs_rank384_median"] = float(np.median(delta))
            summary[f"{metric}_win_rate_vs_rank384"] = float(
                np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
            )
            low, high = bootstrap_ci(delta, samples=5000)
            summary[f"{metric}_delta_vs_rank384_ci95_low"] = low
            summary[f"{metric}_delta_vs_rank384_ci95_high"] = high
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scheme_a_per_prompt.csv", rows)
    write_csv(args.output_dir / "scheme_a_summary.csv", summaries)
    (args.output_dir / "image_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    build_grid(
        args.output_dir / "scheme_a_comparison_grid.png",
        [("BF16", args.reference_dir), *configs], samples,
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
