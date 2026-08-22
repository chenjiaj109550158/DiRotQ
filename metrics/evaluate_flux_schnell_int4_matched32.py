#!/usr/bin/env python3
"""Evaluate matched FLUX.1-schnell INT4/NVFP4/SVDQuant images."""

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

from evaluate_shared_pca_audit import (
    METRICS,
    bootstrap_ci,
    build_grid,
    evaluate_one,
    load_samples,
    validate_images,
)


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/mjhq_5000_samples.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run_root.resolve()
    dataset = args.dataset.resolve()
    reference = run / "dirotq_bf16_reference32"
    configs = [
        ("dirotq-shared-width-r64-int4-w4a16", run / "dirotq_shared_width_r64_int4"),
        ("dirotq-shared-width-r64-nvfp4-w4a16", run / "dirotq_shared_width_r64_nvfp4"),
        ("svdquant-nunchaku-int4-r32", run / "svdquant_nunchaku_int4_r32_matched32"),
    ]
    output = run / "int4_matched_seed_metrics"
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(dataset, 32)
    integrity = validate_images(
        [("bf16-reference", reference), *configs], samples
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

    rows = []
    by_config = {}
    with torch.inference_mode():
        for name, images in configs:
            current = evaluate_one(
                name, images, reference, samples, 2, device, lpips, ssim, clip
            )
            rows.extend(current)
            by_config[name] = {row["image_id"]: row for row in current}

    summaries = []
    for name, _ in configs:
        current = list(by_config[name].values())
        summary = {"config": name, "n": len(current)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in current], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(np.median(values))
        summaries.append(summary)

    target = configs[0][0]
    comparisons = []
    for other, _ in configs[1:]:
        result = {"comparison": f"{target}-minus-{other}", "n": len(samples)}
        for metric in METRICS:
            delta = np.asarray(
                [
                    by_config[target][image_id][metric]
                    - by_config[other][image_id][metric]
                    for image_id, _ in samples
                ],
                dtype=np.float64,
            )
            result[f"{metric}_delta_mean"] = float(delta.mean())
            result[f"{metric}_delta_median"] = float(np.median(delta))
            result[f"{metric}_win_rate"] = float(
                np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
            )
            lo, high = bootstrap_ci(delta, samples=5000, seed=20260822)
            result[f"{metric}_ci95_low"] = lo
            result[f"{metric}_ci95_high"] = high
        comparisons.append(result)

    write_csv(output / "per_prompt.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_comparisons.csv", comparisons)
    (output / "image_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    build_grid(
        output / "comparison_grid.png",
        [("BF16", reference), *configs],
        samples,
    )
    print(json.dumps({"summary": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
