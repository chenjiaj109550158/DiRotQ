#!/usr/bin/env python3
"""Compare matched FLUX.1-schnell quantization degradation across frameworks.

Pixel metrics are computed against each method's own BF16 framework reference;
the evaluator never compares a Nunchaku image to a DiRotQ BF16 image.
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

from evaluate_shared_pca_audit import (
    METRICS,
    bootstrap_ci,
    build_grid,
    evaluate_one,
    load_samples,
    validate_images,
)


def materialize_reference(source: Path, destination: Path, samples) -> Path:
    by_id = {}
    for path in source.rglob("*.png"):
        image_id = path.stem[:-2] if path.stem.endswith("-0") else path.stem
        if image_id in by_id:
            raise RuntimeError(f"duplicate image ID {image_id} under {source}")
        by_id[image_id] = path.resolve()
    expected = {image_id for image_id, _ in samples}
    if set(by_id) != expected:
        raise RuntimeError(
            f"reference ID mismatch: missing={sorted(expected-set(by_id))}, "
            f"extra={sorted(set(by_id)-expected)}"
        )
    for image_id, info in samples:
        path = destination / info["category"] / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        source_path = by_id[image_id]
        if path.is_symlink() and path.resolve() == source_path:
            continue
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace {path}")
        path.symlink_to(source_path)
    return destination


def summary(rows: list[dict]) -> dict:
    output = {"config": rows[0]["config"], "n": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_median"] = float(np.median(values))
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--svdquant-reference", type=Path, required=True)
    parser.add_argument("--svdquant-images", type=Path, required=True)
    parser.add_argument("--dirotq-reference", type=Path, required=True)
    parser.add_argument("--dirotq-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    samples = load_samples(args.dataset, 32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svd_reference = materialize_reference(
        args.svdquant_reference,
        args.output_dir / "category_views" / "svdquant-bf16",
        samples,
    )
    configs = [
        ("svdquant-int4-r32", args.svdquant_images, svd_reference),
        ("dirotq-shared-width-r64-nvfp4", args.dirotq_images, args.dirotq_reference),
    ]
    integrity = {}
    for name, images, reference in configs:
        integrity[name] = validate_images([(name, images), (name + "-bf16", reference)], samples)

    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()

    all_rows = []
    by_config = {}
    with torch.inference_mode():
        for name, images, reference in configs:
            rows = evaluate_one(
                name, images, reference, samples, args.batch_size,
                device, lpips, ssim, clip,
            )
            all_rows.extend(rows)
            by_config[name] = {row["image_id"]: row for row in rows}

    summaries = [summary([row for row in all_rows if row["config"] == name]) for name, _, _ in configs]
    first, second = (config[0] for config in configs)
    comparison = {"comparison": f"{first}-minus-{second}", "n": len(samples)}
    for metric in METRICS:
        delta = np.asarray(
            [
                by_config[first][image_id][metric] - by_config[second][image_id][metric]
                for image_id, _ in samples
            ],
            dtype=np.float64,
        )
        comparison[f"{metric}_delta_mean"] = float(delta.mean())
        comparison[f"{metric}_delta_median"] = float(np.median(delta))
        comparison[f"{metric}_win_rate"] = float(
            np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
        )
        low, high = bootstrap_ci(delta, samples=5000, seed=20260822)
        comparison[f"{metric}_ci95_low"] = low
        comparison[f"{metric}_ci95_high"] = high

    write_csv(args.output_dir / "per_prompt.csv", all_rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "paired_degradation_comparison.csv", [comparison])
    (args.output_dir / "image_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    build_grid(
        args.output_dir / "comparison_grid.png",
        [
            ("SVDQ BF16", svd_reference),
            ("SVDQ INT4", args.svdquant_images),
            ("DiRotQ BF16", args.dirotq_reference),
            ("DiRotQ NVFP4", args.dirotq_images),
        ],
        samples,
    )
    print(json.dumps({"summary": summaries, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
