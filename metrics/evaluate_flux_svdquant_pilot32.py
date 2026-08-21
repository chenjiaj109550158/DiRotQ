#!/usr/bin/env python3
"""Evaluate matched SVDQuant and DiRotQ FLUX MJHQ-32 images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPScore

from evaluate_shared_pca_audit import (
    METRICS, bootstrap_ci, build_grid, evaluate_one, load_samples,
    validate_images, write_csv,
)


def _materialize_category_view(source: Path, destination: Path, samples) -> Path:
    """Create a symlink-only MJHQ category view without copying generated images."""
    source_by_id: dict[str, Path] = {}
    for path in source.rglob("*.png"):
        if path.stem in source_by_id:
            raise RuntimeError(f"duplicate image id {path.stem} under {source}")
        source_by_id[path.stem] = path.resolve()
    expected = {image_id for image_id, _ in samples}
    if set(source_by_id) != expected:
        raise RuntimeError(
            f"{source}: missing={len(expected-set(source_by_id))}, "
            f"extra={len(set(source_by_id)-expected)}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for image_id, info in samples:
        target = destination / info["category"] / f"{image_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == source_by_id[image_id]:
            continue
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"refusing to replace {target}")
        target.symlink_to(source_by_id[image_id])
    return destination


def summarize(rows: list[dict], config: str) -> dict:
    result = {"config": config, "n": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_median"] = float(np.median(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--svdquant-reference-dir", type=Path, required=True)
    parser.add_argument("--dirotq-reference-dir", type=Path, required=True)
    parser.add_argument("--svdquant-dir", type=Path, required=True)
    parser.add_argument("--dirotq-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    samples = load_samples(args.dataset, 32)
    view_root = args.output_dir / "category_views"
    svdquant_reference = _materialize_category_view(
        args.svdquant_reference_dir, view_root / "svdquant-reference", samples
    )
    svdquant_images = _materialize_category_view(
        args.svdquant_dir, view_root / "svdquant-images", samples
    )
    configs = [
        ("svdquant-int4-r32", svdquant_images, svdquant_reference),
        (
            "dirotq-shared-width-r64", args.dirotq_root / "shared-width",
            args.dirotq_reference_dir,
        ),
        (
            "dirotq-shared-operator-r64", args.dirotq_root / "shared-operator",
            args.dirotq_reference_dir,
        ),
    ]
    integrity = {
        "svdquant_bf16": validate_images(
            [("svdquant-bf16", svdquant_reference)], samples
        ),
        "dirotq_bf16": validate_images(
            [("dirotq-bf16", args.dirotq_reference_dir)], samples
        ),
        "candidates": validate_images([(name, root) for name, root, _ in configs], samples),
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

    all_rows: list[dict] = []
    summaries: list[dict] = []
    indexed: dict[str, dict[str, dict]] = {}
    comparisons: list[dict] = []
    with torch.inference_mode():
        for name, root, reference_root in configs:
            rows = evaluate_one(
                name, root, reference_root, samples, args.batch_size,
                device, lpips, ssim, clip,
            )
            all_rows.extend(rows)
            summaries.append(summarize(rows, name))
            indexed[name] = {row["image_id"]: row for row in rows}

        for other in ("dirotq-shared-width-r64", "dirotq-shared-operator-r64"):
            result = {"comparison": f"svdquant-minus-{other}", "n": len(samples)}
            for metric in METRICS:
                delta = np.asarray([
                    indexed["svdquant-int4-r32"][image_id][metric]
                    - indexed[other][image_id][metric]
                    for image_id, _ in samples
                ], dtype=np.float64)
                result[f"{metric}_delta_mean"] = float(delta.mean())
                result[f"{metric}_delta_median"] = float(np.median(delta))
                result[f"{metric}_win_rate"] = float(
                    np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
                )
                lo, hi = bootstrap_ci(delta, samples=5000, seed=20260821)
                result[f"{metric}_ci95_low"] = lo
                result[f"{metric}_ci95_high"] = hi
            comparisons.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_prompt.csv", all_rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    (args.output_dir / "image_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    build_grid(
        args.output_dir / "comparison_grid.png",
        [
            ("SVDQ BF16", svdquant_reference),
            ("SVDQuant", svdquant_images),
            ("DiRotQ BF16", args.dirotq_reference_dir),
            ("DiRotQ width", args.dirotq_root / "shared-width"),
            ("DiRotQ operator", args.dirotq_root / "shared-operator"),
        ],
        samples,
    )
    print(json.dumps({"summary": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
