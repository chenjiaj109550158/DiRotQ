#!/usr/bin/env python3
"""Paired BF16/control metrics for rank-64 NVFP4 W4A16 Pilot32."""

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


SCHEMES = ("shared-width", "shared-operator")


def summarize(rows: list[dict], prefix: str) -> dict:
    out = {"config": prefix, "n": len(rows)}
    for metric in METRICS:
        values = np.asarray([r[metric] for r in rows], dtype=np.float64)
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_median"] = float(np.median(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--w4a16-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    samples = load_samples(args.dataset, 32)
    controls = [(s, args.control_root / s) for s in SCHEMES]
    tested = [(s, args.w4a16_root / "pilot32" / s) for s in SCHEMES]
    integrity = {
        "bf16": validate_images([("bf16", args.reference_dir)], samples),
        "bf16_weight_controls": validate_images(controls, samples),
        "nvfp4_w4a16": validate_images(tested, samples),
    }
    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="none", normalize=True).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction="none").to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()

    all_rows, summaries, comparisons = [], [], []
    indexed = {}
    with torch.inference_mode():
        for kind, configs in (("control", controls), ("w4a16", tested)):
            for scheme, root in configs:
                rows = evaluate_one(
                    f"{kind}-{scheme}", root, args.reference_dir, samples,
                    args.batch_size, device, lpips, ssim, clip,
                )
                all_rows.extend(rows)
                indexed[(kind, scheme)] = {r["image_id"]: r for r in rows}
                summaries.append(summarize(rows, f"{kind}-{scheme}"))

        for scheme in SCHEMES:
            result = {"comparison": f"w4a16-minus-bf16-weight-{scheme}", "n": 32}
            for metric in METRICS:
                delta = np.asarray([
                    indexed[("w4a16", scheme)][image_id][metric]
                    - indexed[("control", scheme)][image_id][metric]
                    for image_id, _ in samples
                ])
                result[f"{metric}_delta_mean"] = float(delta.mean())
                result[f"{metric}_delta_median"] = float(np.median(delta))
                result[f"{metric}_win_rate"] = float(
                    np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
                )
                lo, hi = bootstrap_ci(delta, samples=5000, seed=20260820)
                result[f"{metric}_ci95_low"] = lo
                result[f"{metric}_ci95_high"] = hi
            comparisons.append(result)

        result = {"comparison": "w4a16-shared-operator-minus-shared-width", "n": 32}
        for metric in METRICS:
            delta = np.asarray([
                indexed[("w4a16", "shared-operator")][image_id][metric]
                - indexed[("w4a16", "shared-width")][image_id][metric]
                for image_id, _ in samples
            ])
            result[f"{metric}_delta_mean"] = float(delta.mean())
            result[f"{metric}_delta_median"] = float(np.median(delta))
            result[f"{metric}_win_rate"] = float(
                np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
            )
            lo, hi = bootstrap_ci(delta, samples=5000, seed=20260820)
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
        [("BF16", args.reference_dir)] + [
            (f"{s}-NVFP4-W4A16", args.w4a16_root / "pilot32" / s) for s in SCHEMES
        ], samples,
    )
    print(json.dumps({"summary": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
