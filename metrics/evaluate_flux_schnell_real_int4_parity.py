#!/usr/bin/env python3
"""Compare packed-real and matched fake INT4 FLUX.1-schnell Pilot32."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPScore

from evaluate_shared_pca_audit import (
    METRICS,
    bootstrap_ci,
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


def image_path(root: Path, image_id: str, category: str) -> Path:
    return root / category / f"{image_id}.png"


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
    fake = run / "dirotq_shared_width_r64_int4"
    real = run / "real_int4_fused/images"
    output = run / "real_int4_fused/quality_parity"
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(dataset, 32)
    integrity = validate_images(
        [("bf16-reference", reference), ("fake-int4", fake), ("real-int4", real)],
        samples,
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

    rows: list[dict] = []
    indexed: dict[str, dict[str, dict]] = {}
    with torch.inference_mode():
        for name, path in (("fake-int4", fake), ("real-int4", real)):
            current = evaluate_one(
                name, path, reference, samples, 2, device, lpips, ssim, clip
            )
            rows.extend(current)
            indexed[name] = {row["image_id"]: row for row in current}

    summaries = []
    for name in ("fake-int4", "real-int4"):
        current = list(indexed[name].values())
        summary = {"config": name, "n": len(current)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in current], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(np.median(values))
        summaries.append(summary)

    comparison = {"comparison": "real-minus-fake", "n": len(samples)}
    for metric in METRICS:
        delta = np.asarray(
            [
                indexed["real-int4"][image_id][metric]
                - indexed["fake-int4"][image_id][metric]
                for image_id, _ in samples
            ],
            dtype=np.float64,
        )
        comparison[f"{metric}_delta_mean"] = float(delta.mean())
        comparison[f"{metric}_delta_median"] = float(np.median(delta))
        comparison[f"{metric}_delta_abs_mean"] = float(np.abs(delta).mean())
        comparison[f"{metric}_win_rate"] = float(
            np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
        )
        lo, hi = bootstrap_ci(delta, samples=5000, seed=20260822)
        comparison[f"{metric}_ci95_low"] = lo
        comparison[f"{metric}_ci95_high"] = hi

    pixel_rows = []
    for image_id, info in samples:
        category = info["category"]
        fake_pixels = np.asarray(
            Image.open(image_path(fake, image_id, category)).convert("RGB"),
            dtype=np.int16,
        )
        real_pixels = np.asarray(
            Image.open(image_path(real, image_id, category)).convert("RGB"),
            dtype=np.int16,
        )
        error = real_pixels.astype(np.float64) - fake_pixels.astype(np.float64)
        mse = float(np.mean(error * error))
        pixel_rows.append(
            {
                "image_id": image_id,
                "category": category,
                "unequal_channels": int(np.count_nonzero(error)),
                "channel_count": int(error.size),
                "max_abs_u8": int(np.max(np.abs(error))),
                "mean_abs_u8": float(np.mean(np.abs(error))),
                "mse_u8": mse,
                "psnr_real_vs_fake": (
                    float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
                ),
            }
        )
    pixel_summary = {
        "exact_images": sum(row["unequal_channels"] == 0 for row in pixel_rows),
        "images": len(pixel_rows),
        "max_abs_u8": max(row["max_abs_u8"] for row in pixel_rows),
        "mean_abs_u8": float(np.mean([row["mean_abs_u8"] for row in pixel_rows])),
        "mean_psnr_real_vs_fake": float(
            np.mean([row["psnr_real_vs_fake"] for row in pixel_rows])
        ),
    }

    write_csv(output / "per_prompt.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "pixel_parity.csv", pixel_rows)
    (output / "result.json").write_text(
        json.dumps(
            {
                "integrity": integrity,
                "summaries": summaries,
                "paired": comparison,
                "pixel_parity": pixel_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"summaries": summaries, "paired": comparison, "pixel_parity": pixel_summary},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
