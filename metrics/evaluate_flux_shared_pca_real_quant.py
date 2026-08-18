#!/usr/bin/env python3
"""Evaluate packed-real versus matched fake-quant FLUX shared-basis arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPScore

from evaluate_shared_pca_audit import (
    METRICS,
    bootstrap_ci,
    evaluate_one,
    image_path,
    load_samples,
    validate_images,
    write_csv,
)


SCHEMES = (
    "per-layer-pca",
    "shared-width",
    "shared-operator",
    "shared-operator-stage4",
    "representative-operator",
)


def read_fake_rows(path: Path, count: int) -> dict[tuple[str, str], dict]:
    rows = list(csv.DictReader(path.open()))
    result = {}
    for row in rows:
        if row["config"] not in SCHEMES:
            continue
        result[(row["config"], row["image_id"])] = {
            metric: float(row[metric]) for metric in METRICS
        }
    if len(result) != count * len(SCHEMES):
        raise RuntimeError(
            f"fake metric coverage mismatch: {len(result)} != {count * len(SCHEMES)}"
        )
    return result


def image_diff(real: Path, fake: Path) -> dict[str, float | bool]:
    real_bytes, fake_bytes = real.read_bytes(), fake.read_bytes()
    with Image.open(real) as a, Image.open(fake) as b:
        aa = np.asarray(a.convert("RGB"), dtype=np.int16)
        bb = np.asarray(b.convert("RGB"), dtype=np.int16)
    delta = np.abs(aa - bb)
    return {
        "png_sha_equal": hashlib.sha256(real_bytes).digest() == hashlib.sha256(fake_bytes).digest(),
        "pixel_max_abs": int(delta.max()),
        "pixel_mean_abs": float(delta.mean()),
    }


def parse_memory(log: Path) -> dict:
    text = log.read_text()
    storage_match = re.search(r"Real INT4 persistent storage: (\{.*\})", text)
    peak_match = re.search(
        r"Inference-only peak CUDA memory: allocated=(\d+) bytes, reserved=(\d+) bytes",
        text,
    )
    if not storage_match or not peak_match:
        raise RuntimeError(f"missing storage/peak record in {log}")
    return {
        **json.loads(storage_match.group(1)),
        "inference_peak_allocated": int(peak_match.group(1)),
        "inference_peak_reserved": int(peak_match.group(2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--fake-per-prompt-csv", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.count)
    real_configs = [(scheme, args.real_root / "pilot32" / scheme) for scheme in SCHEMES]
    fake_configs = [(scheme, args.fake_root / scheme) for scheme in SCHEMES]
    integrity = {
        "real": validate_images(real_configs, samples),
        "fake": validate_images(fake_configs, samples),
    }
    fake_rows = read_fake_rows(args.fake_per_prompt_csv, args.count)
    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()

    per_prompt = []
    summaries = []
    with torch.inference_mode():
        for scheme, real_dir in real_configs:
            rows = evaluate_one(
                scheme, real_dir, args.reference_dir, samples,
                args.batch_size, device, lpips, ssim, clip,
            )
            fake_dir = args.fake_root / scheme
            for row, (image_id, info) in zip(rows, samples):
                if row["image_id"] != image_id:
                    raise RuntimeError("metric/image ordering mismatch")
                fake = fake_rows[(scheme, image_id)]
                diff = image_diff(
                    image_path(real_dir, image_id, info),
                    image_path(fake_dir, image_id, info),
                )
                row.update(diff)
                for metric in METRICS:
                    row[f"fake_{metric}"] = fake[metric]
                    row[f"real_minus_fake_{metric}"] = row[metric] - fake[metric]
                per_prompt.append(row)

            summary = {"scheme": scheme, "n": len(rows)}
            for metric in METRICS:
                values = np.asarray([row[metric] for row in rows])
                deltas = np.asarray([row[f"real_minus_fake_{metric}"] for row in rows])
                summary[f"real_{metric}_mean"] = float(values.mean())
                summary[f"real_{metric}_median"] = float(np.median(values))
                summary[f"real_minus_fake_{metric}_mean"] = float(deltas.mean())
                summary[f"real_minus_fake_{metric}_median"] = float(np.median(deltas))
                summary[f"real_vs_fake_{metric}_win_rate"] = float(
                    np.mean(deltas < 0) if metric == "lpips" else np.mean(deltas > 0)
                )
                lo, hi = bootstrap_ci(deltas, samples=5000)
                summary[f"real_minus_fake_{metric}_ci95_low"] = lo
                summary[f"real_minus_fake_{metric}_ci95_high"] = hi
            summary["png_hash_equal_count"] = sum(row["png_sha_equal"] for row in rows)
            summary["pixel_max_abs"] = max(row["pixel_max_abs"] for row in rows)
            summary["pixel_mean_abs"] = float(np.mean([row["pixel_mean_abs"] for row in rows]))
            summary.update(parse_memory(args.real_root / "logs" / f"{scheme}-generate.log"))
            summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "real_quant_per_prompt.csv", per_prompt)
    write_csv(args.output_dir / "real_quant_summary.csv", summaries)
    (args.output_dir / "image_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

