#!/usr/bin/env python3
"""Evaluate FLUX adaptive-norm W4A16 against BF16-modulator controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
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


def load_control_rows(path: Path, count: int) -> dict[tuple[str, str], dict]:
    rows = list(csv.DictReader(path.open()))
    result = {}
    for row in rows:
        if row["config"] in SCHEMES:
            result[(row["config"], row["image_id"])] = {
                metric: float(row[metric]) for metric in METRICS
            }
    expected = count * len(SCHEMES)
    if len(result) != expected:
        raise RuntimeError(
            f"control metric coverage mismatch: {len(result)} != {expected}"
        )
    return result


def trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    trim = int(len(values) * proportion)
    ordered = np.sort(values)
    if trim:
        ordered = ordered[trim:-trim]
    return float(ordered.mean())


def build_grid(path: Path, configs, samples) -> None:
    indices = tuple(range(8)) + (15, 23, 31)
    tile, header, label = 256, 28, 20
    selected = [index for index in indices if index < len(samples)]
    canvas = Image.new(
        "RGB", (tile * len(configs), header + (tile + label) * len(selected)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for col, (name, _) in enumerate(configs):
        draw.text((col * tile + 4, 7), name, fill="black")
    for row, index in enumerate(selected):
        image_id, info = samples[index]
        y = header + row * (tile + label)
        draw.text((4, y), f"idx={index} id={image_id[:10]}", fill="black")
        for col, (_, root) in enumerate(configs):
            with Image.open(image_path(root, image_id, info)) as image:
                panel = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(panel, (col * tile, y + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--control-per-prompt-csv", type=Path, required=True)
    parser.add_argument("--w4a16-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.count)
    w4a16_configs = [
        (scheme, args.w4a16_root / "pilot32" / scheme) for scheme in SCHEMES
    ]
    control_configs = [(scheme, args.control_root / scheme) for scheme in SCHEMES]
    integrity = {
        "reference": validate_images([("bf16", args.reference_dir)], samples),
        "control_bf16_modulators": validate_images(control_configs, samples),
        "w4a16": validate_images(w4a16_configs, samples),
    }
    # Validate the historical CSV coverage for provenance, but recompute both
    # sides on the same device in this run to avoid CPU/GPU metric drift.
    load_control_rows(args.control_per_prompt_csv, args.count)

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
        control = {}
        for scheme, image_dir in control_configs:
            control_rows = evaluate_one(
                scheme, image_dir, args.reference_dir, samples,
                args.batch_size, device, lpips, ssim, clip,
            )
            for row in control_rows:
                control[(scheme, row["image_id"])] = {
                    metric: row[metric] for metric in METRICS
                }
        for scheme, image_dir in w4a16_configs:
            rows = evaluate_one(
                scheme, image_dir, args.reference_dir, samples,
                args.batch_size, device, lpips, ssim, clip,
            )
            summary = {"scheme": scheme, "n": len(rows)}
            for row in rows:
                base = control[(scheme, row["image_id"])]
                for metric in METRICS:
                    row[f"control_{metric}"] = base[metric]
                    row[f"w4a16_minus_control_{metric}"] = row[metric] - base[metric]
                per_prompt.append(row)

            for metric in METRICS:
                values = np.asarray([row[metric] for row in rows], dtype=np.float64)
                deltas = np.asarray(
                    [row[f"w4a16_minus_control_{metric}"] for row in rows],
                    dtype=np.float64,
                )
                summary[f"w4a16_{metric}_mean"] = float(values.mean())
                summary[f"w4a16_{metric}_median"] = float(np.median(values))
                control_values = np.asarray(
                    [control[(scheme, row["image_id"])][metric] for row in rows],
                    dtype=np.float64,
                )
                summary[f"control_{metric}_mean"] = float(control_values.mean())
                summary[f"control_{metric}_median"] = float(np.median(control_values))
                summary[f"delta_{metric}_mean"] = float(deltas.mean())
                summary[f"delta_{metric}_median"] = float(np.median(deltas))
                summary[f"delta_{metric}_trimmed_mean_10pct"] = trimmed_mean(deltas)
                summary[f"delta_{metric}_p25"] = float(np.quantile(deltas, 0.25))
                summary[f"delta_{metric}_p75"] = float(np.quantile(deltas, 0.75))
                summary[f"w4a16_win_rate_{metric}"] = float(
                    np.mean(deltas < 0) if metric == "lpips" else np.mean(deltas > 0)
                )
                low, high = bootstrap_ci(deltas, samples=5000)
                summary[f"delta_{metric}_ci95_low"] = low
                summary[f"delta_{metric}_ci95_high"] = high
            summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "w4a16_per_prompt.csv", per_prompt)
    write_csv(args.output_dir / "w4a16_summary.csv", summaries)
    (args.output_dir / "image_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    build_grid(
        args.output_dir / "w4a16_comparison_grid.png",
        [("BF16", args.reference_dir)] + [
            (f"{scheme}-W4A16", root) for scheme, root in w4a16_configs
        ],
        samples,
    )
    for scheme in SCHEMES:
        build_grid(
            args.output_dir / f"{scheme}_control_vs_w4a16_grid.png",
            [
                ("BF16", args.reference_dir),
                (f"{scheme}-control-W16A16-mod", args.control_root / scheme),
                (f"{scheme}-W4A16-mod", args.w4a16_root / "pilot32" / scheme),
            ],
            samples,
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
