#!/usr/bin/env python3
"""Matched PixArt-128 evaluator for shared-PCA quality arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPScore
from torchmetrics.multimodal.clip_score import _clip_score_update


BASELINE = "per-layer-pca"
METRICS = ("psnr", "lpips", "ssim", "clip_score")
GRID_INDICES = tuple(range(8)) + (15, 31, 47, 63, 79, 95, 111, 127)


def parse_config(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("config must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_samples(path: Path, count: int) -> list[tuple[str, dict]]:
    samples = list(json.loads(path.read_text()).items())[:count]
    if len(samples) != count:
        raise RuntimeError(f"dataset has {len(samples)} samples, expected {count}")
    return samples


def image_path(root: Path, image_id: str, info: dict) -> Path:
    return root / info["category"] / f"{image_id}.png"


def validate_images(configs, samples) -> dict:
    expected = {f"{info['category']}/{image_id}.png" for image_id, info in samples}
    report = {}
    for name, root in configs:
        actual = {str(path.relative_to(root)) for path in root.rglob("*.png")}
        if actual != expected:
            raise RuntimeError(
                f"{name}: missing={len(expected-actual)}, extra={len(actual-expected)}"
            )
        extrema = []
        for image_id, info in samples:
            with Image.open(image_path(root, image_id, info)) as image:
                image.load()
                if image.mode != "RGB" or image.size != (1024, 1024):
                    raise RuntimeError(f"{name}/{image_id}: {image.mode} {image.size}")
                ex = image.getextrema()
                if all(lo == hi for lo, hi in ex):
                    raise RuntimeError(f"{name}/{image_id}: flat image")
                extrema.append(ex)
        report[name] = {"count": len(actual), "valid_rgb_1024_nonflat": True}
    return report


def load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def bootstrap_ci(values: np.ndarray, samples: int = 5000, seed: int = 20260818):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(float(x) for x in np.quantile(values[idx].mean(1), (0.025, 0.975)))


def summarize(rows: list[dict], bootstrap_samples: int = 5000) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["config"], []).append(row)
    if BASELINE not in grouped:
        raise ValueError(f"missing {BASELINE} baseline")
    baseline = {row["image_id"]: row for row in grouped[BASELINE]}
    output = []
    for config, config_rows in grouped.items():
        result = {"config": config, "n": len(config_rows)}
        aligned = [baseline[row["image_id"]] for row in config_rows]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in config_rows], dtype=np.float64)
            base = np.asarray([row[metric] for row in aligned], dtype=np.float64)
            delta = values - base
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_median"] = float(np.median(values))
            result[f"{metric}_delta_vs_{BASELINE}"] = float(delta.mean())
            result[f"{metric}_delta_median_vs_{BASELINE}"] = float(np.median(delta))
            result[f"{metric}_win_rate_vs_{BASELINE}"] = float(
                np.mean(delta < 0) if metric == "lpips" else np.mean(delta > 0)
            )
            lo, hi = bootstrap_ci(delta, bootstrap_samples, seed=20260818 + len(output))
            result[f"{metric}_delta_ci95_low"] = lo
            result[f"{metric}_delta_ci95_high"] = hi
        output.append(result)
    return output


def screen_decision(summary: list[dict]) -> dict:
    decisions = {}
    for row in summary:
        if row["config"] == BASELINE:
            continue
        psnr_ok = row[f"psnr_delta_vs_{BASELINE}"] >= -0.10
        lpips_ok = row[f"lpips_delta_vs_{BASELINE}"] <= 0.002
        psnr_ci = row["psnr_delta_ci95_low"] >= 0
        lpips_ci = row["lpips_delta_ci95_high"] <= 0
        clip_not_worse = row["clip_score_delta_ci95_low"] >= 0
        decisions[row["config"]] = {
            "eligible_for_5k": bool(psnr_ok and lpips_ok and not (
                row["psnr_delta_ci95_high"] < 0 or row["lpips_delta_ci95_low"] > 0
            ) and not (row["clip_score_delta_ci95_high"] < 0)),
            "psnr_within_tolerance": psnr_ok,
            "lpips_within_tolerance": lpips_ok,
            "psnr_ci_supports_improvement": psnr_ci,
            "lpips_ci_supports_improvement": lpips_ci,
            "clip_ci_supports_non_degradation": clip_not_worse,
        }
    return decisions


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(name, root, reference_root, samples, batch_size, device, lpips, ssim, clip):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            generated = torch.stack([load_tensor(image_path(root, i, x)) for i, x in batch]).to(device)
            reference = torch.stack([load_tensor(image_path(reference_root, i, x)) for i, x in batch]).to(device)
            mse = (generated - reference).square().flatten(1).mean(1)
            psnr_values = (-10 * mse.log10()).cpu().tolist()
            lpips_values = lpips(generated, reference).flatten().cpu().tolist()
            ssim_values = ssim(generated, reference).flatten().cpu().tolist()
            clip_images = generated.mul(255).round().to(torch.uint8)
            clip_values, _ = _clip_score_update(
                clip_images, [x["prompt"] for _, x in batch], clip.model, clip.processor
            )
            for (image_id, info), p, l, s, c in zip(
                batch, psnr_values, lpips_values, ssim_values, clip_values.cpu().tolist()
            ):
                if not all(math.isfinite(v) for v in (p, l, s, c)):
                    raise RuntimeError(f"{name}/{image_id}: non-finite metric")
                rows.append({
                    "image_id": image_id, "category": info["category"],
                    "prompt": info["prompt"], "config": name,
                    "psnr": p, "lpips": l, "ssim": s, "clip_score": c,
                })
    return rows


def build_grid(path, configs, samples):
    indices = [i for i in GRID_INDICES if i < len(samples)]
    tile, header, label = 256, 28, 20
    canvas = Image.new("RGB", (tile * len(configs), header + (tile + label) * len(indices)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (name, _) in enumerate(configs):
        draw.text((col * tile + 4, 7), name, fill="black")
    for row, index in enumerate(indices):
        image_id, info = samples[index]
        y = header + row * (tile + label)
        draw.text((4, y), f"idx={index} id={image_id}", fill="black")
        for col, (_, root) in enumerate(configs):
            with Image.open(image_path(root, image_id, info)) as image:
                panel = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(panel, (col * tile, y + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--config", action="append", type=parse_config, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    names = [name for name, _ in args.config]
    if len(names) != len(set(names)) or BASELINE not in names:
        raise RuntimeError(f"configs must be unique and include {BASELINE}")
    samples = load_samples(args.dataset, args.count)
    integrity = validate_images(args.config, samples)
    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="none", normalize=True).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction="none").to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()
    rows = []
    for name, root in args.config:
        print(f"Evaluating {name}: {root}", flush=True)
        rows.extend(evaluate_one(name, root, args.reference_dir, samples, args.batch_size, device, lpips, ssim, clip))
    summary = summarize(rows, args.bootstrap_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "shared_pca_per_prompt.csv", rows)
    write_csv(args.output_dir / "shared_pca_summary.csv", summary)
    (args.output_dir / "screen_decision.json").write_text(json.dumps(screen_decision(summary), indent=2) + "\n")
    (args.output_dir / "image_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    build_grid(args.output_dir / "shared_pca_comparison_grid.png", args.config, samples)


if __name__ == "__main__":
    main()

