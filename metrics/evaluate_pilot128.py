#!/usr/bin/env python3
"""Paired per-prompt evaluation for the fixed PixArt-Sigma pilot128 study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPScore
from torchmetrics.multimodal.clip_score import _clip_score_update


GRID_INDICES = tuple(range(8)) + (15, 31, 47, 63, 79, 95, 111, 127)
BOOTSTRAP_CONFIGS = ("block-mix-oracle", "tile-mix-oracle", "e0m3")
BASELINE = "nvfp4-hw"


def _parse_config(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("config must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("config must be NAME=PATH")
    return name, Path(path)


def _load_samples(dataset_path: Path, count: int) -> list[tuple[str, dict]]:
    with dataset_path.open() as f:
        samples = list(json.load(f).items())[:count]
    if len(samples) != count:
        raise RuntimeError(f"dataset contains only {len(samples)} samples, expected {count}")
    return samples


def _image_path(root: Path, img_id: str, info: dict) -> Path:
    return root / info["category"] / f"{img_id}.png"


def _validate_configs(
    configs: list[tuple[str, Path]], samples: list[tuple[str, dict]]
) -> None:
    expected = {f"{info['category']}/{img_id}.png" for img_id, info in samples}
    for name, root in configs:
        actual = {str(path.relative_to(root)) for path in root.rglob("*.png")}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise RuntimeError(
                f"{name}: expected exactly {len(expected)} target PNGs, "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        for img_id, info in samples:
            path = _image_path(root, img_id, info)
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (1024, 1024):
                    raise RuntimeError(
                        f"{name}/{img_id}: expected RGB 1024x1024, "
                        f"got {image.mode} {image.size}"
                    )
                extrema = image.getextrema()
                if all(low == high for low, high in extrema):
                    raise RuntimeError(f"{name}/{img_id}: flat or corrupt image")


def _load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _evaluate_config(
    name: str,
    root: Path,
    reference_root: Path,
    samples: list[tuple[str, dict]],
    batch_size: int,
    device: torch.device,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    ssim_metric: StructuralSimilarityIndexMeasure,
    clip_metric: CLIPScore,
) -> list[dict]:
    lpips_metric.reset()
    ssim_metric.reset()
    psnr_values: list[float] = []
    clip_values: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            generated = torch.stack([
                _load_tensor(_image_path(root, img_id, info)) for img_id, info in batch
            ]).to(device)
            reference = torch.stack([
                _load_tensor(_image_path(reference_root, img_id, info))
                for img_id, info in batch
            ]).to(device)

            mse = (generated - reference).square().flatten(1).mean(1)
            psnr = torch.where(
                mse == 0,
                torch.full_like(mse, torch.inf),
                -10.0 * torch.log10(mse),
            )
            psnr_values.extend(psnr.cpu().tolist())
            lpips_metric.update(generated, reference)
            ssim_metric.update(generated, reference)

            prompts = [info["prompt"] for _, info in batch]
            clip_images = generated.mul(255).round().to(torch.uint8)
            clip_scores, _ = _clip_score_update(
                clip_images, prompts, clip_metric.model, clip_metric.processor
            )
            clip_values.extend(clip_scores.detach().cpu().tolist())

    lpips_values = lpips_metric.compute().detach().cpu().flatten().tolist()
    ssim_values = ssim_metric.compute().detach().cpu().flatten().tolist()
    if not all(len(values) == len(samples) for values in
               (psnr_values, lpips_values, ssim_values, clip_values)):
        raise RuntimeError(f"{name}: metric result count mismatch")

    rows = []
    for index, ((img_id, info), psnr, lpips, ssim, clip) in enumerate(zip(
        samples, psnr_values, lpips_values, ssim_values, clip_values
    )):
        values = (lpips, ssim, clip)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{name}/{img_id}: non-finite LPIPS/SSIM/CLIP result")
        rows.append({
            "prompt_index": index,
            "image_id": img_id,
            "category": info["category"],
            "prompt": info["prompt"],
            "config": name,
            "psnr": psnr,
            "lpips": lpips,
            "ssim": ssim,
            "clip_score": clip,
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], bootstrap_samples: int) -> list[dict]:
    by_config: dict[str, list[dict]] = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)
    baseline = by_config[BASELINE]
    baseline_by_id = {row["image_id"]: row for row in baseline}
    metrics = ("psnr", "lpips", "ssim", "clip_score")
    summary = []

    for config, config_rows in by_config.items():
        out: dict[str, int | float | str] = {"config": config, "n": len(config_rows)}
        aligned_baseline = [baseline_by_id[row["image_id"]] for row in config_rows]
        for metric in metrics:
            values = np.asarray([row[metric] for row in config_rows], dtype=np.float64)
            base = np.asarray([row[metric] for row in aligned_baseline], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_median"] = float(np.median(values))
            delta = values - base
            out[f"{metric}_paired_mean_delta_vs_{BASELINE}"] = float(delta.mean())
            if config in BOOTSTRAP_CONFIGS:
                low, high = _bootstrap_ci(
                    delta, bootstrap_samples, seed=20260810 + BOOTSTRAP_CONFIGS.index(config)
                )
                out[f"{metric}_delta_ci95_low"] = low
                out[f"{metric}_delta_ci95_high"] = high
            else:
                out[f"{metric}_delta_ci95_low"] = ""
                out[f"{metric}_delta_ci95_high"] = ""
        psnr = np.asarray([row["psnr"] for row in config_rows])
        base_psnr = np.asarray([row["psnr"] for row in aligned_baseline])
        lpips = np.asarray([row["lpips"] for row in config_rows])
        base_lpips = np.asarray([row["lpips"] for row in aligned_baseline])
        out[f"psnr_win_rate_vs_{BASELINE}"] = float(np.mean(psnr > base_psnr))
        out[f"lpips_win_rate_vs_{BASELINE}"] = float(np.mean(lpips < base_lpips))
        summary.append(out)
    return summary


def _build_grid(
    output: Path,
    manifest_path: Path,
    configs: list[tuple[str, Path]],
    samples: list[tuple[str, dict]],
) -> None:
    thumb = 256
    header = 28
    row_label = 20
    canvas = Image.new(
        "RGB", (thumb * len(configs), header + (thumb + row_label) * len(GRID_INDICES)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, (name, _) in enumerate(configs):
        draw.text((column * thumb + 4, 7), name, fill="black")

    manifest = []
    for row_index, sample_index in enumerate(GRID_INDICES):
        img_id, info = samples[sample_index]
        y = header + row_index * (thumb + row_label)
        draw.text((4, y), f"idx={sample_index} id={img_id}", fill="black")
        for column, (_, root) in enumerate(configs):
            with Image.open(_image_path(root, img_id, info)) as image:
                panel = image.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(panel, (column * thumb, y + row_label))
        manifest.append({
            "prompt_index": sample_index,
            "image_id": img_id,
            "category": info["category"],
            "prompt": info["prompt"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    with manifest_path.open("w") as f:
        json.dump({"config_order": [name for name, _ in configs], "samples": manifest}, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--config", action="append", type=_parse_config, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configs = args.config
    names = [name for name, _ in configs]
    if len(names) != len(set(names)) or BASELINE not in names:
        raise RuntimeError("config names must be unique and include nvfp4-hw")
    samples = _load_samples(args.dataset, args.count)
    _validate_configs(configs, samples)

    device = torch.device(args.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip_metric = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip_metric.model.eval()

    all_rows = []
    for name, root in configs:
        print(f"Evaluating {name}: {root}", flush=True)
        all_rows.extend(_evaluate_config(
            name, root, args.reference_dir, samples, args.batch_size, device,
            lpips_metric, ssim_metric, clip_metric,
        ))

    per_prompt_path = args.output_dir / "pilot128_per_prompt.csv"
    summary_path = args.output_dir / "pilot128_summary.csv"
    _write_csv(per_prompt_path, all_rows)
    _write_csv(summary_path, _summarize(all_rows, args.bootstrap_samples))
    _build_grid(
        args.output_dir / "pilot128_comparison_grid.png",
        args.output_dir / "pilot128_comparison_grid_manifest.json",
        configs,
        samples,
    )
    print(f"Wrote {per_prompt_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
