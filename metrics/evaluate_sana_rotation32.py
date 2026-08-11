#!/usr/bin/env python3
"""Paired SANA-1.6B residual-rotation causal ablation evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPScore

from evaluate_pilot128 import _evaluate_config, _image_path, _load_samples, _write_csv


CONFIGS = ("nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle")
METRICS = ("psnr", "lpips", "ssim", "clip_score")


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _validate_requested_images(
    name: str,
    root: Path,
    samples: list[tuple[str, dict]],
    *,
    allow_extras: bool,
) -> None:
    expected = {_image_path(root, image_id, info) for image_id, info in samples}
    actual = set(root.rglob("*.png"))
    missing, extras = expected - actual, actual - expected
    if missing or (extras and not allow_extras):
        raise RuntimeError(
            f"{name}: expected={len(expected)}, missing={len(missing)}, "
            f"extra={len(extras)}"
        )
    for image_id, info in samples:
        path = _image_path(root, image_id, info)
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1024, 1024):
                raise RuntimeError(
                    f"{name}/{image_id}: expected RGB 1024x1024, "
                    f"got {image.mode} {image.size}"
                )
            if all(low == high for low, high in image.getextrema()):
                raise RuntimeError(f"{name}/{image_id}: flat or corrupt image")


def _parse_metric_row(row: dict) -> dict:
    parsed = dict(row)
    parsed["prompt_index"] = int(parsed["prompt_index"])
    for metric in METRICS:
        parsed[metric] = float(parsed[metric])
    return parsed


def _load_random_rows(
    main_path: Path,
    block_path: Path,
    samples: list[tuple[str, dict]],
) -> list[dict]:
    requested_ids = {image_id for image_id, _ in samples}
    rows = []
    with main_path.open(newline="") as handle:
        rows.extend(
            _parse_metric_row(row) for row in csv.DictReader(handle)
            if row["config"] in {"nvfp4-hw", "e0m3", "tile-mix-oracle"}
            and row["image_id"] in requested_ids
        )
    with block_path.open(newline="") as handle:
        rows.extend(
            _parse_metric_row(row) for row in csv.DictReader(handle)
            if row["config"] == "block-mix-oracle" and row["image_id"] in requested_ids
        )
    for config in CONFIGS:
        config_rows = [row for row in rows if row["config"] == config]
        if len(config_rows) != len(samples):
            raise RuntimeError(
                f"existing random-R metrics incomplete for {config}: {len(config_rows)}"
            )
    return rows


def _group(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["rotation"], row["config"]), {})[row["image_id"]] = row
    return grouped


def _summary_rows(rows: list[dict], stats_root: Path) -> list[dict]:
    grouped = _group(rows)
    output = []
    for rotation in ("random", "identity"):
        for config in CONFIGS:
            config_rows = list(grouped[(rotation, config)].values())
            with (stats_root / rotation / f"{config}.json").open() as handle:
                stats = json.load(handle)
            summary = {
                "rotation": rotation,
                "config": config,
                "n": len(config_rows),
                "selection_unit": stats["selection_unit"],
                "e2m1_count": stats["e2m1_count"],
                "e0m3_count": stats["e0m3_count"],
                "e0m3_ratio": stats["e0m3_ratio"],
                "activation_signal_energy": stats["signal_energy"],
                "activation_reconstruction_sse": stats["reconstruction_sse"],
                "activation_qsnr_db": stats["qsnr_db"],
            }
            for metric in METRICS:
                values = np.asarray([row[metric] for row in config_rows], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"non-finite metric: {rotation}/{config}/{metric}")
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_median"] = float(np.median(values))
            output.append(summary)
    return output


def _best_identity_format(rows: list[dict]) -> str:
    grouped = _group(rows)
    # LPIPS is the primary paired fidelity metric; PSNR is a deterministic
    # tie-breaker if two means are numerically identical.
    candidates = []
    for config in CONFIGS:
        config_rows = grouped[("identity", config)].values()
        mean_lpips = float(np.mean([row["lpips"] for row in config_rows]))
        mean_psnr = float(np.mean([row["psnr"] for row in config_rows]))
        candidates.append((mean_lpips, -mean_psnr, config))
    return min(candidates)[2]


def _comparison_rows(
    rows: list[dict], bootstrap_samples: int, seed_base: int
) -> tuple[list[dict], str]:
    grouped = _group(rows)
    best_identity = _best_identity_format(rows)
    comparisons = []
    for rotation in ("random", "identity"):
        comparisons.extend([
            ("within_basis", rotation, "block-mix-oracle", rotation, "e0m3"),
            ("within_basis", rotation, "tile-mix-oracle", rotation, "e0m3"),
            ("within_basis", rotation, "block-mix-oracle", rotation, "nvfp4-hw"),
            ("within_basis", rotation, "tile-mix-oracle", rotation, "nvfp4-hw"),
        ])
    comparisons.extend([
        ("cross_basis_identity_best_vs_random_e0", "identity", best_identity,
         "random", "e0m3"),
        ("cross_basis_same_format", "identity", "block-mix-oracle",
         "random", "block-mix-oracle"),
        ("cross_basis_same_format", "identity", "tile-mix-oracle",
         "random", "tile-mix-oracle"),
    ])

    output = []
    for comparison_index, (kind, cand_rot, cand_cfg, ref_rot, ref_cfg) in enumerate(comparisons):
        candidate = grouped[(cand_rot, cand_cfg)]
        reference = grouped[(ref_rot, ref_cfg)]
        image_ids = sorted(candidate, key=lambda image_id: candidate[image_id]["prompt_index"])
        if set(image_ids) != set(reference):
            raise RuntimeError(
                f"unaligned comparison: {cand_rot}/{cand_cfg} vs {ref_rot}/{ref_cfg}"
            )
        for metric_index, metric in enumerate(METRICS):
            cand = np.asarray([candidate[image_id][metric] for image_id in image_ids])
            ref = np.asarray([reference[image_id][metric] for image_id in image_ids])
            delta = cand - ref
            low, high = _bootstrap_ci(
                delta, bootstrap_samples,
                seed_base + comparison_index * len(METRICS) + metric_index,
            )
            wins = cand < ref if metric == "lpips" else cand > ref
            output.append({
                "comparison_type": kind,
                "candidate_rotation": cand_rot,
                "candidate_config": cand_cfg,
                "reference_rotation": ref_rot,
                "reference_config": ref_cfg,
                "metric": metric,
                "n": len(image_ids),
                "paired_mean_delta": float(delta.mean()),
                "prompt_win_rate": float(wins.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })
    return output, best_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--random-per-prompt", type=Path, required=True)
    parser.add_argument("--random-block-per-prompt", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    _validate_requested_images("bf16-reference", args.reference_dir, samples, allow_extras=True)
    for config in CONFIGS:
        _validate_requested_images(
            f"random/{config}", args.random_root / config, samples, allow_extras=True
        )
        _validate_requested_images(
            f"identity/{config}", args.identity_root / config, samples, allow_extras=False
        )

    random_rows = _load_random_rows(
        args.random_per_prompt, args.random_block_per_prompt, samples
    )
    for row in random_rows:
        row["rotation"] = "random"

    device = torch.device(args.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip_metric = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip_metric.model.eval()

    identity_rows = []
    for config in CONFIGS:
        print(f"Evaluating identity/{config}", flush=True)
        current = _evaluate_config(
            config, args.identity_root / config, args.reference_dir, samples,
            args.batch_size, device, lpips_metric, ssim_metric, clip_metric,
        )
        for row in current:
            row["rotation"] = "identity"
        identity_rows.extend(current)

    fields = (
        "prompt_index", "image_id", "category", "prompt", "rotation", "config",
        "psnr", "lpips", "ssim", "clip_score",
    )
    rows = [
        {field: row[field] for field in fields}
        for row in random_rows + identity_rows
    ]
    comparisons, best_identity = _comparison_rows(
        rows, args.bootstrap_samples, seed_base=20260814
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "rotation32_per_prompt.csv", rows)
    _write_csv(args.output_dir / "rotation32_summary.csv", _summary_rows(rows, args.stats_root))
    _write_csv(args.output_dir / "rotation32_paired_comparisons.csv", comparisons)
    with (args.output_dir / "rotation32_selection.json").open("w") as handle:
        json.dump({
            "identity_best_format": best_identity,
            "selection_rule": "lowest mean LPIPS, then highest mean PSNR",
            "bootstrap_samples": args.bootstrap_samples,
        }, handle, indent=2)
        handle.write("\n")
    print(f"Identity best format: {best_identity}")
    print(f"Wrote SANA residual-rotation metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
