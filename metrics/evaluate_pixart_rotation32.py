#!/usr/bin/env python3
"""Paired evaluation for the PixArt residual-random-rotation causal pilot."""

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
    return tuple(float(x) for x in np.quantile(means, (0.025, 0.975)))


def _validate_requested_images(
    name: str,
    root: Path,
    samples: list[tuple[str, dict]],
    *,
    require_no_extras: bool,
) -> None:
    expected = {_image_path(root, image_id, info) for image_id, info in samples}
    actual = set(root.rglob("*.png"))
    missing = expected - actual
    extras = actual - expected
    if missing or (require_no_extras and extras):
        raise RuntimeError(
            f"{name}: missing={len(missing)}, extra={len(extras)}, "
            f"expected={len(expected)}"
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


def _load_random_rows(
    path: Path, samples: list[tuple[str, dict]]
) -> tuple[list[dict], list[dict]]:
    requested_ids = {image_id for image_id, _ in samples}
    with path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    random32 = []
    random128 = []
    for row in raw:
        if row["config"] not in CONFIGS:
            continue
        parsed = dict(row)
        parsed["prompt_index"] = int(parsed["prompt_index"])
        for metric in METRICS:
            parsed[metric] = float(parsed[metric])
        random128.append(parsed)
        if parsed["image_id"] in requested_ids:
            random32.append(parsed)
    for config in CONFIGS:
        count32 = sum(row["config"] == config for row in random32)
        count128 = sum(row["config"] == config for row in random128)
        if count32 != len(samples) or count128 != 128:
            raise RuntimeError(
                f"existing random-R metrics incomplete for {config}: "
                f"first32={count32}, full128={count128}"
            )
    return random32, random128


def _group(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["rotation"], row["config"]), {})[row["image_id"]] = row
    return grouped


def _comparison_rows(
    rows: list[dict], bootstrap_samples: int, seed_base: int
) -> list[dict]:
    grouped = _group(rows)
    output = []
    comparisons = []
    for rotation in ("random", "identity"):
        comparisons.extend([
            (rotation, "block-mix-oracle", rotation, "e0m3", "within_rotation"),
            (rotation, "tile-mix-oracle", rotation, "e0m3", "within_rotation"),
        ])
    for config in CONFIGS:
        comparisons.append(("identity", config, "random", config, "cross_rotation"))
    comparisons.extend([
        ("identity", "block-mix-oracle", "random", "e0m3", "absolute_vs_random_e0"),
        ("identity", "tile-mix-oracle", "random", "e0m3", "absolute_vs_random_e0"),
    ])

    for comparison_index, (cand_rot, cand_cfg, ref_rot, ref_cfg, kind) in enumerate(comparisons):
        candidate = grouped[(cand_rot, cand_cfg)]
        reference = grouped[(ref_rot, ref_cfg)]
        ids = sorted(candidate, key=lambda image_id: candidate[image_id]["prompt_index"])
        if set(ids) != set(reference):
            raise RuntimeError(f"unaligned comparison: {cand_rot}/{cand_cfg} vs {ref_rot}/{ref_cfg}")
        for metric_index, metric in enumerate(METRICS):
            cand = np.asarray([candidate[image_id][metric] for image_id in ids])
            ref = np.asarray([reference[image_id][metric] for image_id in ids])
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
                "n": len(ids),
                "paired_mean_delta": float(delta.mean()),
                "prompt_win_rate": float(wins.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })
    return output


def _random128_block_vs_e0(
    rows: list[dict], bootstrap_samples: int, seed_base: int
) -> list[dict]:
    by_config = {
        config: {row["image_id"]: row for row in rows if row["config"] == config}
        for config in ("block-mix-oracle", "e0m3")
    }
    ids = sorted(
        by_config["block-mix-oracle"],
        key=lambda image_id: by_config["block-mix-oracle"][image_id]["prompt_index"],
    )
    output = []
    for metric_index, metric in enumerate(METRICS):
        candidate = np.asarray([
            by_config["block-mix-oracle"][image_id][metric] for image_id in ids
        ])
        reference = np.asarray([by_config["e0m3"][image_id][metric] for image_id in ids])
        delta = candidate - reference
        low, high = _bootstrap_ci(delta, bootstrap_samples, seed_base + metric_index)
        wins = candidate < reference if metric == "lpips" else candidate > reference
        output.append({
            "candidate_rotation": "random",
            "candidate_config": "block-mix-oracle",
            "reference_rotation": "random",
            "reference_config": "e0m3",
            "metric": metric,
            "n": len(ids),
            "paired_mean_delta": float(delta.mean()),
            "prompt_win_rate": float(wins.mean()),
            "ci95_low": low,
            "ci95_high": high,
        })
    return output


def _summary_rows(rows: list[dict], stats_root: Path) -> list[dict]:
    grouped = _group(rows)
    output = []
    for rotation in ("random", "identity"):
        for config in CONFIGS:
            config_rows = list(grouped[(rotation, config)].values())
            stats_path = stats_root / rotation / f"{config}.json"
            with stats_path.open() as handle:
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
                values = np.asarray([row[metric] for row in config_rows])
                if not np.isfinite(values).all():
                    raise RuntimeError(f"non-finite metric: {rotation}/{config}/{metric}")
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_median"] = float(np.median(values))
            output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--random-per-prompt", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    for config in CONFIGS:
        _validate_requested_images(
            f"random/{config}", args.random_root / config, samples,
            require_no_extras=False,
        )
        _validate_requested_images(
            f"identity/{config}", args.identity_root / config, samples,
            require_no_extras=True,
        )
    _validate_requested_images(
        "fp16-reference", args.reference_dir, samples, require_no_extras=False
    )

    random32, random128 = _load_random_rows(args.random_per_prompt, samples)
    for row in random32:
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

    # Normalize column order across reused and newly evaluated rows.
    all_rows = []
    fields = (
        "prompt_index", "image_id", "category", "prompt", "rotation", "config",
        "psnr", "lpips", "ssim", "clip_score",
    )
    for row in random32 + identity_rows:
        all_rows.append({field: row[field] for field in fields})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "rotation32_per_prompt.csv", all_rows)
    _write_csv(args.output_dir / "rotation32_summary.csv", _summary_rows(all_rows, args.stats_root))
    _write_csv(
        args.output_dir / "rotation32_paired_comparisons.csv",
        _comparison_rows(all_rows, args.bootstrap_samples, seed_base=20260813),
    )
    _write_csv(
        args.output_dir / "random128_blockmix_vs_e0m3.csv",
        _random128_block_vs_e0(random128, args.bootstrap_samples, seed_base=20260913),
    )
    print(f"Wrote causal-pilot metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
