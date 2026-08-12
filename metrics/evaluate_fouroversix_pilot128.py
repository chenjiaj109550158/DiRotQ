#!/usr/bin/env python3
"""Matched Pilot128 evaluation for fair E0 versus Four Over Six TileMix."""

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

from evaluate_pilot128 import (
    _bootstrap_ci,
    _evaluate_config,
    _load_samples,
    _validate_configs,
    _write_csv,
)


METRICS = ("psnr", "lpips", "ssim", "clip_score")
FULL_CONFIGS = (
    "nvfp4-hw", "e0m3", "tile-mix-oracle",
    "e0m3-gscale1536", "tile-mix-e0-e2-4over6",
)
COMPARISONS = (
    ("tile-mix-e0-e2-4over6", "e0m3-gscale1536", 128),
    ("tile-mix-e0-e2-4over6", "tile-mix-oracle", 128),
    ("tile-mix-e0-e2-4over6", "e0m3", 128),
    ("e0m3-gscale1536", "e0m3", 128),
    ("tile-mix-e0-e2-4over6", "a16w4-residual", 64),
)


def parse_config(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("config must be NAME=PATH")
    return name, Path(path)


def summarize(indexed: dict, ordered_ids: list[str]) -> list[dict]:
    rows = []
    for config, by_id in indexed.items():
        ids = ordered_ids[:64] if config == "a16w4-residual" else ordered_ids
        for metric in METRICS:
            values = np.asarray([float(by_id[image_id][metric]) for image_id in ids])
            rows.append({
                "config": config, "n": len(ids), "metric": metric,
                "mean": float(values.mean()), "median": float(np.median(values)),
            })
    return rows


def load_reference_rows(path: Path, ordered_ids: list[str]) -> list[dict]:
    expected = set(ordered_ids)
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["config"] not in {"fp16", "bf16"} or row["image_id"] not in expected:
                continue
            row["config"] = "fp16/bf16-reference"
            for metric in METRICS:
                row[metric] = float(row[metric])
            rows.append(row)
    by_id = {row["image_id"]: row for row in rows}
    if set(by_id) != expected or len(rows) != len(expected):
        raise RuntimeError("reference per-prompt metrics do not match the first128 IDs")
    return [by_id[image_id] for image_id in ordered_ids]


def compare(indexed: dict, ordered_ids: list[str], bootstrap_samples: int) -> list[dict]:
    rows = []
    segment_specs = {
        "first32": slice(0, 32), "new96": slice(32, 128), "full128": slice(0, 128)
    }
    comparison_index = 0
    for candidate, reference, count in COMPARISONS:
        segments = segment_specs if (candidate, reference) in {
            ("tile-mix-e0-e2-4over6", "e0m3-gscale1536"),
            ("tile-mix-e0-e2-4over6", "tile-mix-oracle"),
        } else {f"matched_first{count}": slice(0, count)}
        for segment, selection in segments.items():
            ids = ordered_ids[selection]
            for metric_index, metric in enumerate(METRICS):
                cand = np.asarray([float(indexed[candidate][i][metric]) for i in ids])
                ref = np.asarray([float(indexed[reference][i][metric]) for i in ids])
                delta = cand - ref
                low, high = _bootstrap_ci(
                    delta, bootstrap_samples,
                    20260813 + comparison_index * len(METRICS) + metric_index,
                )
                wins = cand < ref if metric == "lpips" else cand > ref
                rows.append({
                    "candidate": candidate, "reference": reference,
                    "segment": segment, "n": len(ids), "metric": metric,
                    "candidate_mean": float(cand.mean()),
                    "candidate_median": float(np.median(cand)),
                    "reference_mean": float(ref.mean()),
                    "reference_median": float(np.median(ref)),
                    "paired_mean_delta": float(delta.mean()),
                    "ci95_low": low, "ci95_high": high,
                    "prompt_win_rate": float(wins.mean()),
                })
            comparison_index += 1
    # The new TileMix metrics themselves are its absolute paired gap to the
    # FP16/BF16 image reference for PSNR/LPIPS/SSIM; CLIP is prompt alignment.
    new = indexed["tile-mix-e0-e2-4over6"]
    reference = indexed["fp16/bf16-reference"]
    for metric_index, metric in enumerate(METRICS):
        values = np.asarray([float(new[i][metric]) for i in ordered_ids])
        ref_values = np.asarray([float(reference[i][metric]) for i in ordered_ids])
        delta = values - ref_values
        if metric == "clip_score":
            low, high = _bootstrap_ci(delta, bootstrap_samples, 20260900 + metric_index)
            paired_delta, win_rate = float(delta.mean()), float((values > ref_values).mean())
        else:
            low = high = paired_delta = win_rate = ""
        rows.append({
            "candidate": "tile-mix-e0-e2-4over6", "reference": "fp16/bf16-image",
            "segment": "absolute_gap_full128", "n": 128, "metric": metric,
            "candidate_mean": float(values.mean()),
            "candidate_median": float(np.median(values)),
            "reference_mean": float(ref_values.mean()),
            "reference_median": float(np.median(ref_values)),
            "paired_mean_delta": paired_delta,
            "ci95_low": low, "ci95_high": high, "prompt_win_rate": win_rate,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-per-prompt", type=Path, required=True)
    parser.add_argument("--config", action="append", type=parse_config, required=True)
    parser.add_argument("--a16w4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_paths = dict(args.config)
    if set(config_paths) != set(FULL_CONFIGS):
        raise RuntimeError(f"expected configs {FULL_CONFIGS}, got {tuple(config_paths)}")
    samples = _load_samples(args.dataset, 128)
    samples64 = samples[:64]
    # Prefix sources such as SANA Pilot256 may contain extra later IDs, so
    # validate exact materialized views of the requested prefix paths only.
    expected = {(info["category"], image_id) for image_id, info in samples}
    for name, root in config_paths.items():
        materialized = {(path.parent.name, path.stem) for path in root.rglob("*.png")}
        if not expected.issubset(materialized):
            raise RuntimeError(f"{name}: missing matched first128 IDs")
    _validate_configs(
        [(name, root) for name, root in config_paths.items()
         if len(list(root.rglob("*.png"))) == 128],
        samples,
    )
    _validate_configs([("a16w4-residual", args.a16w4_dir)], samples64)

    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction="none").to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()
    indexed, per_prompt = {}, []
    for config in FULL_CONFIGS:
        print(f"Evaluating {config}: {config_paths[config]}", flush=True)
        rows = _evaluate_config(
            config, config_paths[config], args.reference_dir, samples,
            args.batch_size, device, lpips, ssim, clip,
        )
        indexed[config] = {row["image_id"]: row for row in rows}
        per_prompt.extend(rows)
    a16_rows = _evaluate_config(
        "a16w4-residual", args.a16w4_dir, args.reference_dir, samples64,
        args.batch_size, device, lpips, ssim, clip,
    )
    indexed["a16w4-residual"] = {row["image_id"]: row for row in a16_rows}
    per_prompt.extend(a16_rows)
    ids = [image_id for image_id, _ in samples]
    reference_rows = load_reference_rows(args.reference_per_prompt, ids)
    indexed["fp16/bf16-reference"] = {
        row["image_id"]: row for row in reference_rows
    }
    per_prompt.extend(reference_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "fouroversix_pilot128_per_prompt.csv", per_prompt)
    _write_csv(args.output_dir / "fouroversix_pilot128_summary.csv", summarize(indexed, ids))
    _write_csv(
        args.output_dir / "fouroversix_pilot128_paired.csv",
        compare(indexed, ids, args.bootstrap_samples),
    )
    provenance = {
        "count": 128, "a16w4_count": 64, "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "configs": {name: str(path) for name, path in config_paths.items()},
        "reference": str(args.reference_dir),
        "reference_per_prompt": str(args.reference_per_prompt),
        "a16w4": str(args.a16w4_dir),
    }
    with (args.output_dir / "metrics_provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    print(f"Wrote Pilot128 metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
