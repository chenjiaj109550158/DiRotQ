#!/usr/bin/env python3
"""Paired evaluation for the A16W4 residual-activation ceiling Pilot64."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPScore

from evaluate_pilot128 import (
    _evaluate_config,
    _load_samples,
    _validate_configs,
    _write_csv,
)


CANDIDATE = "a16w4-residual"
EXISTING = ("fp16", "nvfp4-hw", "e0m3")
METRICS = ("psnr", "lpips", "ssim", "clip_score")
PAIRED_REFERENCES = ("e0m3", "nvfp4-hw")


def _bootstrap_ci(values, samples, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(float(v) for v in np.quantile(values[indices].mean(1), (0.025, 0.975)))


def _load_existing(path, samples):
    expected_ids = {image_id for image_id, _ in samples}
    rows = {config: {} for config in EXISTING}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            config, image_id = row["config"], row["image_id"]
            if config in rows and image_id in expected_ids:
                if image_id in rows[config]:
                    raise RuntimeError(f"duplicate row: {config}/{image_id}")
                rows[config][image_id] = row
    for config, by_id in rows.items():
        if set(by_id) != expected_ids:
            raise RuntimeError(f"{config}: IDs do not match the dataset first64")
    return rows


def _summary(candidate_rows, existing, bootstrap_samples):
    candidate = {row["image_id"]: row for row in candidate_rows}
    ordered_ids = [row["image_id"] for row in candidate_rows]
    configs = {**existing, CANDIDATE: candidate}
    output = []
    means = {}
    for config, by_id in configs.items():
        means[config] = {}
        for metric in METRICS:
            values = np.asarray([float(by_id[i][metric]) for i in ordered_ids])
            means[config][metric] = float(values.mean())
            output.append({
                "row_type": "config_summary", "config": config, "reference": "",
                "metric": metric, "n": len(values), "mean": float(values.mean()),
                "median": float(np.median(values)), "paired_mean_delta": "",
                "ci95_low": "", "ci95_high": "", "prompt_win_rate": "",
                "recovery_fraction": "", "g_act_e0_minus_a16": "",
                "note": "",
            })

    for ref_index, reference in enumerate(PAIRED_REFERENCES):
        for metric_index, metric in enumerate(METRICS):
            a16 = np.asarray([float(candidate[i][metric]) for i in ordered_ids])
            ref = np.asarray([float(existing[reference][i][metric]) for i in ordered_ids])
            delta = a16 - ref
            low, high = _bootstrap_ci(
                delta, bootstrap_samples,
                20260812 + ref_index * len(METRICS) + metric_index,
            )
            wins = a16 < ref if metric == "lpips" else a16 > ref
            output.append({
                "row_type": "paired_comparison", "config": CANDIDATE,
                "reference": reference, "metric": metric, "n": len(delta),
                "mean": "", "median": "", "paired_mean_delta": float(delta.mean()),
                "ci95_low": low, "ci95_high": high,
                "prompt_win_rate": float(wins.mean()), "recovery_fraction": "",
                "g_act_e0_minus_a16": "", "note": "",
            })

    for metric in METRICS:
        e2, e0, a16 = (means[c][metric] for c in ("nvfp4-hw", "e0m3", CANDIDATE))
        if metric == "psnr":
            denominator = a16 - e2
            numerator = e0 - e2
            note = "(E0-E2)/(A16-E2)"
        elif metric == "lpips":
            denominator = e2 - a16
            numerator = e2 - e0
            note = "(E2-E0)/(E2-A16)"
        else:
            denominator = np.nan
            numerator = np.nan
            note = "recovery fraction is defined only for PSNR and LPIPS"
        expected_direction = numerator >= 0
        recovery = (
            numerator / denominator
            if denominator > 0 and expected_direction else ""
        )
        if metric in {"psnr", "lpips"}:
            if denominator <= 0:
                note += "; undefined because denominator is not positive"
            elif not expected_direction:
                note += "; undefined because E0-vs-E2 direction is reversed"
        output.append({
            "row_type": "ceiling_analysis", "config": "e0m3",
            "reference": CANDIDATE, "metric": metric, "n": len(ordered_ids),
            "mean": "", "median": "", "paired_mean_delta": "",
            "ci95_low": "", "ci95_high": "", "prompt_win_rate": "",
            "recovery_fraction": recovery,
            "g_act_e0_minus_a16": e0 - a16, "note": note,
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--existing-per-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    _validate_configs([(CANDIDATE, args.candidate_dir)], samples)
    existing = _load_existing(args.existing_per_prompt, samples)
    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()
    candidate_rows = _evaluate_config(
        CANDIDATE, args.candidate_dir, args.reference_dir, samples,
        args.batch_size, device, lpips, ssim, clip,
    )
    combined = []
    for config in EXISTING:
        combined.extend(existing[config].values())
    combined.extend(candidate_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "a16w4_ceiling64_per_prompt.csv", combined)
    _write_csv(
        args.output_dir / "a16w4_ceiling64_summary.csv",
        _summary(candidate_rows, existing, args.bootstrap_samples),
    )


if __name__ == "__main__":
    main()
