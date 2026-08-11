#!/usr/bin/env python3
"""Incremental paired evaluation for the SANA E0-aware joint-GPTQ Pilot16."""

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


CANDIDATE = "e0joint-gptq"
STANDARD = "e0m3"
CEILING = "a16w4-residual"
METRICS = ("psnr", "lpips", "ssim", "clip_score")


def _load_config(path: Path, config: str, expected_ids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["config"] != config or row["image_id"] not in expected_ids:
                continue
            if row["image_id"] in rows:
                raise RuntimeError(f"duplicate existing row: {config}/{row['image_id']}")
            rows[row["image_id"]] = row
    if set(rows) != expected_ids:
        raise RuntimeError(f"{config}: existing metric IDs do not match Pilot16")
    return rows


def _summary(candidate_rows: list[dict], standard: dict, ceiling: dict) -> list[dict]:
    candidate = {row["image_id"]: row for row in candidate_rows}
    ids = [row["image_id"] for row in candidate_rows]
    configs = {STANDARD: standard, CEILING: ceiling, CANDIDATE: candidate}
    output: list[dict] = []
    means: dict[str, dict[str, float]] = {}
    for config, by_id in configs.items():
        means[config] = {}
        for metric in METRICS:
            values = np.asarray([float(by_id[i][metric]) for i in ids])
            means[config][metric] = float(values.mean())
            output.append({
                "row_type": "config_summary", "config": config, "reference": "",
                "metric": metric, "n": len(values), "mean": float(values.mean()),
                "median": float(np.median(values)), "paired_mean_delta": "",
                "prompt_win_rate": "", "headroom_recovery": "", "note": "",
            })

    for reference, by_id in ((STANDARD, standard), (CEILING, ceiling)):
        for metric in METRICS:
            joint = np.asarray([float(candidate[i][metric]) for i in ids])
            ref = np.asarray([float(by_id[i][metric]) for i in ids])
            delta = joint - ref
            wins = joint < ref if metric == "lpips" else joint > ref
            output.append({
                "row_type": "paired_comparison", "config": CANDIDATE,
                "reference": reference, "metric": metric, "n": len(delta),
                "mean": "", "median": "", "paired_mean_delta": float(delta.mean()),
                "prompt_win_rate": float(wins.mean()), "headroom_recovery": "",
                "note": "candidate minus reference",
            })

    for metric in ("psnr", "lpips"):
        e0 = means[STANDARD][metric]
        joint = means[CANDIDATE][metric]
        a16 = means[CEILING][metric]
        if metric == "psnr":
            numerator, denominator = joint - e0, a16 - e0
            note = "(P_joint-P_E0)/(P_A16-P_E0)"
        else:
            numerator, denominator = e0 - joint, e0 - a16
            note = "(L_E0-L_joint)/(L_E0-L_A16)"
        recovery: float | str = numerator / denominator if denominator > 0 else ""
        if denominator <= 0:
            note += "; undefined because denominator is not positive"
        output.append({
            "row_type": "headroom_recovery", "config": CANDIDATE,
            "reference": f"{STANDARD}->{CEILING}", "metric": metric, "n": len(ids),
            "mean": "", "median": "", "paired_mean_delta": "",
            "prompt_win_rate": "", "headroom_recovery": recovery, "note": note,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--standard-per-prompt", type=Path, required=True)
    parser.add_argument("--ceiling-per-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    _validate_configs([(CANDIDATE, args.candidate_dir)], samples)
    expected_ids = {image_id for image_id, _ in samples}
    standard = _load_config(args.standard_per_prompt, STANDARD, expected_ids)
    ceiling = _load_config(args.ceiling_per_prompt, CEILING, expected_ids)

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = list(standard.values()) + list(ceiling.values()) + candidate_rows
    _write_csv(args.output_dir / "e0joint_pilot16_per_prompt.csv", combined)
    _write_csv(
        args.output_dir / "e0joint_pilot16_summary.csv",
        _summary(candidate_rows, standard, ceiling),
    )


if __name__ == "__main__":
    main()
