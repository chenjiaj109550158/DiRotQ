#!/usr/bin/env python3
"""SANA E0-activation x W16 residual-weight ceiling gate."""

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
    _evaluate_config,
    _load_samples,
    _validate_configs,
    _write_csv,
)


CANDIDATE = "e0a-w16-residual"
BASELINE = "e0m3"
REFERENCE = "fp16"  # SANA is BF16; legacy directory/config label is fp16.
METRICS = ("psnr", "lpips", "ssim", "clip_score")


def _bootstrap(values: np.ndarray, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(float(x) for x in np.quantile(values[indices].mean(1), (.025, .975)))


def _load_existing(path: Path, samples):
    ids = [image_id for image_id, _ in samples]
    output = {config: {} for config in (REFERENCE, BASELINE)}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            config = row["config"]
            if config in output and row["image_id"] in ids:
                output[config][row["image_id"]] = row
    for config, rows in output.items():
        if set(rows) != set(ids):
            raise RuntimeError(f"{config}: existing metrics do not match first32 IDs")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--existing-per-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
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
    candidate = {row["image_id"]: row for row in candidate_rows}
    ids = [image_id for image_id, _ in samples]

    summary = []
    for config, rows in ((REFERENCE, existing[REFERENCE]),
                         (BASELINE, existing[BASELINE]),
                         (CANDIDATE, candidate)):
        for metric in METRICS:
            values = np.asarray([float(rows[i][metric]) for i in ids])
            summary.append({
                "row_type": "config", "config": config, "reference": "",
                "metric": metric, "n": len(values),
                "mean": float(values.mean()), "median": float(np.median(values)),
                "paired_delta": "", "ci95_low": "", "ci95_high": "",
                "win_rate": "",
            })

    gate_values = {}
    for metric_index, metric in enumerate(METRICS):
        w16 = np.asarray([float(candidate[i][metric]) for i in ids])
        e2w = np.asarray([float(existing[BASELINE][i][metric]) for i in ids])
        delta = w16 - e2w
        low, high = _bootstrap(delta, args.bootstrap_samples, 20260812 + metric_index)
        wins = w16 < e2w if metric == "lpips" else w16 > e2w
        summary.append({
            "row_type": "paired", "config": CANDIDATE, "reference": BASELINE,
            "metric": metric, "n": len(delta), "mean": "", "median": "",
            "paired_delta": float(delta.mean()), "ci95_low": low,
            "ci95_high": high, "win_rate": float(wins.mean()),
        })
        gate_values[metric] = {
            "candidate_minus_baseline": float(delta.mean()),
            "ci95": [low, high], "win_rate": float(wins.mean()),
        }

    psnr_gain = gate_values["psnr"]["candidate_minus_baseline"]
    lpips_gain = -gate_values["lpips"]["candidate_minus_baseline"]
    passed = psnr_gain >= .15 or lpips_gain >= .003
    gate = {
        "candidate": CANDIDATE, "baseline": "E0A x existing standard E2 GPTQ",
        "n": len(ids), "bootstrap_samples": args.bootstrap_samples,
        "psnr_improvement_db": psnr_gain, "lpips_improvement": lpips_gain,
        "thresholds": {"psnr_db": .15, "lpips": .003},
        "passed": passed,
        "classification_if_failed": "WEIGHT HEADROOM LIMITED",
        "metrics": gate_values,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in (REFERENCE, BASELINE):
        rows.extend(existing[config].values())
    rows.extend(candidate_rows)
    _write_csv(args.output_dir / "weight_headroom32_per_prompt.csv", rows)
    _write_csv(args.output_dir / "weight_headroom32_summary.csv", summary)
    (args.output_dir / "weight_headroom32_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
