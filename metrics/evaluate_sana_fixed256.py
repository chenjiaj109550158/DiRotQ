#!/usr/bin/env python3
"""Evaluate the fixed E0M3 SANA-256 confirmation and cross-model deltas."""

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

from evaluate_pilot128 import (
    _evaluate_config,
    _load_samples,
    _validate_configs,
    _write_csv,
)


CONFIGS = ("bf16", "nvfp4-hw", "e0m3")
METRICS = ("psnr", "lpips", "ssim", "clip_score")
GRID_INDICES = tuple(range(8)) + (31, 63, 95, 127, 159, 191, 223, 255)


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(float(x) for x in np.quantile(values[indices].mean(1), (0.025, 0.975)))


def _index_rows(rows: list[dict]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for row in rows:
        result.setdefault(row["config"], {})[row["image_id"]] = row
    return result


def _summary(rows: list[dict], bootstrap_samples: int) -> list[dict]:
    indexed = _index_rows(rows)
    ordered_ids = list(indexed["e0m3"])
    output = []
    for config in CONFIGS:
        out: dict[str, str | int | float] = {"config": config, "n": len(ordered_ids)}
        for metric in METRICS:
            values = np.asarray([float(indexed[config][i][metric]) for i in ordered_ids])
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_median"] = float(np.median(values))
            if config == "e0m3":
                baseline = np.asarray([
                    float(indexed["nvfp4-hw"][i][metric]) for i in ordered_ids
                ])
                delta = values - baseline
                low, high = _bootstrap_ci(
                    delta, bootstrap_samples, 20260811 + METRICS.index(metric)
                )
                wins = values < baseline if metric == "lpips" else values > baseline
                out[f"{metric}_paired_delta_vs_nvfp4-hw"] = float(delta.mean())
                out[f"{metric}_delta_ci95_low"] = low
                out[f"{metric}_delta_ci95_high"] = high
                out[f"{metric}_win_rate_vs_nvfp4-hw"] = float(wins.mean())
            else:
                out[f"{metric}_paired_delta_vs_nvfp4-hw"] = ""
                out[f"{metric}_delta_ci95_low"] = ""
                out[f"{metric}_delta_ci95_high"] = ""
                out[f"{metric}_win_rate_vs_nvfp4-hw"] = ""
        output.append(out)
    return output


def _load_metric_rows(path: Path, count: int) -> list[dict]:
    with path.open(newline="") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["config"] in {"nvfp4-hw", "e0m3"}
            and int(row["prompt_index"]) < count
        ]
    expected = 2 * count
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} fixed-format rows, got {len(rows)}")
    return rows


def _audit_aggregate(root: Path) -> dict[str, float]:
    with (root / "scale_ablation.csv").open(newline="") as f:
        scale = list(csv.DictReader(f))
    weights = np.asarray([float(row["block_count"]) for row in scale])

    def weighted(column: str) -> float:
        values = np.asarray([float(row[column]) for row in scale])
        return float(np.average(values, weights=weights))

    with (root / "layer_timestep_summary.csv").open(newline="") as f:
        layer_time = list(csv.DictReader(f))
    crest_weights = np.asarray([float(row["block_count"]) for row in layer_time])
    crests = np.asarray([float(row["mean_crest"]) for row in layer_time])
    e2_exact, e0_exact = weighted("e2_exact_sse"), weighted("e0_exact_sse")
    e2_rounded, e0_rounded = weighted("e2_rounded_sse"), weighted("e0_rounded_sse")
    return {
        "exact_scale_block_e0_win_rate": weighted("exact_e0_win_rate"),
        "rounded_scale_block_e0_win_rate": weighted("rounded_e0_win_rate"),
        "e0_exact_sse_reduction_vs_e2": 1.0 - e0_exact / e2_exact,
        "e0_rounded_sse_reduction_vs_e2": 1.0 - e0_rounded / e2_rounded,
        "mean_crest_factor": float(np.average(crests, weights=crest_weights)),
    }


def _cross_model_rows(
    sana_rows: list[dict], pixart_rows: list[dict],
    sana_audit: Path, pixart_audit: Path, bootstrap_samples: int,
) -> list[dict]:
    output = []
    for model_index, (model, rows, audit_root) in enumerate((
        ("pixart-sigma", pixart_rows, pixart_audit),
        ("sana-1.6b", sana_rows, sana_audit),
    )):
        indexed = _index_rows(rows)
        ids = list(indexed["e0m3"])
        audit = _audit_aggregate(audit_root)
        for metric_index, metric in enumerate(METRICS):
            e2 = np.asarray([float(indexed["nvfp4-hw"][i][metric]) for i in ids])
            e0 = np.asarray([float(indexed["e0m3"][i][metric]) for i in ids])
            delta = e0 - e2
            # Reuse the Pilot256 summary seeds for SANA so its primary and
            # cross-model tables report one identical bootstrap interval.
            seed_base = 20260811 if model == "sana-1.6b" else 20260820
            low, high = _bootstrap_ci(
                delta, bootstrap_samples, seed_base + metric_index,
            )
            wins = e0 < e2 if metric == "lpips" else e0 > e2
            output.append({
                "model": model, "n": len(ids), "metric": metric,
                "hw_e2_mean": float(e2.mean()), "hw_e2_median": float(np.median(e2)),
                "fixed_e0_mean": float(e0.mean()), "fixed_e0_median": float(np.median(e0)),
                "fixed_e0_minus_hw_e2_paired_delta": float(delta.mean()),
                "ci95_low": low, "ci95_high": high,
                "fixed_e0_prompt_win_rate": float(wins.mean()), **audit,
            })
    return output


def _grid(path: Path, manifest_path: Path, roots: dict[str, Path], samples) -> None:
    thumb, header, label = 256, 28, 20
    canvas = Image.new(
        "RGB", (thumb * len(CONFIGS), header + (thumb + label) * len(GRID_INDICES)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, config in enumerate(CONFIGS):
        draw.text((column * thumb + 4, 7), config, fill="black")
    manifest = []
    for row_index, index in enumerate(GRID_INDICES):
        image_id, info = samples[index]
        y = header + row_index * (thumb + label)
        draw.text((4, y), f"idx={index} id={image_id}", fill="black")
        for column, config in enumerate(CONFIGS):
            image_path = roots[config] / info["category"] / f"{image_id}.png"
            with Image.open(image_path) as image:
                panel = image.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(panel, (column * thumb, y + label))
        manifest.append({
            "prompt_index": index, "image_id": image_id,
            "category": info["category"], "prompt": info["prompt"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    manifest_path.write_text(json.dumps({
        "selection_rule": "first 8 plus fixed indices 31,63,95,127,159,191,223,255",
        "config_order": list(CONFIGS), "samples": manifest,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pixart-per-prompt", type=Path, required=True)
    parser.add_argument("--sana-audit", type=Path, required=True)
    parser.add_argument("--pixart-audit", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    roots = {config: args.root / config for config in CONFIGS}
    configs = list(roots.items())
    _validate_configs(configs, samples)
    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()
    rows = []
    for config in CONFIGS:
        print(f"Evaluating {config}: {roots[config]}", flush=True)
        rows.extend(_evaluate_config(
            config, roots[config], roots["bf16"], samples,
            args.batch_size, device, lpips, ssim, clip,
        ))

    metrics_dir = args.root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_dir / "pilot256_per_prompt.csv", rows)
    _write_csv(metrics_dir / "pilot256_summary.csv", _summary(rows, args.bootstrap_samples))
    pixart_rows = _load_metric_rows(args.pixart_per_prompt, 128)
    sana_fixed_rows = [row for row in rows if row["config"] != "bf16"]
    _write_csv(metrics_dir / "cross_model_fixed_e0_table.csv", _cross_model_rows(
        sana_fixed_rows, pixart_rows, args.sana_audit, args.pixart_audit,
        args.bootstrap_samples,
    ))
    _grid(
        metrics_dir / "pilot256_comparison_grid.png",
        metrics_dir / "pilot256_comparison_grid_manifest.json",
        roots, samples,
    )
    print(f"Wrote metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
