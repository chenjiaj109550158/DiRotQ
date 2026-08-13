#!/usr/bin/env python3
"""Matched SANA Pilot32 evaluation for activation TileMix x fixed E0 weight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPScore

try:
    from metrics.evaluate_pilot128 import (
        _bootstrap_ci,
        _evaluate_config,
        _load_samples,
        _validate_configs,
        _write_csv,
    )
except ModuleNotFoundError:  # direct ``python metrics/<script>.py`` execution
    from evaluate_pilot128 import (
        _bootstrap_ci,
        _evaluate_config,
        _load_samples,
        _validate_configs,
        _write_csv,
    )


C0 = "fixed-e0a-fixed-e0w"
C1 = "fixed-e2a-fixed-e0w"
C2 = "tilemix-a-fixed-e0w"
METRICS = ("psnr", "lpips", "ssim", "clip_score")
COMPARISONS = ((C2, C0), (C2, C1), (C0, C1))
GRID_INDICES = (*range(8), 15, 23, 31)
TAIL_ADVERSE_RATE_LIMIT = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_array(by_config, config: str, ids: list[str], metric: str) -> np.ndarray:
    return np.asarray(
        [float(by_config[config][image_id][metric]) for image_id in ids],
        dtype=np.float64,
    )


def _trimmed_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    cut = int(math.floor(len(values) * fraction))
    ordered = np.sort(values)
    kept = ordered[cut:len(values) - cut] if cut else ordered
    return float(kept.mean())


def classify_result(paired: dict, tail: dict, e0_ratio: float) -> tuple[str, dict]:
    """Apply the preregistered asymmetric Pilot32 decision rules."""
    c2_c0 = paired[f"{C2}_minus_{C0}"]
    psnr = c2_c0["psnr"]
    lpips = c2_c0["lpips"]
    clip = c2_c0["clip_score"]
    mean_favorable = psnr["mean_delta"] > 0 and lpips["mean_delta"] < 0
    trimmed_favorable = (
        psnr["trimmed_mean_10pct"] > 0 and lpips["trimmed_mean_10pct"] < 0
    )
    reaches_effect = psnr["mean_delta"] >= 0.15 or lpips["mean_delta"] <= -0.003
    clip_significantly_degrades = clip["ci95"][1] < 0
    tail_worsened = (
        tail["psnr_delta_below_minus_0p5_ratio"] > TAIL_ADVERSE_RATE_LIMIT
        or tail["lpips_delta_above_plus_0p01_ratio"] > TAIL_ADVERSE_RATE_LIMIT
    )
    nondegenerate = e0_ratio <= 0.95 and (1.0 - e0_ratio) <= 0.95
    criteria = {
        "mean_psnr_lpips_favorable": mean_favorable,
        "trimmed_psnr_lpips_favorable": trimmed_favorable,
        "minimum_effect_reached": reaches_effect,
        "clip_significantly_degrades": clip_significantly_degrades,
        "tail_adverse_rate_limit": TAIL_ADVERSE_RATE_LIMIT,
        "tail_worsened": tail_worsened,
        "format_distribution_nondegenerate": nondegenerate,
    }
    if (mean_favorable and trimmed_favorable and reaches_effect
            and not clip_significantly_degrades and not tail_worsened and nondegenerate):
        return "ASYMMETRIC MIX PROMISING", criteria
    robustly_adverse = (
        psnr["mean_delta"] < 0 and lpips["mean_delta"] > 0
        and psnr["trimmed_mean_10pct"] < 0
        and lpips["trimmed_mean_10pct"] > 0
    )
    if robustly_adverse or clip_significantly_degrades or tail_worsened:
        return "ASYMMETRIC MIX REGRESSES", criteria
    psnr_ci = psnr["ci95"]
    lpips_ci = lpips["ci95"]
    effects_small = abs(psnr["mean_delta"]) < 0.15 and abs(lpips["mean_delta"]) < 0.003
    cis_cross_zero = psnr_ci[0] <= 0 <= psnr_ci[1] and lpips_ci[0] <= 0 <= lpips_ci[1]
    if effects_small and cis_cross_zero:
        return "ASYMMETRIC MIX NEUTRAL", criteria
    return "INCONCLUSIVE", criteria


def _integrity_manifest(samples, directories: dict[str, Path]) -> list[dict]:
    _validate_configs(list(directories.items()), samples)
    rows = []
    for config, root in directories.items():
        for index, (image_id, info) in enumerate(samples):
            path = root / info["category"] / f"{image_id}.png"
            with Image.open(path) as image:
                image.load()
                extrema = image.getextrema()
                if image.mode != "RGB" or image.size != (1024, 1024):
                    raise RuntimeError(f"{config}/{image_id}: invalid image shape/mode")
                if all(low == high for low, high in extrema):
                    raise RuntimeError(f"{config}/{image_id}: flat image")
            rows.append({
                "prompt_index": index,
                "image_id": image_id,
                "category": info["category"],
                "config": config,
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "mode": "RGB",
                "width": 1024,
                "height": 1024,
                "channel_extrema": repr(extrema),
                "flat": False,
            })
    return rows


def _build_grid(output: Path, samples, reference_dir: Path, directories: dict[str, Path]):
    columns = [("BF16", reference_dir), ("fixed E2 A x E0 W", directories[C1]),
               ("fixed E0 A x E0 W", directories[C0]),
               ("TileMix A x E0 W", directories[C2])]
    thumb, header, label = 256, 30, 20
    canvas = Image.new(
        "RGB",
        (thumb * len(columns), header + (thumb + label) * len(GRID_INDICES)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, (name, _) in enumerate(columns):
        draw.text((column * thumb + 4, 8), name, fill="black")
    manifest = []
    for row, index in enumerate(GRID_INDICES):
        image_id, info = samples[index]
        y = header + row * (thumb + label)
        draw.text((4, y + 2), f"idx={index} id={image_id}", fill="black")
        for column, (_, root) in enumerate(columns):
            path = root / info["category"] / f"{image_id}.png"
            with Image.open(path) as image:
                panel = image.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(panel, (column * thumb, y + label))
        manifest.append({
            "prompt_index": index,
            "image_id": image_id,
            "category": info["category"],
            "prompt": info["prompt"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    output.with_suffix(".json").write_text(
        json.dumps({"columns": [name for name, _ in columns], "samples": manifest}, indent=2)
        + "\n"
    )


def _selector_correlations(stats: dict, delta_by_metric: dict[str, np.ndarray], ids: list[str]):
    per_prompt = stats.get("per_prompt", {})
    if set(per_prompt) != set(ids):
        raise RuntimeError(
            "TileMix stats prompt IDs do not exactly match Pilot32: "
            f"missing={sorted(set(ids) - set(per_prompt))}, "
            f"extra={sorted(set(per_prompt) - set(ids))}"
        )
    features = {
        "e0m3_ratio": np.asarray([per_prompt[i]["e0m3_ratio"] for i in ids]),
        "sse_gain_vs_fixed_e0": np.asarray(
            [per_prompt[i]["selected_reduction_vs_e0"] for i in ids]
        ),
    }
    rows = []
    for feature_name, feature in features.items():
        for metric, delta in delta_by_metric.items():
            result = spearmanr(feature, delta)
            rows.append({
                "feature": feature_name,
                "quality_delta": f"C2_minus_C0_{metric}",
                "n": len(ids),
                "spearman_rho": float(result.statistic),
                "p_value_exploratory": float(result.pvalue),
                "causal_claim": False,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--c1-dir", type=Path, required=True)
    parser.add_argument("--c2-dir", type=Path, required=True)
    parser.add_argument("--c2-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = _load_samples(args.dataset, args.count)
    ids = [image_id for image_id, _ in samples]
    directories = {C0: args.c0_dir, C1: args.c1_dir, C2: args.c2_dir}
    integrity = _integrity_manifest(samples, directories)
    # The reused reference directory contains Pilot64.  Require every matched
    # first32 image but do not misclassify its untouched indices 32--63 as an
    # error or copy them into this Pilot32.
    for image_id, info in samples:
        path = args.reference_dir / info["category"] / f"{image_id}.png"
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1024, 1024):
                raise RuntimeError(f"BF16 reference {image_id}: invalid image")

    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        data_range=(0.0, 1.0), reduction="none"
    ).to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()

    all_rows = []
    by_config = {}
    for name, path in directories.items():
        print(f"Evaluating {name}: {path}", flush=True)
        rows = _evaluate_config(
            name, path, args.reference_dir, samples, args.batch_size,
            device, lpips, ssim, clip,
        )
        all_rows.extend(rows)
        by_config[name] = {row["image_id"]: row for row in rows}

    summary_rows = []
    config_summary = {}
    for config in (C0, C1, C2):
        config_summary[config] = {}
        for metric in METRICS:
            values = _metric_array(by_config, config, ids, metric)
            record = {"mean": float(values.mean()), "median": float(np.median(values))}
            config_summary[config][metric] = record
            summary_rows.append({
                "row_type": "config", "candidate": config, "baseline": "BF16",
                "metric": metric, "n": len(ids), **record,
                "paired_mean_delta": "", "paired_median_delta": "", "win_rate": "",
                "ci95_low": "", "ci95_high": "", "p25_delta": "",
                "p50_delta": "", "p75_delta": "", "trimmed_mean_10pct": "",
            })

    paired = {}
    delta_c2_c0 = {}
    for comparison_index, (candidate, baseline) in enumerate(COMPARISONS):
        key = f"{candidate}_minus_{baseline}"
        paired[key] = {}
        for metric_index, metric in enumerate(METRICS):
            candidate_values = _metric_array(by_config, candidate, ids, metric)
            baseline_values = _metric_array(by_config, baseline, ids, metric)
            delta = candidate_values - baseline_values
            low, high = _bootstrap_ci(
                delta, args.bootstrap_samples,
                seed=20260813 + comparison_index * 97 + metric_index,
            )
            lower_is_better = metric == "lpips"
            win_rate = float(np.mean(delta < 0 if lower_is_better else delta > 0))
            quantiles = np.quantile(delta, (0.25, 0.50, 0.75))
            record = {
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "win_rate": win_rate,
                "ci95": [low, high],
                "p25_delta": float(quantiles[0]),
                "p50_delta": float(quantiles[1]),
                "p75_delta": float(quantiles[2]),
                "trimmed_mean_10pct": _trimmed_mean(delta),
            }
            paired[key][metric] = record
            if candidate == C2 and baseline == C0:
                delta_c2_c0[metric] = delta
            summary_rows.append({
                "row_type": "paired", "candidate": candidate, "baseline": baseline,
                "metric": metric, "n": len(ids), "mean": "", "median": "",
                "paired_mean_delta": record["mean_delta"],
                "paired_median_delta": record["median_delta"],
                "win_rate": win_rate, "ci95_low": low, "ci95_high": high,
                "p25_delta": record["p25_delta"], "p50_delta": record["p50_delta"],
                "p75_delta": record["p75_delta"],
                "trimmed_mean_10pct": record["trimmed_mean_10pct"],
            })

    psnr_delta = delta_c2_c0["psnr"]
    lpips_delta = delta_c2_c0["lpips"]
    tail = {
        "psnr_delta_below_minus_0p5_count": int((psnr_delta < -0.5).sum()),
        "psnr_delta_below_minus_0p5_ratio": float((psnr_delta < -0.5).mean()),
        "lpips_delta_above_plus_0p01_count": int((lpips_delta > 0.01).sum()),
        "lpips_delta_above_plus_0p01_ratio": float((lpips_delta > 0.01).mean()),
        "worst8_psnr_delta_mean": float(np.sort(psnr_delta)[:8].mean()),
        "worst8_lpips_delta_mean": float(np.sort(lpips_delta)[-8:].mean()),
    }

    stats = json.loads(args.c2_stats.read_text())
    correlations = _selector_correlations(stats, delta_c2_c0, ids)
    classification, classification_criteria = classify_result(
        paired, tail, float(stats["e0m3_ratio"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "asymmetric_tilemix_pilot32_per_prompt.csv", all_rows)
    _write_csv(args.output_dir / "asymmetric_tilemix_pilot32_summary.csv", summary_rows)
    _write_csv(args.output_dir / "asymmetric_tilemix_pilot32_image_manifest.csv", integrity)
    _write_csv(args.output_dir / "asymmetric_tilemix_pilot32_correlations.csv", correlations)
    _build_grid(
        args.output_dir / "asymmetric_tilemix_pilot32_grid.png",
        samples, args.reference_dir, directories,
    )
    result = {
        "n": len(ids),
        "bootstrap_samples": args.bootstrap_samples,
        "classification": classification,
        "classification_criteria": classification_criteria,
        "config_summary": config_summary,
        "paired_comparisons": paired,
        "c2_minus_c0_tail_diagnostics": tail,
        "selector_global": {
            key: stats[key] for key in (
                "e0m3_count", "e2m1_count", "total_count", "e0m3_ratio",
                "flip_count", "flip_total", "flip_rate", "all_e0_sse",
                "all_e2_sse", "selected_sse", "selected_reduction_vs_e0",
                "selected_reduction_vs_e2", "reconstruction_sse", "qsnr_db",
            )
        },
        "selector_per_layer": stats["per_layer"],
        "selector_per_timestep": stats["per_timestep"],
        "selector_quality_correlations": correlations,
        "fixed_grid_indices": list(GRID_INDICES),
        "notes": [
            "CLIP Score is prompt alignment, not a BF16-paired image similarity metric.",
            "Spearman correlations are exploratory associations and are not causal claims.",
            "Pure-PyTorch fake-quant runtime is not a packed-kernel latency estimate.",
        ],
    }
    (args.output_dir / "asymmetric_tilemix_pilot32_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
