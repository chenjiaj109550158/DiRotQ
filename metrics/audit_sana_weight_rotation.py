#!/usr/bin/env python3
"""Read-only 120-layer SANA random-R versus PCA-only weight audit."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from diffusers import SanaTransformer2DModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.weight_rotation_audit import (
    CREST_EDGES,
    CREST_FINE_BINS,
    E0_VALUES,
    E2_VALUES,
    KURTOSIS_FINE_BINS,
    MAG_HIST_BINS,
    analyze_weight_basis,
    assert_files_unchanged,
    basis_for_layer,
    file_snapshot,
    layer_kind,
    transform_weight_hessian_pair,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def histogram_median(hist: list[int], low: float, high: float) -> float:
    total = sum(hist)
    if not total:
        return 0.0
    target = (total - 1) // 2 + 1
    running = 0
    for index, count in enumerate(hist):
        running += count
        if running >= target:
            return low + (index + .5) * (high-low) / len(hist)
    return high


def add_lists(left: list[int], right: list[int]) -> list[int]:
    return [int(a)+int(b) for a, b in zip(left, right)]


def merge_scale_hist(target: dict[str, int], source: dict[str, int]) -> None:
    for value, count in source.items():
        target[value] = target.get(value, 0) + int(count)


def aggregate_basis(rows: list[dict], details: list[dict]) -> tuple[dict, dict]:
    blocks = sum(int(row["block_count"]) for row in rows)
    nonzero = sum(int(row["nonzero_block_count"]) for row in rows)
    elements = sum(int(row["element_count"]) for row in rows)
    crest_hist = [0] * CREST_FINE_BINS
    kurt_hist = [0] * KURTOSIS_FINE_BINS
    crest_counts = [0] * 7
    crest_exact = [0] * 7
    crest_rounded = [0] * 7
    norm_hist = [0] * MAG_HIST_BINS
    occupancy = {
        "hardware-fixed-e2": [0]*15,
        "hardware-fixed-e0": [0]*15,
    }
    scale_hist = {"hardware-fixed-e2": {}, "hardware-fixed-e0": {}}
    for detail in details:
        crest_hist = add_lists(crest_hist, detail["crest_fine_hist"])
        kurt_hist = add_lists(kurt_hist, detail["kurtosis_fine_hist"])
        crest_counts = add_lists(crest_counts, detail["crest_counts"])
        crest_exact = add_lists(crest_exact, detail["crest_exact_e0_wins"])
        crest_rounded = add_lists(crest_rounded, detail["crest_rounded_e0_wins"])
        norm_hist = add_lists(norm_hist, detail["normalized_magnitude_hist"])
        for fmt in occupancy:
            occupancy[fmt] = add_lists(occupancy[fmt], detail["codebook_occupancy"][fmt])
            merge_scale_hist(scale_hist[fmt], detail["block_scale_histogram"][fmt])

    output = {
        "layer_count": len(rows), "block_count": blocks,
        "nonzero_block_count": nonzero, "element_count": elements,
        "zero_element_count": sum(int(row["zero_element_count"]) for row in rows),
        "zero_rate": sum(int(row["zero_element_count"]) for row in rows) / elements,
        "zero_block_rate": (blocks-nonzero)/blocks,
        "crest_mean": sum(row["crest_mean"]*row["nonzero_block_count"] for row in rows)/nonzero,
        "crest_median_histogram": histogram_median(crest_hist, 1, 4),
        "kurtosis_mean": sum(row["kurtosis_mean"]*row["nonzero_block_count"] for row in rows)/nonzero,
        "kurtosis_median_histogram": histogram_median(kurt_hist, 0, 16),
        "global_scale_min": min(row["global_scale"] for row in rows),
        "global_scale_median": sorted(row["global_scale"] for row in rows)[len(rows)//2],
        "global_scale_max": max(row["global_scale"] for row in rows),
    }
    for label in ("exact", "rounded"):
        win_count = sum(int(row[f"{label}_block_e0_win_count"]) for row in rows)
        output[f"{label}_block_e0_win_rate"] = win_count / blocks
        for kind in ("raw", "hessian"):
            e2 = sum(row[f"{label}_{kind}_e2_loss"] for row in rows)
            e0 = sum(row[f"{label}_{kind}_e0_loss"] for row in rows)
            output[f"{label}_{kind}_e2_loss"] = e2
            output[f"{label}_{kind}_e0_loss"] = e0
            output[f"{label}_{kind}_e0_advantage"] = (e2-e0)/e2
    output["exact_rounded_winner_disagreement_rate"] = sum(
        row["exact_rounded_winner_disagreement_rate"]*row["block_count"] for row in rows
    ) / blocks
    for prefix in ("e2", "e0"):
        output[f"{prefix}_scale_rounding_relative_mean"] = sum(
            row[f"{prefix}_scale_rounding_relative_mean"]*row["nonzero_block_count"]
            for row in rows
        ) / nonzero
        output[f"{prefix}_saturation_rate"] = sum(
            row[f"{prefix}_saturation_rate"]*row["element_count"] for row in rows
        ) / elements
        output[f"{prefix}_block_scale_min"] = min(row[f"{prefix}_scale_min"] for row in rows)
        output[f"{prefix}_block_scale_max"] = max(row[f"{prefix}_scale_max"] for row in rows)

    tile_count = sum(int(row["tile_count"]) for row in rows)
    tile_e0 = sum(int(row["e0_tile_count"]) for row in rows)
    tile_e2_sse = sum(row["e2_sse"] for row in rows)
    tile_e0_sse = sum(row["e0_sse"] for row in rows)
    selected = sum(row["selected_sse"] for row in rows)
    best = min(tile_e2_sse, tile_e0_sse)
    output.update({
        "tile_count": tile_count, "e0_tile_count": tile_e0,
        "e2_tile_count": tile_count-tile_e0, "e0_tile_ratio": tile_e0/tile_count,
        "tile_fixed_e2_sse": tile_e2_sse, "tile_fixed_e0_sse": tile_e0_sse,
        "tile_selected_sse": selected,
        "tilemix_gain_vs_best_fixed": (best-selected)/best,
    })
    histograms = {
        "crest_counts": crest_counts, "crest_exact_wins": crest_exact,
        "crest_rounded_wins": crest_rounded, "normalized_magnitude": norm_hist,
        "occupancy": occupancy, "scale_hist": scale_hist,
    }
    return output, histograms


def family_rows(paired: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in paired:
        groups[(row["layer_family"], row["projection_kind"])].append(row)
    output = []
    fields = (
        "crest_delta", "rounded_block_e0_win_rate_delta",
        "rounded_raw_e0_advantage_delta", "rounded_hessian_e0_advantage_delta",
        "e0_tile_ratio_delta",
    )
    for (family, projection), rows in sorted(groups.items()):
        record = {"layer_family": family, "projection_kind": projection, "layer_count": len(rows)}
        for field in fields:
            values = sorted(float(row[field]) for row in rows)
            record[field+"_mean"] = sum(values)/len(values)
            record[field+"_median"] = values[len(values)//2]
            record[field+"_min"] = values[0]
            record[field+"_max"] = values[-1]
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--hessian-pre", type=Path, required=True)
    parser.add_argument("--hessian-identity-copy", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SANA weight rotation audit requires CUDA; no CPU fallback")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    inputs = [args.config, args.basis, args.rotation, args.hessian_pre, args.hessian_identity_copy]
    before = file_snapshot(inputs)
    if before[str(args.hessian_pre)]["sha256"] != before[str(args.hessian_identity_copy)]["sha256"]:
        raise RuntimeError("random/identity Hessian filenames do not contain identical H_pre")
    cfg = yaml.safe_load(args.config.read_text())
    basis = torch.load(args.basis, map_location="cpu", weights_only=False, mmap=True)
    rotation = torch.load(args.rotation, map_location="cpu", weights_only=False, mmap=True)
    hessians = torch.load(args.hessian_pre, map_location="cpu", weights_only=False, mmap=True)
    if len(hessians) != 120:
        raise RuntimeError(f"expected 120 H_pre layers, found {len(hessians)}")

    transformer_dir = args.model_snapshot / "transformer"
    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"missing local transformer snapshot: {transformer_dir}")
    transformer = SanaTransformer2DModel.from_pretrained(
        transformer_dir, torch_dtype=torch.bfloat16, use_safetensors=True,
        local_files_only=True,
    )
    parameters = dict(transformer.named_parameters())
    layer_rows, details_by_basis = [], {"identity": [], "random": []}
    validations = []
    for index, name in enumerate(sorted(hessians)):
        key = f"{name}.weight"
        if key not in parameters:
            raise RuntimeError(f"original BF16 transformer missing {key}")
        original_weight = parameters[key]
        if original_weight.dtype != torch.bfloat16:
            raise RuntimeError(f"{name}: original weight dtype is {original_weight.dtype}, expected BF16")
        pca, residual, per_head = basis_for_layer(name, basis, rotation)
        high = rotation["high_len_head"] if per_head else rotation["high_len_hidden"]
        pairs, validation = transform_weight_hessian_pair(
            original_weight.to(args.device), hessians[name].to(args.device),
            pca.to(args.device), residual.to(args.device), int(high),
            per_head=per_head,
        )
        tolerance = 2e-4
        if validation["pca_low_orthogonality_max_abs"] > tolerance:
            raise RuntimeError(f"{name}: PCA low basis is not orthogonal: {validation}")
        if validation["residual_orthogonality_max_abs"] > tolerance:
            raise RuntimeError(f"{name}: residual R is not orthogonal: {validation}")
        if validation["weight_frobenius_relative_error"] > tolerance:
            raise RuntimeError(f"{name}: identity/random Frobenius norm mismatch: {validation}")
        if validation["hessian_trace_relative_error"] > tolerance:
            raise RuntimeError(f"{name}: identity/random Hessian trace mismatch: {validation}")
        if validation["unquantized_output_relative_max_error"] > tolerance:
            raise RuntimeError(f"{name}: unquantized basis output mismatch: {validation}")
        for mode in ("identity", "random"):
            sanity = validation["psd_sanity"][mode]
            if sanity["symmetry_relative_max"] > tolerance:
                raise RuntimeError(f"{name}/{mode}: Hessian is not symmetric")
            if sanity["minimum_sampled_quadratic"] < -tolerance * abs(sanity["trace"]):
                raise RuntimeError(f"{name}/{mode}: Hessian sampled PSD sanity failed")
        validations.append({"layer": name, **validation})

        family, projection = layer_kind(name)
        for mode in ("identity", "random"):
            source, hessian = pairs[mode]
            row, detail = analyze_weight_basis(source, hessian)
            row = {
                "basis": mode, "layer": name, "layer_family": family,
                "projection_kind": projection, "per_head": per_head,
                **row,
            }
            layer_rows.append(row)
            details_by_basis[mode].append(detail)
        del pairs, original_weight, pca, residual
        torch.cuda.empty_cache()
        if (index+1) % 10 == 0:
            print(f"Audited {index+1}/120 layers", flush=True)

    by_basis = {
        mode: [row for row in layer_rows if row["basis"] == mode]
        for mode in ("identity", "random")
    }
    aggregates, histograms = {}, {}
    for mode in ("identity", "random"):
        aggregates[mode], histograms[mode] = aggregate_basis(
            by_basis[mode], details_by_basis[mode]
        )

    id_rows = {row["layer"]: row for row in by_basis["identity"]}
    rand_rows = {row["layer"]: row for row in by_basis["random"]}
    paired = []
    for name in sorted(id_rows):
        identity, random = id_rows[name], rand_rows[name]
        paired.append({
            "layer": name, "layer_family": identity["layer_family"],
            "projection_kind": identity["projection_kind"],
            "crest_identity": identity["crest_mean"], "crest_random": random["crest_mean"],
            "crest_delta": random["crest_mean"]-identity["crest_mean"],
            "rounded_block_e0_win_rate_identity": identity["rounded_block_e0_win_rate"],
            "rounded_block_e0_win_rate_random": random["rounded_block_e0_win_rate"],
            "rounded_block_e0_win_rate_delta": random["rounded_block_e0_win_rate"]-identity["rounded_block_e0_win_rate"],
            "rounded_raw_e0_advantage_identity": identity["rounded_raw_e0_advantage"],
            "rounded_raw_e0_advantage_random": random["rounded_raw_e0_advantage"],
            "rounded_raw_e0_advantage_delta": random["rounded_raw_e0_advantage"]-identity["rounded_raw_e0_advantage"],
            "rounded_hessian_e0_advantage_identity": identity["rounded_hessian_e0_advantage"],
            "rounded_hessian_e0_advantage_random": random["rounded_hessian_e0_advantage"],
            "rounded_hessian_e0_advantage_delta": random["rounded_hessian_e0_advantage"]-identity["rounded_hessian_e0_advantage"],
            "e0_tile_ratio_identity": identity["e0_tile_ratio"],
            "e0_tile_ratio_random": random["e0_tile_ratio"],
            "e0_tile_ratio_delta": random["e0_tile_ratio"]-identity["e0_tile_ratio"],
        })
    top20 = sorted(
        paired, key=lambda row: abs(row["rounded_hessian_e0_advantage_delta"]), reverse=True
    )[:20]
    top20 = [{"rank": index+1, **row} for index, row in enumerate(top20)]

    crest_rows, norm_rows, occupancy_rows, scale_rows = [], [], [], []
    for mode in ("identity", "random"):
        hist = histograms[mode]
        for index in range(7):
            count = hist["crest_counts"][index]
            crest_rows.append({
                "basis": mode, "bin_index": index,
                "lower": CREST_EDGES[index], "upper": CREST_EDGES[index+1],
                "count": count,
                "block_mass": count/sum(hist["crest_counts"]),
                "exact_e0_win_rate": hist["crest_exact_wins"][index]/count if count else 0,
                "rounded_e0_win_rate": hist["crest_rounded_wins"][index]/count if count else 0,
            })
        total_norm = sum(hist["normalized_magnitude"])
        for index, count in enumerate(hist["normalized_magnitude"]):
            norm_rows.append({
                "basis": mode, "bin_index": index,
                "lower": index/MAG_HIST_BINS, "upper": (index+1)/MAG_HIST_BINS,
                "count": count, "mass": count/total_norm,
            })
        for fmt, values in (("hardware-fixed-e2", E2_VALUES), ("hardware-fixed-e0", E0_VALUES)):
            counts = hist["occupancy"][fmt]
            total = sum(counts)
            for value, count in zip(values, counts):
                occupancy_rows.append({
                    "basis": mode, "format": fmt, "level": value,
                    "count": count, "ratio": count/total,
                })
            for scale, count in sorted(hist["scale_hist"][fmt].items(), key=lambda item: float(item[0])):
                scale_rows.append({
                    "basis": mode, "format": fmt, "e4m3_scale": scale,
                    "count": count,
                })

    difference = {
        "D_raw": aggregates["random"]["rounded_raw_e0_advantage"] - aggregates["identity"]["rounded_raw_e0_advantage"],
        "D_H": aggregates["random"]["rounded_hessian_e0_advantage"] - aggregates["identity"]["rounded_hessian_e0_advantage"],
        "D_raw_exact_scale": aggregates["random"]["exact_raw_e0_advantage"] - aggregates["identity"]["exact_raw_e0_advantage"],
        "D_H_exact_scale": aggregates["random"]["exact_hessian_e0_advantage"] - aggregates["identity"]["exact_hessian_e0_advantage"],
        "block_e0_win_rate_delta": aggregates["random"]["rounded_block_e0_win_rate"] - aggregates["identity"]["rounded_block_e0_win_rate"],
        "tile_e0_ratio_delta": aggregates["random"]["e0_tile_ratio"] - aggregates["identity"]["e0_tile_ratio"],
        "layers_raw_advantage_increased": sum(row["rounded_raw_e0_advantage_delta"] > 0 for row in paired)/len(paired),
        "layers_hessian_advantage_increased": sum(row["rounded_hessian_e0_advantage_delta"] > 0 for row in paired)/len(paired),
        "layers_block_win_rate_increased": sum(row["rounded_block_e0_win_rate_delta"] > 0 for row in paired)/len(paired),
        "layers_tile_ratio_increased": sum(row["e0_tile_ratio_delta"] > 0 for row in paired)/len(paired),
    }

    assert_files_unchanged(before)
    elapsed = time.perf_counter()-started
    provenance = {
        "command": sys.argv, "model": cfg["model"], "model_id": cfg["model_id"],
        "model_snapshot": str(args.model_snapshot.resolve()),
        "model_revision": args.model_snapshot.name,
        "model_dtype": "bfloat16", "active_layers": 120,
        "basis_comparison": {
            "identity": "PCA low subspace U_l; residual R is identity",
            "random": "same PCA low subspace followed by cached random R",
            "high_branch_excluded": True,
        },
        "hessian_semantics": "full-precision pre-wrapper H_pre=2/n X^T X; not E0 H_Z",
        "input_files_before_and_after_identical": True,
        "inputs": before, "runtime_seconds": elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/1024**3,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "validation_maxima": {
            key: max(row[key] for row in validations)
            for key in (
                "pca_low_orthogonality_max_abs", "residual_orthogonality_max_abs",
                "projection_presymmetry_identity_relative_max",
                "projection_presymmetry_random_relative_max",
                "weight_frobenius_relative_error",
                "hessian_trace_relative_error", "unquantized_output_relative_max_error",
            )
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/"basis_distribution_summary.csv", [
        {"basis": mode, **aggregates[mode]} for mode in ("identity", "random")
    ])
    write_csv(args.output_dir/"crest_bin_summary.csv", crest_rows)
    write_csv(args.output_dir/"normalized_magnitude_histogram.csv", norm_rows)
    write_csv(args.output_dir/"codebook_occupancy.csv", occupancy_rows)
    write_csv(args.output_dir/"e4m3_block_scale_histogram.csv", scale_rows)
    write_csv(args.output_dir/"layer_basis_summary.csv", layer_rows)
    write_csv(args.output_dir/"layer_paired_summary.csv", paired)
    write_csv(args.output_dir/"layer_family_summary.csv", family_rows(paired))
    write_csv(args.output_dir/"largest_20_layer_differences.csv", top20)
    (args.output_dir/"mechanism_summary.json").write_text(json.dumps({
        "aggregates": aggregates, "random_minus_identity": difference,
    }, indent=2)+"\n")
    (args.output_dir/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"random_minus_identity": difference, **provenance}, indent=2))


if __name__ == "__main__":
    main()
