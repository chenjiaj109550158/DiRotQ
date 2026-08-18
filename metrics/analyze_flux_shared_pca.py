#!/usr/bin/env python3
"""Protected-energy and exact dense-rotation memory audit for FLUX schemes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.flux_shared_pca_basis import (  # noqa: E402
    FluxBasisConfig,
    iter_sources,
    theoretical_basis_bytes,
)
from utils.shared_pca_basis import sha256_file  # noqa: E402


def parse_basis(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("basis must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def covariance(mapping: dict, key: str):
    U = mapping[key].double()
    values = mapping[f"{key}.eigenvalues"].double()
    return (U * values.unsqueeze(0)) @ U.T


def protected_fraction(H: torch.Tensor, U: torch.Tensor, high: int) -> float:
    protected = U[:, -high:]
    return float((protected.T @ H @ protected).diagonal().sum() / H.diagonal().sum())


def summary_families(cfg: FluxBasisConfig) -> tuple[str, ...]:
    return ("ALL", "hidden", "down") if cfg.include_down else ("ALL", "hidden")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-basis", type=Path, required=True)
    parser.add_argument("--basis", action="append", type=parse_basis, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-down", action="store_true")
    args = parser.parse_args()
    cfg = FluxBasisConfig(include_down=not args.exclude_down)
    source = torch.load(args.source_basis, map_location="cpu", weights_only=False)
    candidates = [("per-layer-pca", source)] + [
        (name, torch.load(path, map_location="cpu", weights_only=False))
        for name, path in args.basis
    ]

    rows = []
    for item in iter_sources(cfg):
        H = covariance(source, item.key)
        high = round(
            0.125 * (cfg.intermediate if item.width == "down" else cfg.hidden)
        )
        reference = None
        for name, mapping in candidates:
            value = protected_fraction(H, mapping[item.key].double(), high)
            if reference is None:
                reference = value
            rows.append({
                "scheme": name,
                "kind": item.kind,
                "block": item.block,
                "family": item.family,
                "width": item.width,
                "high_rank": high,
                "protected_energy_fraction": value,
                "relative_to_per_layer": value / reference,
            })
        del H

    summary = []
    for name, _ in candidates:
        subset = [row for row in rows if row["scheme"] == name]
        for family in summary_families(cfg):
            selected = subset if family == "ALL" else [
                row for row in subset if row["width"] == family
            ]
            values = torch.tensor([row["protected_energy_fraction"] for row in selected])
            relative = torch.tensor([row["relative_to_per_layer"] for row in selected])
            memory = theoretical_basis_bytes(
                None if name == "per-layer-pca" else name, cfg=cfg,
            )
            summary.append({
                "scheme": name,
                "family": family,
                "n_sources": len(selected),
                "protected_energy_mean": float(values.mean()),
                "protected_energy_median": float(values.median()),
                "relative_to_per_layer_mean": float(relative.mean()),
                "relative_to_per_layer_min": float(relative.min()),
                "runtime_rotation_bytes_bf16": memory["total_bytes"],
                "runtime_rotation_reduction": (
                    theoretical_basis_bytes(None, cfg=cfg)["total_bytes"] / memory["total_bytes"]
                ),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("per_source.csv", rows), ("summary.csv", summary)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    provenance = {
        "source_basis": str(args.source_basis),
        "source_basis_sha256": sha256_file(args.source_basis),
        "candidate_sha256": {name: sha256_file(path) for name, path in args.basis},
        "memory_scope": "dense online PCA rotations only; excludes weights, activations and allocator overhead",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
