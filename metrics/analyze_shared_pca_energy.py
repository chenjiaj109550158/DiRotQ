#!/usr/bin/env python3
"""Read-only PCA protected-energy audit for derived PixArt shared bases."""

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

from utils.shared_pca_basis import (  # noqa: E402
    FAMILIES, HEAD_FAMILIES, PixArtBasisConfig, basis_key, hessian_key,
    sha256_file,
)


def _head_diagonal(H, cfg):
    H4 = H.reshape(cfg.num_heads, cfg.head_dim, cfg.num_heads, cfg.head_dim)
    idx = torch.arange(cfg.num_heads)
    return H4[idx, :, idx, :]


def _basis_source_covariance(source, block, family):
    key = basis_key(block, family)
    U = source[key].double()
    evals = source[f"{key}.eigenvalues"].double()
    return U @ torch.diag_embed(evals) @ U.transpose(-1, -2)


def _protected_fraction(H, U, high):
    if H.ndim == 2:
        projected = U[:, -high:].T @ H @ U[:, -high:]
        return float(projected.diagonal().sum() / H.diagonal().sum())
    values = []
    masses = []
    for head in range(H.shape[0]):
        projected = U[head, :, -high:].T @ H[head] @ U[head, :, -high:]
        values.append(projected.diagonal().sum())
        masses.append(H[head].diagonal().sum())
    return float(torch.stack(values).sum() / torch.stack(masses).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-basis", type=Path, default=ROOT / "models/pixart-sigma/basis/U-pixart-sigma.pt")
    parser.add_argument("--hessians", type=Path, default=ROOT / "models/pixart-sigma/quantized_cache/hessians_n5120_l224.pt")
    parser.add_argument("--basis", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = PixArtBasisConfig()
    source = torch.load(args.source_basis, map_location="cpu", weights_only=False)
    hessians = torch.load(args.hessians, map_location="cpu", weights_only=False)
    candidates = [("per-layer-pca", source)]
    for path in args.basis:
        data = torch.load(path, map_location="cpu", weights_only=False)
        candidates.append((data["__shared_basis_scheme__"], data))

    rows = []
    for block in range(cfg.num_layers):
        for family in FAMILIES:
            if family in HEAD_FAMILIES:
                # Match basis_utils.py: value PCA observes to_v outputs per
                # head, whereas the GPTQ Hessian observes the later to_out
                # input.  The eigensystem is the available exact provenance
                # for the former and must be used for this comparison.
                H = _basis_source_covariance(source, block, family)
                high = round(cfg.head_dim * 0.125)
            else:
                H = hessians[hessian_key(block, family)].double()
            if family == "ffn.down_proj":
                high = round(cfg.intermediate * 0.125)
            elif family not in HEAD_FAMILIES:
                high = round(cfg.hidden * 0.125)
            reference = None
            for name, basis in candidates:
                U = basis[basis_key(block, family)].double()
                fraction = _protected_fraction(H, U, high)
                if name == "per-layer-pca":
                    reference = fraction
                rows.append({
                    "scheme": name, "block": block, "family": family,
                    "high_rank": high, "protected_energy_fraction": fraction,
                    "relative_to_per_layer": 1.0 if reference is None else fraction / reference,
                })
            del H

    # The first item for each block/family is always the per-layer reference;
    # fill ratios now that every reference is known.
    refs = {(row["block"], row["family"]): row["protected_energy_fraction"]
            for row in rows if row["scheme"] == "per-layer-pca"}
    for row in rows:
        row["relative_to_per_layer"] = row["protected_energy_fraction"] / refs[(row["block"], row["family"])]

    summary = []
    for name, _ in candidates:
        selected = [row for row in rows if row["scheme"] == name]
        aggregates = {
            "ALL_EQUAL_SOURCE": selected,
            # Official PixArt W4A4 quality commands preserve ff.net.2 in
            # high precision, so its basis is not part of the active method.
            "ALL_ACTIVE_EQUAL_SOURCE": [
                row for row in selected if row["family"] != "ffn.down_proj"
            ],
        }
        for family in (*FAMILIES, *aggregates):
            subset = aggregates[family] if family in aggregates else [
                row for row in selected if row["family"] == family
            ]
            values = torch.tensor([row["protected_energy_fraction"] for row in subset])
            ratios = torch.tensor([row["relative_to_per_layer"] for row in subset])
            summary.append({
                "scheme": name, "family": family, "n_sources": len(subset),
                "protected_energy_mean": float(values.mean()),
                "protected_energy_median": float(values.median()),
                "relative_to_per_layer_mean": float(ratios.mean()),
                "relative_to_per_layer_min": float(ratios.min()),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("per_source.csv", rows), ("summary.csv", summary)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    provenance = {
        "source_basis": str(args.source_basis),
        "source_basis_sha256": sha256_file(args.source_basis),
        "hessians": str(args.hessians),
        "hessians_sha256": sha256_file(args.hessians),
        "candidate_sha256": {str(path): sha256_file(path) for path in args.basis},
        "interpretation": "equal-source PCA protected-energy diagnostic; not image quality",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
