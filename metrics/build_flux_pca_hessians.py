#!/usr/bin/env python3
"""Reconstruct FLUX pre-rotation Hessians from the PCA eigensystem.

The FLUX PCA collector stores eigenpairs of ``C + damping*mean(diag(C))*I``.
This tool analytically removes that known isotropic term, multiplies by two to
match ``collect_hessians``' convention, and aliases Q/K/V entries that consume
the same activation.  FFN-down is intentionally absent for the frozen
no-rotation/configured-RTN speed-path contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct(mapping: dict, key: str, damping: float) -> torch.Tensor:
    U = mapping[key].double()
    eigenvalues = mapping[f"{key}.eigenvalues"].double()
    raw_mean = eigenvalues.mean() / (1.0 + damping)
    raw_values = eigenvalues - damping * raw_mean
    covariance = (U * raw_values.unsqueeze(0)) @ U.T
    covariance = (covariance + covariance.T) * 0.5
    return (2.0 * covariance).float().contiguous()


def name_groups():
    for block in range(19):
        prefix = f"transformer_blocks.{block}"
        yield f"layer.{block}.img_attn", [
            f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v",
        ]
        yield f"layer.{block}.txt_attn", [
            f"{prefix}.attn.add_q_proj", f"{prefix}.attn.add_k_proj",
            f"{prefix}.attn.add_v_proj",
        ]
        yield f"layer.{block}.img_attn.value", [f"{prefix}.attn.to_out.0"]
        yield f"layer.{block}.txt_attn.value", [f"{prefix}.attn.to_add_out"]
        yield f"layer.{block}.img_ffn", [f"{prefix}.ff.net.0.proj"]
        yield f"layer.{block}.txt_ffn", [f"{prefix}.ff_context.net.0.proj"]
    for block in range(38):
        prefix = f"single_transformer_blocks.{block}"
        yield f"single.{block}.attn", [
            f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v",
        ]
        yield f"single.{block}.mlp", [f"{prefix}.proj_mlp"]
        yield f"single.{block}.attn_out.value", [f"{prefix}.proj_out.linears.0"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--damping", type=float, default=0.01)
    args = parser.parse_args()
    basis = torch.load(args.basis, map_location="cpu", weights_only=False)
    hessians = {}
    source_count = 0
    for key, names in name_groups():
        H = reconstruct(basis, key, args.damping)
        source_count += 1
        for name in names:
            hessians[name] = H
    if source_count != 228 or len(hessians) != 380:
        raise RuntimeError(
            f"unexpected FLUX Hessian coverage: sources={source_count}, layers={len(hessians)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(hessians, temporary)
    temporary.replace(args.output)
    manifest = {
        "schema": "dirotq.flux_pca_reconstructed_hessians",
        "version": 1,
        "basis": str(args.basis),
        "basis_sha256": sha256_file(args.basis),
        "damping_removed": args.damping,
        "normalization": "2 * mean(X^T X)",
        "unique_sources": source_count,
        "gptq_layers": len(hessians),
        "configured_rtn_layers": 76,
        "configured_rtn_patterns": [".net.2", "proj_out.linears.1"],
        "artifact_sha256": sha256_file(args.output),
        "artifact_size": args.output.stat().st_size,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
