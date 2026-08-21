#!/usr/bin/env python3
"""Create the immutable rank-64 residual rotation for FLUX Scheme A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import torch

from utils.flux_scheme_a import (
    build_hidden_residual_rotation,
    validate_hidden_residual_rotation,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=3072)
    parser.add_argument("--high-rank", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--intermediate-dim", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-source-seed", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite Scheme A rotation: {args.output}")

    source = torch.load(args.source_rotation, map_location="cpu", weights_only=False)
    source_high = int(source["high_len_hidden"])
    source_report = validate_hidden_residual_rotation(source["R1"], source_high)
    seed_match = None
    seed_match_max_abs = None
    if args.verify_source_seed:
        regenerated = build_hidden_residual_rotation(
            args.hidden_dim, source_high, seed=args.seed, device=args.device
        )
        seed_match_max_abs = float((regenerated - source["R1"]).abs().max())
        seed_match = seed_match_max_abs <= 2e-10
        if not seed_match:
            raise RuntimeError(
                "production source R1 does not match the declared algorithm/seed: "
                f"max_abs={seed_match_max_abs:.8g}"
            )
        del regenerated

    rotation = build_hidden_residual_rotation(
        args.hidden_dim, args.high_rank, seed=args.seed, device=args.device
    )
    report = validate_hidden_residual_rotation(rotation, args.high_rank)
    fraction = args.high_rank / args.hidden_dim
    payload = {
        "R1": rotation,
        # FLUX speed-compatible shared-width has no down-projection basis and
        # routes attention output through the flat hidden frame. R2/R_down are
        # deliberately absent rather than allocating unused matrices.
        "high_len_hidden": args.high_rank,
        "high_len_head": round(fraction * args.head_dim),
        "high_len_down": round(fraction * args.intermediate_dim),
        "high_fraction": fraction,
        "scheme_a_metadata": {
            "algorithm": "torch-float64-gaussian-qr-sign-diag-blockdiag-identity-tail",
            "seed": args.seed,
            "source_rotation_sha256": sha256_file(args.source_rotation),
            "source_high_rank": source_high,
            "source_seed_verified": seed_match,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    manifest = {
        "schema": "dirotq.flux_shared_width_scheme_a_rotation",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.output),
        "artifact_sha256": sha256_file(args.output),
        "artifact_size": args.output.stat().st_size,
        "source_rotation": str(args.source_rotation),
        "source_rotation_sha256": sha256_file(args.source_rotation),
        "algorithm": payload["scheme_a_metadata"]["algorithm"],
        "seed": args.seed,
        "source_seed_match_max_abs": seed_match_max_abs,
        "source_validation": source_report,
        "scheme_a_validation": report,
        "omitted_unused_matrices": ["R2", "R_down"],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
