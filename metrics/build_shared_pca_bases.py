#!/usr/bin/env python3
"""Build one or all pre-registered PixArt shared-basis artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.shared_pca_basis import (  # noqa: E402
    SCHEMES,
    build_shared_basis,
    sha256_file,
    validate_shared_basis,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-basis", type=Path, default=ROOT / "models/pixart-sigma/basis/U-pixart-sigma.pt")
    parser.add_argument("--hessians", type=Path, default=ROOT / "models/pixart-sigma/quantized_cache/hessians_n5120_l224.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/pixart-sigma/shared_pca_basis_audit/bases")
    parser.add_argument("--scheme", choices=(*SCHEMES, "all"), default="all")
    args = parser.parse_args()

    basis_sha = sha256_file(args.source_basis)
    hessian_sha = sha256_file(args.hessians)
    source = torch.load(args.source_basis, map_location="cpu", weights_only=False)
    hessians = torch.load(args.hessians, map_location="cpu", weights_only=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schemes = SCHEMES if args.scheme == "all" else (args.scheme,)
    summary_path = args.output_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            summary = {}
    else:
        summary = {}
    for scheme in schemes:
        derived, manifest = build_shared_basis(
            source, hessians, scheme,
            source_basis_sha256=basis_sha,
            hessian_sha256=hessian_sha,
        )
        validation = validate_shared_basis(derived)
        output = args.output_dir / f"U-pixart-sigma-{scheme}.pt"
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(derived, temporary)
        temporary.replace(output)
        manifest.update(validation)
        manifest.update({
            "artifact": str(output.relative_to(ROOT)),
            "artifact_sha256": sha256_file(output),
            "artifact_size": output.stat().st_size,
        })
        manifest_path = output.with_suffix(".manifest.json")
        write_manifest(manifest_path, manifest)
        summary[scheme] = manifest
        print(json.dumps({"scheme": scheme, **validation, "path": str(output)}))
        del derived
    write_manifest(summary_path, summary)


if __name__ == "__main__":
    main()
