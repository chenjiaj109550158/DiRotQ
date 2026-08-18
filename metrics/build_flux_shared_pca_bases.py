#!/usr/bin/env python3
"""Build the four frozen FLUX shared-PCA basis artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.flux_shared_pca_basis import (  # noqa: E402
    SCHEMES,
    build_flux_shared_basis,
    validate_flux_shared_basis,
)
from utils.shared_pca_basis import sha256_file, write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("flux-dev", "flux-schnell"), default="flux-dev")
    parser.add_argument("--source-basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheme", choices=(*SCHEMES, "all"), default="all")
    args = parser.parse_args()

    source_sha = sha256_file(args.source_basis)
    source = torch.load(args.source_basis, map_location="cpu", weights_only=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    schemes = SCHEMES if args.scheme == "all" else (args.scheme,)
    for scheme in schemes:
        derived, manifest = build_flux_shared_basis(
            source, scheme, source_basis_sha256=source_sha,
        )
        validation = validate_flux_shared_basis(derived)
        output = args.output_dir / f"U-{args.model}-{scheme}.pt"
        temporary = output.with_suffix(".pt.tmp")
        torch.save(derived, temporary)
        temporary.replace(output)
        manifest.update(validation)
        manifest["model"] = args.model
        manifest.update({
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "artifact_size": output.stat().st_size,
        })
        write_manifest(output.with_suffix(".manifest.json"), manifest)
        summary[scheme] = manifest
        write_manifest(summary_path, summary)
        print(json.dumps({"scheme": scheme, "path": str(output), **validation}))
        del derived


if __name__ == "__main__":
    main()
