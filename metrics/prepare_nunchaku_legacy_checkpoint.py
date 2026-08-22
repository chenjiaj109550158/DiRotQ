#!/usr/bin/env python3
"""Losslessly split a monolithic Nunchaku checkpoint for legacy loaders.

Nunchaku 0.1.x expects ``unquantized_layers.safetensors`` and
``transformer_blocks.safetensors``.  Current official checkpoints contain the
same tensors in one safetensors file.  This utility only separates those
stored tensors; it does not decode, requantize, or otherwise alter them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    required = {"config", "quantization_config", "model_class"}
    if missing := required - set(metadata):
        raise RuntimeError(f"checkpoint metadata missing {sorted(missing)}")
    if metadata["model_class"] != "NunchakuFluxTransformer2dModel":
        raise RuntimeError(f"unexpected model class {metadata['model_class']!r}")

    state = load_file(source, device="cpu")
    quantized = {
        key: value
        for key, value in state.items()
        if key.startswith(("transformer_blocks.", "single_transformer_blocks."))
    }
    unquantized = {key: value for key, value in state.items() if key not in quantized}
    if not quantized or not unquantized:
        raise RuntimeError("invalid split: one side is empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        save_file(quantized, temporary / "transformer_blocks.safetensors")
        save_file(unquantized, temporary / "unquantized_layers.safetensors")
        (temporary / "config.json").write_text(metadata["config"])
        manifest = {
            "schema": "nunchaku.lossless_legacy_split",
            "source": str(source),
            "source_sha256": sha256_file(source),
            "source_size": source.stat().st_size,
            "model_class": metadata["model_class"],
            "quantization_config": json.loads(metadata["quantization_config"]),
            "tensor_counts": {
                "all": len(state),
                "quantized": len(quantized),
                "unquantized": len(unquantized),
            },
        }
        for name in (
            "transformer_blocks.safetensors",
            "unquantized_layers.safetensors",
            "config.json",
        ):
            path = temporary / name
            manifest[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        (temporary / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
