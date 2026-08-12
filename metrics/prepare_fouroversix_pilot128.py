#!/usr/bin/env python3
"""Validate Pilot128 provenance and link the exact Four Over Six Pilot32 prefix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/mjhq_5000_samples.json"
CHECKPOINT = "2357540a82f2483e249e78424ae164397475c799"
MODEL_SPECS = {
    "pixart-sigma": {
        "model_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "revision": "e102b3591cc82e97071b8b4cb90d834d0c487207",
        "dtype": "fp16",
        "cache": "models/pixart-sigma/quantized_cache/nvfp4_g16_gptq_skipc27eed7e_model.pt",
        "pilot32": "models/pixart-sigma/fouroversix_pilot32/pilot32",
        "pilot32_logs": "models/pixart-sigma/fouroversix_pilot32/logs",
        "target": "models/pixart-sigma/fouroversix_pilot128",
        "baselines": {
            "reference": "models/pixart-sigma/pilot128/fp16",
            "nvfp4-hw": "models/pixart-sigma/pilot128/nvfp4-hw",
            "e0m3": "models/pixart-sigma/pilot128/e0m3",
            "tile-mix-oracle": "models/pixart-sigma/pilot128/tile-mix-oracle",
            "a16w4-residual": "models/pixart-sigma/a16w4_ceiling64/pilot64",
        },
        "baseline_logs": {
            "reference": "models/pixart-sigma/pilot128/logs/fp16.log",
            "nvfp4-hw": "models/pixart-sigma/pilot128/logs/nvfp4-hw.log",
            "e0m3": "models/pixart-sigma/pilot128/logs/e0m3.log",
            "tile-mix-oracle": "models/pixart-sigma/pilot128/logs/tile-mix-oracle.log",
        },
        "skip": "--skip-quant-layers ff.net.2",
    },
    "sana-1.6b": {
        "model_id": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "revision": "e2b3c0cbffebcd09d83805e88b9f5f106afc74ac",
        "dtype": "bf16",
        "cache": "models/sana-1.6b/quantized_cache/nvfp4_g16_gptq_model.pt",
        "pilot32": "models/sana-1.6b/fouroversix_pilot32/pilot32",
        "pilot32_logs": "models/sana-1.6b/fouroversix_pilot32/logs",
        "target": "models/sana-1.6b/fouroversix_pilot128",
        "baselines": {
            "reference": "models/sana-1.6b/pilot256/bf16",
            "nvfp4-hw": "models/sana-1.6b/pilot256/nvfp4-hw",
            "e0m3": "models/sana-1.6b/pilot256/e0m3",
            "tile-mix-oracle": "models/sana-1.6b/residual_rotation128/random/tile-mix-oracle",
            "a16w4-residual": "models/sana-1.6b/a16w4_ceiling64/pilot64",
        },
        "baseline_logs": {
            "reference": "models/sana-1.6b/pilot256/logs/bf16.log",
            "nvfp4-hw": "models/sana-1.6b/pilot256/logs/nvfp4-hw.log",
            "e0m3": "models/sana-1.6b/pilot256/logs/e0m3.log",
            "tile-mix-oracle": "models/sana-1.6b/residual_rotation128/logs/random_tile_missing64.log",
        },
        "skip": None,
    },
}
NEW_CONFIGS = ("e0m3-gscale1536", "tile-mix-e0-e2-4over6")


def seed_for_id(image_id: str) -> int:
    value = 0
    for char in image_id:
        value = (value * 31 + ord(char)) % (10**9 + 7)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (1024, 1024):
            raise RuntimeError(f"invalid image {path}: {image.mode} {image.size}")
        if all(low == high for low, high in image.getextrema()):
            raise RuntimeError(f"flat image: {path}")


def expected_paths(directory: Path, samples) -> list[Path]:
    return [directory / info["category"] / f"{image_id}.png" for image_id, info in samples]


def validate_prefix(directory: Path, samples, *, permit_extra: bool) -> None:
    paths = expected_paths(directory, samples)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{directory}: missing {len(missing)} expected images")
    for path in paths:
        validate_image(path)
    all_ids = [path.stem for path in directory.rglob("*.png")]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(f"{directory}: duplicate image IDs")
    expected_ids = {image_id for image_id, _ in samples}
    if not expected_ids.issubset(all_ids):
        raise RuntimeError(f"{directory}: expected ID set mismatch")
    if not permit_extra and set(all_ids) != expected_ids:
        raise RuntimeError(f"{directory}: unexpected extra images")


def require_markers(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(errors="replace")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(f"{path}: provenance markers absent: {missing}")


def cache_metadata(model: str, spec: dict) -> dict:
    paths = {
        "pca": ROOT / f"models/{model}/basis/U-{model}.pt",
        "random_rotation": ROOT / f"models/{model}/basis/R-{model}.pt",
        "hessian": ROOT / (
            "models/pixart-sigma/quantized_cache/hessians_n5120_l224.pt"
            if model == "pixart-sigma"
            else "models/sana-1.6b/quantized_cache/hessians_n5120_l120.pt"
        ),
        "nvfp4_e2m1_gptq_weight": ROOT / spec["cache"],
    }
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for name, path in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    args = parser.parse_args()
    model, spec = args.model, MODEL_SPECS[args.model]
    samples128 = list(json.loads(DATASET.read_text()).items())[:128]
    samples32 = samples128[:32]
    samples64 = samples128[:64]
    target = ROOT / spec["target"]
    target.mkdir(parents=True, exist_ok=True)

    hub_name = "models--" + spec["model_id"].replace("/", "--")
    hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/share2/huggingface/hub")) / hub_name
    revision = (hub / "refs/main").read_text().strip()
    snapshots = sorted(path.name for path in (hub / "snapshots").iterdir() if path.is_dir())
    if revision != spec["revision"] or snapshots != [revision]:
        raise RuntimeError(
            f"{model}: model revision is not the single verified snapshot: "
            f"main={revision}, snapshots={snapshots}"
        )

    # Validate the exact Pilot32 source settings before creating any links.
    pilot32 = ROOT / spec["pilot32"]
    pilot32_logs = ROOT / spec["pilot32_logs"]
    for config in NEW_CONFIGS:
        validate_prefix(pilot32 / config, samples32, permit_extra=False)
        markers = (
            f"--model {model}",
            f"--activation-format {config}",
            "--residual-rotation random",
            "--gptq-batch-size 4",
            "--max-images 32 --batch-size 4",
            spec["cache"],
            f"Activation fake-quant format: {config}",
            "Loading PCA basis from",
            "Loading rotations from",
            "Loading quantized weights from cache:",
        ) + ((spec["skip"],) if spec["skip"] else ())
        require_markers(pilot32_logs / f"pilot32_{config}.log", markers)

    baseline_markers = {
        "reference": ("FP16 reference verified: 0 ActQuantWrapper layers", "batch_size=4"),
        "nvfp4-hw": ("Activation fake-quant format: nvfp4-hw", "Loading quantized weights from cache:"),
        "e0m3": ("Activation fake-quant format: e0m3", "Loading quantized weights from cache:"),
        "tile-mix-oracle": (
            "Activation fake-quant format: tile-mix-oracle",
            "Loading quantized weights from cache:",
        ),
    }
    for config, relative in spec["baselines"].items():
        count_samples = samples64 if config == "a16w4-residual" else samples128
        validate_prefix(ROOT / relative, count_samples, permit_extra=True)
        if config in spec["baseline_logs"]:
            markers = baseline_markers[config] + (("--residual-rotation random",)
                if model == "sana-1.6b" and config != "reference" else ())
            require_markers(ROOT / spec["baseline_logs"][config], markers)

    manifest_rows = []
    for config in NEW_CONFIGS:
        for image_id, info in samples32:
            source = pilot32 / config / info["category"] / f"{image_id}.png"
            destination = target / config / info["category"] / f"{image_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise RuntimeError(f"wrong existing symlink target: {destination}")
            elif destination.exists():
                raise RuntimeError(f"refusing to overwrite existing output: {destination}")
            else:
                destination.symlink_to(os.path.relpath(source, destination.parent))
            source_hash = sha256(source)
            if sha256(destination) != source_hash:
                raise RuntimeError(f"symlink content hash mismatch: {destination}")
            manifest_rows.append({
                "model": model,
                "config": config,
                "index": samples128.index((image_id, info)),
                "image_id": image_id,
                "category": info["category"],
                "seed": seed_for_id(image_id),
                "source": str(source.relative_to(ROOT)),
                "destination": str(destination.relative_to(ROOT)),
                "sha256": source_hash,
            })

    manifest_path = target / "reuse_manifest.csv"
    with manifest_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    provenance = {
        "experiment": f"{model} Four Over Six TileMix Pilot128",
        "source_checkpoint": CHECKPOINT,
        "model_id": spec["model_id"],
        "model_revision": revision,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_prefix_count": 128,
        "seed_rule": "31-polynomial hash of image_id modulo 1,000,000,007",
        "generation": {
            "dtype": spec["dtype"], "steps": 20, "guidance_scale": 4.5,
            "height": 1024, "width": 1024, "batch_size": 4,
            "physical_gpu": 4, "logical_gpu": 0,
            "NVIDIA_TF32_OVERRIDE": "0", "residual_rotation": "random",
            "activation_global_denominator": 1536,
        },
        "reuse": {
            config: {
                "count": 32,
                "source": str((pilot32 / config).relative_to(ROOT)),
                "destination": str((target / config).relative_to(ROOT)),
            }
            for config in NEW_CONFIGS
        },
        "baselines": spec["baselines"],
        "cache_before": cache_metadata(model, spec),
        "manifest": str(manifest_path.relative_to(ROOT)),
    }
    provenance_path = target / "provenance_before.json"
    with provenance_path.open("x") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
