#!/usr/bin/env python3
"""Validate immutable SANA WeightMix inputs and link the Pilot16 prefix."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/mjhq_5000_samples.json"
TARGET = ROOT / "models/sana-1.6b/weight_mix_feasibility/pilot64"
PILOT16 = ROOT / "models/sana-1.6b/weight_mix_feasibility/pilot16"
LOGS = ROOT / "models/sana-1.6b/weight_mix_feasibility/logs"
CALIBRATION = ROOT / "models/sana-1.6b/weight_mix_feasibility/calibration"
MODEL_REVISION = "e2b3c0cbffebcd09d83805e88b9f5f106afc74ac"
CONFIGS = {
    "reoptimized-fixed-e2": {
        "kind": "fixed-e2",
        "cache": "models/sana-1.6b/quantized_cache/nvfp4_g16_e0h_weightmix_fixed-e2_gptq_model.pt",
    },
    "reoptimized-fixed-e0": {
        "kind": "fixed-e0",
        "cache": "models/sana-1.6b/quantized_cache/nvfp4_g16_e0h_weightmix_fixed-e0_gptq_model.pt",
    },
    "weight-tilemix": {
        "kind": "tilemix",
        "cache": "models/sana-1.6b/quantized_cache/nvfp4_g16_e0h_weightmix_tilemix_gptq_model.pt",
    },
}
CACHE_FILES = {
    "standard_e2": (
        "models/sana-1.6b/quantized_cache/nvfp4_g16_gptq_model.pt",
        "4c88e701ecee9cb7532e2d8a77223aa2a8bb97c531b1dc3e477ddc0d08fede7e",
    ),
    "e0_activation_hessian": (
        "models/sana-1.6b/quantized_cache/hessians_e0a_n5120_l120_rr-random_g16_tile64x8.pt",
        "55e4f866179c5cefb2dec31847af64ff7c1bb2f3037fa53f71e88ea807d2ae07",
    ),
    "reoptimized_fixed_e2": (
        CONFIGS["reoptimized-fixed-e2"]["cache"],
        "d867c9cc3e9bef11d8aaf031224841e89728621af1b7952eae681693bb809e37",
    ),
    "reoptimized_fixed_e0": (
        CONFIGS["reoptimized-fixed-e0"]["cache"],
        "cbc116d50e9d8f1eeeadad0212a93b8910ee27bb5e4d24676a03f410d03547c3",
    ),
    "weight_tilemix": (
        CONFIGS["weight-tilemix"]["cache"],
        "b6aca81fd6dd0fe350769e6bf426063c903a999046913fad6b08ba804bf4d59d",
    ),
    "pca_basis": (
        "models/sana-1.6b/basis/U-sana-1.6b.pt",
        "7fb5d472e4607b774c545fc3fb9e9c949d5ebe8c531ff3512a28ab90364d9662",
    ),
    "random_rotation": (
        "models/sana-1.6b/basis/R-sana-1.6b.pt",
        "1a57e93c617bb7f4ead54c10166553ed6b8b6eae832df5270f755cc887d6fcb7",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for_id(image_id: str) -> int:
    value = 0
    for char in image_id:
        value = (value * 31 + ord(char)) % (10**9 + 7)
    return value


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (1024, 1024):
            raise RuntimeError(f"invalid image {path}: {image.mode} {image.size}")
        if all(low == high for low, high in image.getextrema()):
            raise RuntimeError(f"flat image: {path}")


def expected_path(directory: Path, sample) -> Path:
    image_id, info = sample
    return directory / info["category"] / f"{image_id}.png"


def validate_prefix(directory: Path, samples, permit_extra: bool) -> None:
    paths = [expected_path(directory, sample) for sample in samples]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{directory}: missing {len(missing)} expected images")
    for path in paths:
        validate_image(path)
    ids = [path.stem for path in directory.rglob("*.png")]
    expected = {image_id for image_id, _ in samples}
    if len(ids) != len(set(ids)) or not expected.issubset(ids):
        raise RuntimeError(f"{directory}: duplicate or mismatched image IDs")
    if not permit_extra and set(ids) != expected:
        raise RuntimeError(f"{directory}: unexpected extra images")


def require_markers(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(errors="replace")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(f"{path}: missing provenance markers {missing}")


def validate_model_revision() -> str:
    hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/share2/huggingface/hub"))
    hub /= "models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers"
    revision = (hub / "refs/main").read_text().strip()
    snapshots = sorted(path.name for path in (hub / "snapshots").iterdir() if path.is_dir())
    if revision != MODEL_REVISION or snapshots != [MODEL_REVISION]:
        raise RuntimeError(f"unexpected SANA revision: main={revision}, snapshots={snapshots}")
    model_utils = (ROOT / "models/sana-1.6b/model_utils.py").read_text()
    for marker in (
        "num_inference_steps=20", "guidance_scale=4.5", "height=1024", "width=1024"
    ):
        if marker not in model_utils:
            raise RuntimeError(f"SANA generation setting changed: missing {marker}")
    return revision


def validate_caches(layer_names: set[str]) -> dict:
    records = {}
    for name, (relative, expected_hash) in CACHE_FILES.items():
        path = ROOT / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"{name}: SHA-256 mismatch")
        stat = path.stat()
        records[name] = {
            "path": relative, "sha256": actual_hash,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }

    hmeta = json.loads(Path(str(ROOT / CACHE_FILES["e0_activation_hessian"][0]) + ".metadata.json").read_text())
    if not (
        hmeta.get("cache_kind") == "e0-activation-hessian"
        and hmeta.get("activation_hessian") == "hardware-e0m3-gscale2688-per-calibration-chunk"
        and hmeta.get("active_layers") == 120
        and hmeta.get("basis_sha256") == CACHE_FILES["pca_basis"][1]
        and hmeta.get("rotation_sha256") == CACHE_FILES["random_rotation"][1]
    ):
        raise RuntimeError("E0 Hessian metadata mismatch")

    for config, spec in CONFIGS.items():
        path = ROOT / spec["cache"]
        metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
        expected = {
            "model": "sana-1.6b", "objective_version": "e0a-weight-mix-gptq-v1",
            "activation_hessian": "hardware-e0m3-gscale2688-per-calibration-chunk",
            "active_layers": 120, "gptq_layers": 120,
            "high_branch_unchanged_layers": 120, "weight_group_size": 16,
            "logical_weight_tile": [64, 8], "stored_weight_tile": [8, 64],
            "residual_rotation": "random", "weight_format": spec["kind"],
            "basis_sha256": CACHE_FILES["pca_basis"][1],
            "rotation_sha256": CACHE_FILES["random_rotation"][1],
            "standard_cache_sha256": CACHE_FILES["standard_e2"][1],
            "rtn_fallbacks": [],
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"{config}: metadata mismatch for {key}")
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        wrapper = sum(f"{name}.weight" in state for name in layer_names)
        modules = sum(f"{name}.module.weight" in state for name in layer_names)
        if wrapper != 120 or modules != 120:
            raise RuntimeError(f"{config}: state coverage {wrapper}/{modules}, expected 120/120")
        records[config] = {
            "metadata": metadata, "wrapper_coverage": wrapper,
            "module_coverage": modules,
        }
        del state

    prior = json.loads((PILOT16 / "cache_provenance.json").read_text())
    validation = prior["validation"]
    if not (
        validation["active_layers"] == 120
        and validation["high_branch_bitwise_equal_layers_each"] == 120
        and validation["low_branch_differs_from_standard_layers_each"] == 120
        and validation["rtn_fallbacks"] == 0
    ):
        raise RuntimeError("Pilot16 layer-by-layer cache validation is incomplete")
    return records


def main() -> None:
    samples64 = list(json.loads(DATASET.read_text()).items())[:64]
    samples16 = samples64[:16]
    revision = validate_model_revision()

    rows = list(csv.DictReader((CALIBRATION / "weight_mix_layer_objectives.csv").open()))
    tile_rows = [row for row in rows if row["mode"] == "tilemix"]
    layer_names = {row["layer"] for row in tile_rows}
    if len(tile_rows) != 120 or len(layer_names) != 120:
        raise RuntimeError("TileMix calibration report does not cover 120 unique layers")
    if any(row["gptq_status"] != "gptq" or row["fallback_reason"] for row in rows):
        raise RuntimeError("calibration report contains GPTQ failure/fallback")
    cache_records = validate_caches(layer_names)

    for config, spec in CONFIGS.items():
        source = PILOT16 / config
        validate_prefix(source, samples16, permit_extra=False)
        require_markers(LOGS / f"pilot16_{config}.log", (
            "--model sana-1.6b", "--activation-format e0m3",
            "--residual-rotation random", "--gptq-batch-size 4",
            f"--weight-mix-cache-kind {spec['kind']}", spec["cache"],
            "--max-images 16 --batch-size 4", "Activation fake-quant format: e0m3",
            "Wrapped 120 layers with ActQuantWrapper.",
            f"Validated E0-Hessian weight cache kind={spec['kind']}",
            "Loading PCA basis from", "Loading rotations from",
            "Loading quantized weights from cache:",
        ))

    baselines = {
        "bf16-reference": ROOT / "models/sana-1.6b/pilot64/fp16",
        "existing-standard-e2": ROOT / "models/sana-1.6b/pilot64/e0m3",
        "w16-ceiling-first32": ROOT / "models/sana-1.6b/weight_mix_feasibility/e0a-w16-residual32",
    }
    validate_prefix(baselines["bf16-reference"], samples64, permit_extra=False)
    validate_prefix(baselines["existing-standard-e2"], samples64, permit_extra=False)
    validate_prefix(baselines["w16-ceiling-first32"], samples64[:32], permit_extra=False)

    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = []
    for config in CONFIGS:
        for index, sample in enumerate(samples16):
            image_id, info = sample
            source = expected_path(PILOT16 / config, sample)
            destination = expected_path(TARGET / config, sample)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise RuntimeError(f"wrong symlink target: {destination}")
            elif destination.exists():
                raise RuntimeError(f"refusing to overwrite {destination}")
            else:
                destination.symlink_to(os.path.relpath(source, destination.parent))
            source_hash = sha256(source)
            if sha256(destination) != source_hash:
                raise RuntimeError(f"symlink hash mismatch: {destination}")
            manifest.append({
                "config": config, "index": index, "image_id": image_id,
                "category": info["category"], "seed": seed_for_id(image_id),
                "source": str(source.relative_to(ROOT)),
                "destination": str(destination.relative_to(ROOT)),
                "sha256": source_hash,
            })

    manifest_path = TARGET / "reuse_manifest.csv"
    with manifest_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    provenance = {
        "experiment": "SANA E0 activation WeightMix Pilot64",
        "checkpoint": "58eec28",
        "model_id": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "model_revision": revision,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_prefix_count": 64,
        "seed_rule": "31-polynomial hash of image_id modulo 1,000,000,007",
        "generation": {
            "dtype": "bf16", "steps": 20, "guidance_scale": 4.5,
            "height": 1024, "width": 1024, "batch_size": 4,
            "physical_gpu": 4, "logical_gpu": 0,
            "NVIDIA_TF32_OVERRIDE": "0", "residual_rotation": "random",
            "activation_format": "e0m3", "activation_global_denominator": 2688,
            "execution": "BF16 reconstructed fake-quant weights, not packed kernel",
        },
        "reuse": {
            config: {"count": 16, "source": str((PILOT16 / config).relative_to(ROOT))}
            for config in CONFIGS
        },
        "generate": {config: {"indices": [16, 63], "count": 48} for config in CONFIGS},
        "baselines": {name: str(path.relative_to(ROOT)) for name, path in baselines.items()},
        "cache_before": cache_records,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "tile_choice_provenance": (
            "The materialized cache has no per-tile choice map. The immutable build report "
            "retains per-layer E0/E2 counts and the implementation fixes stored 8x64 / logical "
            "64x8 mapping; replay would require prohibited GPTQ rebuilding."
        ),
    }
    with (TARGET / "provenance_before.json").open("x") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
