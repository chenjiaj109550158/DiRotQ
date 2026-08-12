#!/usr/bin/env python3
"""Finalize SANA WeightMix Pilot64 integrity and immutable-cache provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models/sana-1.6b/weight_mix_feasibility/pilot64"
DATASET = ROOT / "datasets/mjhq_5000_samples.json"
CONFIGS = ("reoptimized-fixed-e2", "reoptimized-fixed-e0", "weight-tilemix")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_record(config: str, index: int, image_id: str, info: dict) -> dict:
    path = TARGET / config / info["category"] / f"{image_id}.png"
    if not path.is_file():
        raise RuntimeError(f"missing image {path}")
    with Image.open(path) as image:
        image.load()
        extrema = image.getextrema()
        stddev = ImageStat.Stat(image).stddev
        if image.mode != "RGB" or image.size != (1024, 1024):
            raise RuntimeError(f"invalid image {path}: {image.mode} {image.size}")
        if all(low == high for low, high in extrema) or max(stddev) == 0:
            raise RuntimeError(f"flat image {path}")
    if index < 16:
        if not path.is_symlink():
            raise RuntimeError(f"reused prefix is not a symlink: {path}")
        source = path.resolve()
        if not source.is_file() or sha256(source) != sha256(path):
            raise RuntimeError(f"bad reused symlink: {path}")
        provenance = "reused-pilot16-symlink"
        source_text = str(source.relative_to(ROOT))
    else:
        if path.is_symlink():
            raise RuntimeError(f"new suffix unexpectedly symlinked: {path}")
        provenance, source_text = "generated-pilot64", ""
    return {
        "config": config, "index": index, "image_id": image_id,
        "category": info["category"], "path": str(path.relative_to(ROOT)),
        "provenance": provenance, "source": source_text,
        "sha256": sha256(path), "mode": "RGB", "width": 1024, "height": 1024,
        "channel_stddev_min": min(stddev), "channel_stddev_max": max(stddev),
    }


def log_record(config: str) -> dict:
    log = TARGET / "logs" / f"generate_{config}.log"
    text = log.read_text(errors="replace")
    kind = {"reoptimized-fixed-e2": "fixed-e2", "reoptimized-fixed-e0": "fixed-e0",
            "weight-tilemix": "tilemix"}[config]
    required = (
        "Activation fake-quant format: e0m3", "residual_rotation=random",
        "Wrapped 120 layers with ActQuantWrapper.",
        f"Validated E0-Hessian weight cache kind={kind}",
        "Found 16/64 target images already generated.",
        "Generating 48 images (batch_size=4)...", "Exit status: 0",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"{config}: missing log markers {missing}")
    forbidden = ("Collecting Hessian", "E0 activation Hessian calibration",
                 "E0-Hessian weight GPTQ formats", "Saving quantized weights", "RTN fallback")
    present = [marker for marker in forbidden if marker in text]
    if present:
        raise RuntimeError(f"{config}: forbidden build/fallback markers {present}")
    wall_match = re.search(r"Elapsed \(wall clock\) time.*?:\s*(\d+):(\d+\.\d+)", text)
    if not wall_match:
        raise RuntimeError(f"{config}: missing wall time")
    wall = int(wall_match.group(1)) * 60 + float(wall_match.group(2))
    progress = re.findall(r"12/12 \[(\d+):(\d+)<", text)
    if not progress:
        raise RuntimeError(f"{config}: missing generation progress time")
    generation = int(progress[-1][0]) * 60 + int(progress[-1][1])
    gpu_path = TARGET / "logs" / f"generate_{config}_gpu.csv"
    peak = 0
    with gpu_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].startswith("timestamp"):
                continue
            if row:
                peak = max(peak, int(row[3].strip().split()[0]))
    return {
        "log": str(log.relative_to(ROOT)), "generation_seconds": generation,
        "wall_seconds": wall, "sampled_peak_vram_mib": peak,
        "cache_build_markers": [], "fallback_markers": [],
    }


def main() -> None:
    samples = list(json.loads(DATASET.read_text()).items())[:64]
    before_path = TARGET / "provenance_before.json"
    before = json.loads(before_path.read_text())
    rows = []
    for config in CONFIGS:
        directory = TARGET / config
        actual = list(directory.rglob("*.png"))
        if len(actual) != 64 or len({path.stem for path in actual}) != 64:
            raise RuntimeError(f"{config}: expected exactly 64 unique PNGs")
        rows.extend(image_record(config, index, image_id, info)
                    for index, (image_id, info) in enumerate(samples))
    _manifest = TARGET / "image_manifest.csv"
    with _manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cache_after = {}
    for name, record in before["cache_before"].items():
        if "path" not in record:
            continue
        path = ROOT / record["path"]
        current = {
            "path": record["path"], "sha256": sha256(path),
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
        }
        if any(current[key] != record[key] for key in ("sha256", "size", "mtime_ns")):
            raise RuntimeError(f"cache mutated: {name}")
        cache_after[name] = current

    output = {
        "experiment": before["experiment"], "checkpoint": before["checkpoint"],
        "generation": before["generation"], "cache_build_performed": False,
        "cache_after": cache_after, "cache_before_after_identical": True,
        "images": {
            config: {"total": 64, "reused_symlinks": 16, "generated": 48,
                     "rgb_1024_decode_nonflat_pass": 64}
            for config in CONFIGS
        },
        "runs": {config: log_record(config) for config in CONFIGS},
        "manifest": str(_manifest.relative_to(ROOT)),
        "runtime_contract": {
            "activation": "fixed hardware e0m3, global denominator 2688",
            "residual_rotation": "random", "active_layers": 120,
            "execution": "BF16 reconstructed fake-quant weights; not packed kernel",
            "tile_mapping": "logical 64x8 / stored 8x64",
            "tile_choice_map": (
                "not saved; immutable build report supplies per-layer counts, while replay "
                "would require prohibited GPTQ rebuilding"
            ),
        },
    }
    (TARGET / "provenance_after.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
