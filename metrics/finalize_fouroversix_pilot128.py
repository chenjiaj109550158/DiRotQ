#!/usr/bin/env python3
"""Finalize Pilot128 image/cache/runtime provenance without changing images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODES = ("e0m3-gscale1536", "tile-mix-e0-e2-4over6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_after(before: dict) -> dict:
    result = {}
    for name, expected in before.items():
        path = ROOT / expected["path"]
        current = {
            "path": expected["path"], "sha256": sha256(path),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if current != expected:
            raise RuntimeError(f"cache changed during Pilot128: {name}")
        result[name] = current
    return result


def gpu_peak(path: Path) -> int:
    peak = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row = {key.strip(): value for key, value in row.items()}
            if int(row["index"].strip()) != 4:
                continue
            memory = int(row["memory.used [MiB]"].replace("MiB", "").strip())
            peak = max(peak, memory)
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pixart-sigma", "sana-1.6b"), required=True)
    args = parser.parse_args()
    root = ROOT / f"models/{args.model}/fouroversix_pilot128"
    before = json.loads((root / "provenance_before.json").read_text())
    samples = list(json.loads((ROOT / before["dataset"]).read_text()).items())[:128]
    expected = {(info["category"], image_id) for image_id, info in samples}
    manifest = list(csv.DictReader((root / "reuse_manifest.csv").open(newline="")))
    manifests = {(row["config"], row["image_id"]): row for row in manifest}
    validation, runtime = {}, {}
    forbidden = re.compile(
        r"\b(?:nan|inf|oom)\b|out of memory|cuda error|silent fallback|"
        r"collecting hessians|running gptq|quantizing weights",
        re.IGNORECASE,
    )
    for mode in MODES:
        directory = root / mode
        files = list(directory.rglob("*.png"))
        actual = {(path.parent.name, path.stem) for path in files}
        if len(files) != 128 or actual != expected:
            raise RuntimeError(f"{args.model}/{mode}: ID/count mismatch")
        symlinks = regular = 0
        minimum_range = 255
        for path in files:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (1024, 1024):
                    raise RuntimeError(f"invalid PNG: {path}")
                spread = max(high - low for low, high in image.getextrema())
                if spread == 0:
                    raise RuntimeError(f"flat image: {path}")
                minimum_range = min(minimum_range, spread)
            if path.is_symlink():
                symlinks += 1
                row = manifests.get((mode, path.stem))
                if row is None or path.resolve() != (ROOT / row["source"]).resolve():
                    raise RuntimeError(f"symlink provenance mismatch: {path}")
                if sha256(path) != row["sha256"]:
                    raise RuntimeError(f"symlink SHA-256 mismatch: {path}")
            else:
                regular += 1
        if (symlinks, regular) != (32, 96):
            raise RuntimeError(f"{mode}: expected 32 reused + 96 new images")
        log_path = root / "logs" / f"{mode}_missing96.log"
        log = log_path.read_text(errors="replace")
        for marker in ("Found 32/128", "Generating 96 images (batch_size=4)", "All done.", "Exit status: 0"):
            if marker not in log:
                raise RuntimeError(f"{log_path}: missing completion marker {marker!r}")
        hits = sorted(set(match.group(0) for match in forbidden.finditer(log)))
        if hits:
            raise RuntimeError(f"{log_path}: forbidden runtime messages: {hits}")
        wall_match = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:]+(?:\.[0-9]+)?)", log)
        progress = re.findall(r"24/24 \[([^<]+)<00:00", log)
        if wall_match is None or not progress:
            raise RuntimeError(f"{log_path}: runtime parsing failed")
        gpu_path = root / "gpu" / f"{mode}_missing96.csv"
        validation[mode] = {
            "count": 128, "symlink_count": symlinks, "new_count": regular,
            "unique_expected_ids": True, "rgb_1024x1024": True,
            "fully_decodable": True, "flat_images": 0,
            "minimum_channel_range": minimum_range,
            "symlink_source_sha256_identical": True,
            "runtime_error_scan": [],
        }
        runtime[mode] = {
            "generation_progress": progress[-1],
            "wall": wall_match.group(1),
            "physical_gpu4_peak_vram_mib": gpu_peak(gpu_path),
        }
    result = {
        **before,
        "reuse_and_generation": validation,
        "runtime": runtime,
        "cache_after": cache_after(before["cache_before"]),
        "cache_sha256_and_mtime_unchanged": True,
        "calibration_pca_rotation_hessian_gptq_recomputed": False,
    }
    with (root / "provenance_after.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"validation": validation, "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
