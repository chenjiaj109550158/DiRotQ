#!/usr/bin/env python3
"""Integrity/cache immutability finalizer for asymmetric SANA Pilot32."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

try:
    from metrics.prepare_asymmetric_tilemix_pilot32 import file_record, validate_images
except ModuleNotFoundError:  # direct ``python metrics/<script>.py`` execution
    from prepare_asymmetric_tilemix_pilot32 import file_record, validate_images


def _log_record(path: Path) -> dict:
    text = path.read_text(errors="replace")
    required = (
        "Verified packed payload/E4M3 scales/global scale against reconstructed runtime weights: "
        "{'verified_layers': 120, 'format': 'hardware-fixed-e0', 'fallback': False}",
        "Done. Images saved to",
        "Process peak CUDA memory:",
    )
    missing = [token for token in required if token not in text]
    forbidden = (
        "Collecting GPTQ Hessians",
        "Running GPTQ",
        "Building hardware",
        "Saving quantized",
        "RTN fallback",
        "CUDA out of memory",
    )
    found_forbidden = [token for token in forbidden if token in text]
    if missing or found_forbidden:
        raise RuntimeError(
            f"{path}: missing={missing}, forbidden={found_forbidden}"
        )
    if re.search(r"\b(?:nan|inf)\b", text, flags=re.IGNORECASE):
        raise RuntimeError(f"{path}: non-finite marker in generation log")
    match = re.search(
        r"Process peak CUDA memory: allocated=(\d+) bytes, reserved=(\d+) bytes", text
    )
    return {
        "path": str(path),
        "sha256": file_record(path)["sha256"],
        "peak_cuda_allocated_bytes": int(match.group(1)),
        "peak_cuda_reserved_bytes": int(match.group(2)),
    }


def _time_record(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            values[key.strip()] = value.strip()
    return {"path": str(path), "fields": values, "sha256": file_record(path)["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--c1-dir", type=Path, required=True)
    parser.add_argument("--c2-dir", type=Path, required=True)
    parser.add_argument("--c1-log", type=Path, required=True)
    parser.add_argument("--c2-log", type=Path, required=True)
    parser.add_argument("--c1-time", type=Path, required=True)
    parser.add_argument("--c2-time", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preflight = json.loads(args.preflight.read_text())
    with args.dataset.open() as handle:
        samples = list(json.load(handle).items())[:32]
    images = {
        "c1": validate_images(args.c1_dir, samples, exact=True),
        "c2": validate_images(args.c2_dir, samples, exact=True),
    }
    after = {}
    mismatches = {}
    for name, before in preflight["artifacts_before"].items():
        current = file_record(Path(before["path"]))
        after[name] = current
        changed = {
            key: {"before": before[key], "after": current[key]}
            for key in ("sha256", "size", "mtime_ns") if before[key] != current[key]
        }
        if changed:
            mismatches[name] = changed
    if mismatches:
        raise RuntimeError(f"read-only cache provenance changed: {mismatches}")

    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "preflight": str(args.preflight),
        "artifacts_after": after,
        "cache_hash_size_mtime_unchanged": True,
        "images": images,
        "logs": {"c1": _log_record(args.c1_log), "c2": _log_record(args.c2_log)},
        "timing": {"c1": _time_record(args.c1_time), "c2": _time_record(args.c2_time)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
