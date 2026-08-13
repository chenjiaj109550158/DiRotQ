#!/usr/bin/env python3
"""Read-only provenance preflight for the SANA asymmetric Pilot32."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import torch
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).astimezone().isoformat(),
    }


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def validate_images(root: Path, samples, *, exact: bool) -> list[dict]:
    expected = {f"{info['category']}/{image_id}.png" for image_id, info in samples}
    actual = {str(path.relative_to(root)) for path in root.rglob("*.png")}
    if expected - actual or (exact and actual - expected):
        raise RuntimeError(
            f"{root}: exact first32 mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    rows = []
    for image_id, info in samples:
        path = root / info["category"] / f"{image_id}.png"
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1024, 1024):
                raise RuntimeError(f"{path}: invalid image {image.mode} {image.size}")
            if all(lo == hi for lo, hi in image.getextrema()):
                raise RuntimeError(f"{path}: flat image")
        rows.append({"image_id": image_id, "relative_path": str(path.relative_to(root)),
                     "sha256": sha256(path)})
    return rows


def generation_command(args, activation_format: str, output_dir: Path, stats: Path | None):
    command = [
        "env", "NVIDIA_TF32_OVERRIDE=0", "CUDA_VISIBLE_DEVICES=4",
        "conda", "run", "--no-capture-output", "-n", "dirotq",
        "python", "-u", "apply_dirotq.py",
        "--model", "sana-1.6b",
        "--dataset", str(args.dataset),
        "--gptq", "--nvfp4",
        "--activation-format", activation_format,
        "--residual-rotation", "random",
        "--gptq-batch-size", "4",
        "--hardware-weight-cache-kind", "hardware-fixed-e0",
        "--hardware-weight-hessian-sha256", args.hessian_sha256,
        "--quantized-cache", str(args.weight_cache),
        "--max-images", "32", "--batch-size", "4",
        "--output-dir", str(output_dir),
    ]
    if stats is not None:
        command.extend(("--collect-format-stats", "--format-stats-output", str(stats)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--weight-cache", type=Path, required=True)
    parser.add_argument("--packing-sidecar", type=Path, required=True)
    parser.add_argument("--hessian-sha256", required=True)
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--c0-log", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--model-ref", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if git("branch", "--show-current") != "exp/activation-tilemix-fakequant":
        raise RuntimeError("unexpected branch")
    tracked = git("status", "--short", "--untracked-files=no")
    if tracked:
        raise RuntimeError("tracked worktree must be clean before Pilot32")

    with args.dataset.open() as handle:
        samples = list(json.load(handle).items())[:32]
    if len(samples) != 32:
        raise RuntimeError("dataset has fewer than 32 prompts")
    c0_manifest = validate_images(args.c0_dir, samples, exact=True)
    reference_manifest = validate_images(args.reference_dir, samples, exact=False)
    log_text = args.c0_log.read_text(errors="replace")
    required_log_tokens = (
        "--activation-format e0m3",
        "--residual-rotation random",
        "--hardware-weight-cache-kind hardware-fixed-e0",
        str(args.weight_cache),
        "--max-images 32",
        "--batch-size 4",
        "fallback': False",
    )
    missing = [token for token in required_log_tokens if token not in log_text]
    if missing:
        raise RuntimeError(f"C0 provenance log is missing required tokens: {missing}")

    c1_dir = args.output_root / "fixed-e2a-fixed-e0w"
    c2_dir = args.output_root / "tilemix-a-fixed-e0w"
    for target in (c1_dir, c2_dir):
        if target.exists() and any(target.rglob("*.png")):
            raise RuntimeError(f"refusing to overwrite existing images under {target}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_root / "stats" / "tilemix_format_stats.json"
    commands = {
        "no_generate_cache_preflight": generation_command(
            args, "e0m3", args.output_root / "no-generate", None
        ) + ["--no-generate"],
        "c1": generation_command(args, "nvfp4-hw", c1_dir, None),
        "c2": generation_command(args, "tile-mix-oracle", c2_dir, stats_path),
    }
    artifacts = {
        name: file_record(path) for name, path in {
            "pca": args.basis,
            "random_rotation": args.rotation,
            "hardware_fixed_e0_weight_cache": args.weight_cache,
            "hardware_fixed_e0_packing_sidecar": args.packing_sidecar,
        }.items()
    }
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "remote_head": git("rev-parse", "origin/exp/activation-tilemix-fakequant"),
        "tracked_status_at_preflight": tracked.splitlines(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "e0_activation_hessian_sha256": args.hessian_sha256,
        "model_id": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "model_revision": args.model_ref.read_text().strip(),
        "model_ref_path": str(args.model_ref),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "artifacts_before": artifacts,
        "c0_source": {"path": str(args.c0_dir), "log": str(args.c0_log),
                      "images": c0_manifest},
        "bf16_reference": {"path": str(args.reference_dir), "images": reference_manifest},
        "commands": commands,
    }
    temporary = args.output_root / ".preflight_provenance.json.tmp"
    final = args.output_root / "preflight_provenance.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(final)
    print(final)
    for key, command in commands.items():
        print(key + ":", " ".join(command))


if __name__ == "__main__":
    main()
