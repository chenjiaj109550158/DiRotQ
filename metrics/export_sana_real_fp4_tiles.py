#!/usr/bin/env python3
"""Run one stats-only SANA trajectory and export paired real FP4 packages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models/sana-1.6b"
QUANTIZED = MODEL_ROOT / "quantized_cache"
E2_CACHE = QUANTIZED / "nvfp4_g16_e0h_hardware-fixed-e2_gptq_model.pt"
E0_CACHE = QUANTIZED / "nvfp4_g16_e0h_hardware-fixed-e0_gptq_model.pt"
BASIS = MODEL_ROOT / "basis/U-sana-1.6b.pt"
ROTATION = MODEL_ROOT / "basis/R-sana-1.6b.pt"
MODEL_NAME = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
HF_REF = (
    Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/share2/huggingface/hub"))
    / "models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/refs/main"
)
PACKAGE_NAMES = ("sana_real_e0xe2_v1", "sana_real_e0xe0_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_snapshot(path: Path) -> dict:
    info = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "mtime": datetime.fromtimestamp(info.st_mtime, timezone.utc).astimezone().isoformat(),
    }


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise RuntimeError(f"package tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def tracked_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def receiver_commit(receiver_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=receiver_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _write_manifest_fixed_size(package: Path, manifest: dict) -> None:
    path = package / "manifest.json"
    previous = None
    for _ in range(16):
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        actual = sum(item.stat().st_size for item in package.rglob("*") if item.is_file())
        if manifest["package_size_bytes"] == actual and previous == actual:
            return
        manifest["package_size_bytes"] = actual
        previous = actual
    raise RuntimeError("package_size_bytes did not converge")


def strict_verify(receiver_root: Path, package: Path) -> dict:
    verifier = receiver_root / "kernels/blackwell_e0_probe/real_tile_handoff/verify_package.py"
    if not verifier.is_file():
        raise FileNotFoundError(f"receiver verifier not found: {verifier}")
    completed = subprocess.run(
        [sys.executable, "-m",
         "kernels.blackwell_e0_probe.real_tile_handoff.verify_package",
         str(package)], cwd=receiver_root,
        check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    report = json.loads(completed.stdout)
    if not report.get("passed") or report.get("cuda_touched"):
        raise RuntimeError(f"strict receiver verification failed: {report}")
    return report


def deterministic_archive(output_root: Path) -> dict:
    destination = output_root / "sana_real_fp4_tiles_v1.tar"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".real-fp4-", suffix=".tar", dir=output_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
            for package_name in PACKAGE_NAMES:
                package = output_root / package_name
                for path in [package, *sorted(package.rglob("*"))]:
                    if path.is_symlink():
                        raise RuntimeError(f"archive input contains a symlink: {path}")
                    arcname = path.relative_to(output_root).as_posix()
                    info = archive.gettarinfo(str(path), arcname=arcname)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def finalize_packages(receiver_root: Path, output_root: Path) -> dict:
    commit = tracked_commit()
    packages = {}
    for name in PACKAGE_NAMES:
        package = output_root / name
        manifest_path = package / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing package manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        manifest["producer"]["git_commit"] = commit
        _write_manifest_fixed_size(package, manifest)
        verification = strict_verify(receiver_root, package)
        packages[name] = {
            "path": str(package.resolve()),
            "tree_sha256": tree_sha256(package),
            "manifest_sha256": sha256_file(manifest_path),
            "size_bytes": verification["package_size_bytes"],
            "case_count": verification["case_count"],
            "receiver": verification,
        }
    archive = deterministic_archive(output_root)
    result = {
        "producer_commit": commit,
        "receiver_commit": receiver_commit(receiver_root),
        "packages": packages,
        "archive": archive,
    }
    (output_root / "final_handoff.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def capture_config(receiver_root: Path, output_root: Path) -> dict:
    if not HF_REF.is_file():
        raise FileNotFoundError(f"local SANA revision ref is missing: {HF_REF}")
    model_revision = HF_REF.read_text().strip()
    if len(model_revision) != 40:
        raise RuntimeError(f"local SANA revision is not a full commit: {model_revision!r}")
    dataset = json.loads((ROOT / "datasets/mjhq_5000_samples.json").read_text())
    prompt_id = next(iter(dataset))
    return {
        "receiver_root": str(receiver_root.resolve()),
        "output_root": str(output_root.resolve()),
        "producer_commit": tracked_commit(),
        "model_name": MODEL_NAME,
        "model_revision": model_revision,
        "prompt_image_id": prompt_id,
        "pca_basis_sha256": sha256_file(BASIS),
        "rotation_sha256": sha256_file(ROTATION),
        "e2_cache_path": str(E2_CACHE.resolve()),
        "e0_cache_path": str(E0_CACHE.resolve()),
        "cases": [
            {
                "case_id": "attn_input_early_aligned",
                "layer_name": "transformer_blocks.0.attn1.to_q",
                "timestep_index": 0, "timestep_occurrence": 0,
                "row_start": 0, "column_start": 0, "M": 16, "N": 8,
            },
            {
                "case_id": "attn_output_early_tail",
                "layer_name": "transformer_blocks.0.attn1.to_out.0",
                "timestep_index": 0, "timestep_occurrence": 0,
                "row_start": 0, "column_start": 8, "M": 17, "N": 9,
            },
            {
                "case_id": "attn_input_mid_tail",
                "layer_name": "transformer_blocks.10.attn1.to_q",
                "timestep_index": 10, "timestep_occurrence": 0,
                "row_start": 16, "column_start": 16, "M": 17, "N": 9,
            },
            {
                "case_id": "attn_output_mid_aligned",
                "layer_name": "transformer_blocks.10.attn1.to_out.0",
                "timestep_index": 10, "timestep_occurrence": 0,
                "row_start": 16, "column_start": 32, "M": 16, "N": 8,
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    receiver_root = args.receiver_root.resolve()
    output_root = args.output_root.resolve()
    if receiver_commit(receiver_root) != "5dedc04376a8e68ae0f1b9103900e4c78db5478e":
        raise RuntimeError("receiver worktree is not pinned to 5dedc043")
    if args.finalize_only:
        print(json.dumps(finalize_packages(receiver_root, output_root), indent=2, sort_keys=True))
        return
    if any((output_root / name).exists() for name in PACKAGE_NAMES):
        raise FileExistsError("refusing to overwrite an existing real-tile package")
    output_root.mkdir(parents=True, exist_ok=True)

    protected = [
        BASIS, ROTATION, E2_CACHE, Path(str(E2_CACHE) + ".packing.pt"),
        E0_CACHE, Path(str(E0_CACHE) + ".packing.pt"),
    ]
    before = {str(path.resolve()): file_snapshot(path) for path in protected}
    config = capture_config(receiver_root, output_root)
    config_path = output_root / "export_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    command = [
        sys.executable, str(ROOT / "apply_dirotq.py"),
        "--model", "sana-1.6b",
        "--dataset", str(ROOT / "datasets/mjhq_5000_samples.json"),
        "--gptq", "--nvfp4", "--activation-format", "e0m3",
        "--residual-rotation", "random",
        "--hardware-weight-cache-kind", "hardware-fixed-e2",
        "--quantized-cache", str(E2_CACHE),
        "--stats-only", "--max-images", "1", "--batch-size", "1",
        "--output-dir", str(output_root / "no_images"),
        "--real-tile-export-config", str(config_path),
    ]
    environment = dict(os.environ)
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    environment["CUDA_VISIBLE_DEVICES"] = "4"
    environment["HF_HUB_OFFLINE"] = "1"
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    wall = time.monotonic() - started
    if completed.returncode:
        raise SystemExit(completed.returncode)

    after = {str(path.resolve()): file_snapshot(path) for path in protected}
    if before != after:
        changed = [path for path in before if before[path] != after[path]]
        raise RuntimeError(f"protected cache/basis files changed during capture: {changed}")
    verification = {
        name: strict_verify(receiver_root, output_root / name) for name in PACKAGE_NAMES
    }
    report = {
        "command": command,
        "environment": {
            key: environment[key]
            for key in ("NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")
        },
        "wall_time_seconds": wall,
        "receiver_commit": receiver_commit(receiver_root),
        "cache_before": before,
        "cache_after": after,
        "cache_unchanged": True,
        "verification": verification,
    }
    (output_root / "export_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
