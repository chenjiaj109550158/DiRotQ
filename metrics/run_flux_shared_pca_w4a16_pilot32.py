#!/usr/bin/env python3
"""Generate the matched five-arm FLUX W4A16-modulator Pilot32.

The one-image W4A16 smoke is reused by symlink.  Each arm is otherwise run in
its own process so CUDA memory and failures remain attributable.  Existing
images are never overwritten: ``apply_dirotq.generate_images`` skips the
expected paths and this driver rejects unexpected PNGs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_shared_pca_audit import image_path, load_samples, validate_images
from run_flux_shared_pca_w4a16_memory import (
    DEFAULT_FORMAL,
    MODEL_REVISION,
    ROOT,
    SCHEMES,
    parse_measurements,
    run_logged,
    sha256_file,
)


def immutable_stat(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def expected_relpaths(samples: list[tuple[str, dict]]) -> set[str]:
    return {f"{info['category']}/{image_id}.png" for image_id, info in samples}


def choose_log(log_dir: Path, scheme: str) -> Path:
    primary = log_dir / f"{scheme}-generate.log"
    if not primary.exists():
        return primary
    attempt = 2
    while (candidate := log_dir / f"{scheme}-generate-attempt{attempt}.log").exists():
        attempt += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()

    if args.count != 32:
        raise ValueError("formal W4A16 quality run is frozen to 32 samples")
    formal = args.formal_root.resolve()
    snapshot = args.model_snapshot.resolve()
    output = args.output_root.resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
        raise RuntimeError(f"expected exact FLUX revision {MODEL_REVISION}: {snapshot}")

    dataset = ROOT / "datasets/mjhq_5000_samples.json"
    rotation = formal / "R-flux-dev.pt"
    calibration = formal / "calibration_dataset/caches"
    packed_root = formal / "real_quant/packed_cache"
    smoke_root = formal / "real_quant_w4a16_modulators_b1/images"
    w4a16_cache = (
        formal / "real_quant_w4a16_modulators_b1/packed_cache/"
        "flux-modulators-w4a16-g64-bf16.pt"
    )
    required = (dataset, rotation, calibration, w4a16_cache)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    samples = load_samples(dataset, args.count)
    expected = expected_relpaths(samples)
    first_id, first_info = samples[0]

    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    records_path = output / "run_manifest.json"
    if records_path.exists():
        manifest = json.loads(records_path.read_text())
        records = manifest.get("records", [])
    else:
        records = []

    watched = [rotation, w4a16_cache]
    for scheme, relative_basis in SCHEMES.items():
        watched.extend([
            formal / relative_basis,
            formal / "quantized_cache" / f"{scheme}.pt",
            packed_root / f"{scheme}.packed-int4.pt",
            packed_root / f"{scheme}.packed-int4.pt.manifest.json",
        ])
    for path in watched:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path): immutable_stat(path) for path in watched}

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "NVIDIA_TF32_OVERRIDE": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })

    def write_manifest() -> None:
        after = {str(path): immutable_stat(path) for path in watched}
        data = {
            "schema": "dirotq.flux_shared_pca_w4a16_pilot32",
            "version": 1,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "model_revision": MODEL_REVISION,
            "dataset_sha256": sha256_file(dataset),
            "rotation_sha256": sha256_file(rotation),
            "w4a16_cache_sha256": sha256_file(w4a16_cache),
            "generation": {
                "count": args.count,
                "batch_size": 1,
                "num_inference_steps": 25,
                "guidance_scale": 3.5,
                "height": 1024,
                "width": 1024,
                "physical_gpu": args.physical_gpu,
                "logical_gpu": 0,
                "tf32_override": 0,
            },
            "artifact_stats_before": before,
            "artifact_stats_after": after,
            "artifact_stats_unchanged": before == after,
            "records": records,
        }
        records_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    common = [
        sys.executable, str(ROOT / "apply_dirotq.py"),
        "--model", "flux-dev", "--model-id", str(snapshot),
        "--dataset", str(dataset), "--rotation-path", str(rotation),
        "--calib-dir", str(calibration), "--gptq",
        "--gptq-calib-files", "3200", "--gptq-batch-size", "8",
        "--gptq-rtn-layers", ".net.2", "proj_out.linears.1",
        "--real-int4", "--real-w4a16-modulators",
        "--real-w4a16-cache", str(w4a16_cache),
        "--batch-size", "1", "--max-images", str(args.count),
    ]

    for scheme, relative_basis in SCHEMES.items():
        image_dir = output / "pilot32" / scheme
        actual = {
            str(path.relative_to(image_dir)) for path in image_dir.rglob("*.png")
        } if image_dir.exists() else set()
        if not actual.issubset(expected):
            raise RuntimeError(f"{scheme}: unexpected existing PNGs: {actual - expected}")

        source = image_path(smoke_root / scheme, first_id, first_info)
        destination = image_path(image_dir, first_id, first_info)
        if not source.is_file():
            raise FileNotFoundError(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"{scheme}: reused smoke image hash mismatch")

        actual = {str(path.relative_to(image_dir)) for path in image_dir.rglob("*.png")}
        if actual == expected:
            validate_images([(scheme, image_dir)], samples)
            print(f"{scheme}: preserving completed 32-image run", flush=True)
            continue

        packed = packed_root / f"{scheme}.packed-int4.pt"
        packed_manifest = packed.with_suffix(packed.suffix + ".manifest.json")
        provenance = json.loads(packed_manifest.read_text())["provenance"]
        command = common + [
            "--basis-path", str(formal / relative_basis),
            "--quantized-cache", str(formal / "quantized_cache" / f"{scheme}.pt"),
            "--real-int4-cache", str(packed),
            "--real-int4-fake-cache-sha256", provenance["fake_quant_cache_sha256"],
            "--real-int4-hessian-sha256", provenance["hessian_sha256"],
            "--output-dir", str(image_dir),
        ]
        log = choose_log(output / "logs", scheme)
        print(
            f"{scheme}: starting with {len(actual)}/32 images; log={log}", flush=True
        )
        record = run_logged(command, log, env, args.physical_gpu)
        actual_after = {
            str(path.relative_to(image_dir)) for path in image_dir.rglob("*.png")
        }
        if actual_after != expected:
            raise RuntimeError(
                f"{scheme}: missing={expected-actual_after}, extra={actual_after-expected}"
            )
        validate_images([(scheme, image_dir)], samples)
        record.update(parse_measurements(log))
        record.update({
            "scheme": scheme,
            "reused_image_count": 1,
            "generated_image_count": args.count - len(actual),
            "final_image_count": len(actual_after),
            "reused_image": str(destination),
            "reused_image_sha256": sha256_file(destination),
        })
        records.append(record)
        write_manifest()
        print(f"{scheme}: complete (32/32)", flush=True)

    write_manifest()
    validate_images(
        [(scheme, output / "pilot32" / scheme) for scheme in SCHEMES], samples
    )
    print(f"All five W4A16 Pilot32 arms complete: {output}", flush=True)


if __name__ == "__main__":
    main()
