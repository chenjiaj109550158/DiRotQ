#!/usr/bin/env python3
"""Build and evaluate the two rank-64 shared-basis FLUX NVFP4 arms.

This is deliberately the repository's legacy DiRotQ NVFP4 numerical path:
E2M1 group-16 weights/activations with high-precision group scales.  The Ada
runtime is fake quantization, not a packed/native NVFP4 kernel.  Each stage is
restartable and refuses to overwrite completed images or immutable inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from evaluate_shared_pca_audit import load_samples, validate_images


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL = Path("/tmp/dirotq_flux_shared_pca.iQPi5A/formal")
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
SCHEMES = {
    "shared-width": "bases/U-flux-dev-shared-width.pt",
    "shared-operator": "bases/U-flux-dev-shared-operator.pt",
}
ROTATION_REL = (
    "scheme_a_rank64_bf16_scales_20260820/"
    "R-flux-dev-shared-width-r64.pt"
)
PRIOR_MEMORY = {
    "shared-width": (
        "scheme_a_rank64_bf16_scales_20260820/"
        "transformer_only_memory_fused.json"
    ),
    "shared-operator": (
        "shared_operator_rank64_bf16_scales_matched_20260820/"
        "transformer_only_memory_fused.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def stat_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def gpu_processes(physical_gpu: int) -> list[dict]:
    result = None
    for attempt in range(5):
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={physical_gpu}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True, capture_output=True,
        )
        if result.returncode == 0:
            break
        time.sleep(0.5 * (attempt + 1))
    assert result is not None
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi query failed")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, used = (field.strip() for field in line.split(","))
        rows.append({"pid": int(pid), "used_memory_mib": int(used)})
    return rows


class ProcessSampler:
    def __init__(self, gpu: int, pid: int):
        self.gpu = gpu
        self.pid = pid
        self.peak_mib = 0
        self.foreign: set[int] = set()
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for row in gpu_processes(self.gpu):
                    if row["pid"] == self.pid:
                        self.peak_mib = max(self.peak_mib, row["used_memory_mib"])
                        self.samples += 1
                    else:
                        self.foreign.add(row["pid"])
            except RuntimeError:
                pass
            self._stop.wait(0.2)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def run_logged(
    command: list[str], log: Path, env: dict, gpu: int, *, allow_foreign: bool,
) -> dict:
    if log.exists():
        raise FileExistsError(f"refusing to overwrite log: {log}")
    preflight_query_error = None
    try:
        foreign = gpu_processes(gpu)
    except RuntimeError as exc:
        if not allow_foreign:
            raise
        foreign = []
        preflight_query_error = str(exc)
    if foreign and not allow_foreign:
        raise RuntimeError(f"GPU {gpu} is not idle: {foreign}")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
        sampler = ProcessSampler(gpu, process.pid)
        sampler.start()
        returncode = process.wait()
        sampler.stop()
    if returncode:
        raise RuntimeError(f"command failed ({returncode}); see {log}")
    if sampler.foreign and not allow_foreign:
        raise RuntimeError(
            f"foreign GPU processes contaminated {log}: {sorted(sampler.foreign)}"
        )
    return {
        "command": command,
        "log": str(log),
        "wall_seconds": time.time() - started,
        "process_peak_mib": sampler.peak_mib,
        "process_memory_samples": sampler.samples,
        "foreign_processes_before": foreign,
        "foreign_pids_seen": sorted(sampler.foreign),
        "board_total_memory_contaminated": bool(
            foreign or sampler.foreign or preflight_query_error
        ),
        "gpu_preflight_query_error": preflight_query_error,
    }


def choose_log(log: Path) -> Path:
    if not log.exists():
        return log
    attempt = 2
    while True:
        candidate = log.with_name(f"{log.stem}-attempt{attempt}{log.suffix}")
        if not candidate.exists():
            return candidate
        attempt += 1


def common_command(
    snapshot: Path, dataset: Path, calibration: Path, rotation: Path,
    basis: Path, cache: Path,
) -> list[str]:
    return [
        sys.executable, str(ROOT / "apply_dirotq.py"),
        "--model", "flux-dev", "--model-id", str(snapshot),
        "--dataset", str(dataset), "--rotation-path", str(rotation),
        "--basis-path", str(basis), "--calib-dir", str(calibration),
        "--gptq", "--nvfp4", "--activation-format", "nvfp4",
        "--gptq-calib-files", "3200", "--gptq-batch-size", "8",
        "--gptq-rtn-layers", ".net.2", "proj_out.linears.1",
        "--quantized-cache", str(cache),
    ]


def write_serialized_estimate(formal: Path, output: Path) -> None:
    """Account packed legacy NVFP4 without claiming an Ada packed runtime.

    The matched INT4 artifacts expose exact low payload/high/bias/frame byte
    counts.  Low K dimensions are all divisible by 64, so changing K64 to K16
    produces exactly four times as many group scales.  The official NVFP4 skip
    list keeps adaptive-norm linears BF16; their dense BF16 weight size is four
    times the prior nibble payload (no padding for these shapes).
    """
    report = {
        "schema": "dirotq.flux_rank64_nvfp4_serialized_estimate",
        "version": 1,
        "warning": (
            "Ada has no packed NVFP4 runtime in this repository.  These are "
            "serialized-contract estimates, not measured CUDA allocations."
        ),
        "schemes": {},
    }
    for scheme, relative in PRIOR_MEMORY.items():
        source = formal / relative
        data = json.loads(source.read_text())["persistent_storage"]
        common = {
            "packed_low_e2m1_payload": data["packed_low_payload"],
            "protected_high_bf16": data["protected_high_bf16"],
            "active_bias": data["active_bias"],
            "adaptive_norm_bf16_weight": data["w4a16_modulator_payload"] * 4,
            "adaptive_norm_bias": data["w4a16_modulator_bias"],
            "other_model_parameters_and_buffers": data[
                "other_model_parameters_and_buffers"
            ],
            "online_pca_residual_frames": data["online_pca_residual_frames"],
        }
        # Prior BF16 K64 scale bytes = groups64 * 2.  Legacy DiRotQ NVFP4
        # needs groups16 * 4-byte high-precision scale => multiply by eight.
        legacy_scale = data["low_group_scales_bf16"] * 8
        # Sidecars for context only.  Either would change the exact numerical
        # contract tested in the images and is therefore not used as a result.
        bf16_scale = data["low_group_scales_bf16"] * 4
        ue4m3_scale = data["low_group_scales_bf16"] * 2
        exact = {**common, "low_group_scales_fp32": legacy_scale}
        exact["total_bytes"] = sum(exact.values())
        report["schemes"][scheme] = {
            "source_memory_record": str(source),
            "legacy_exact_fp32_group_scale": exact,
            "counterfactual_bf16_group_scale_total_bytes": (
                sum(common.values()) + bf16_scale
            ),
            "counterfactual_ue4m3_group_scale_total_bytes": (
                sum(common.values()) + ue4m3_scale
            ),
            "counterfactual_warning": (
                "BF16/UE4M3 scale variants do not reproduce the tested legacy "
                "FP32-scale NVFP4 quantizer."
            ),
        }
    (output / "serialized_memory_estimate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--allow-foreign-processes", action="store_true")
    parser.add_argument(
        "--stage", choices=("build", "generate", "memory", "all"), default="all"
    )
    args = parser.parse_args()

    formal = args.formal_root.resolve()
    snapshot = args.model_snapshot.resolve()
    output = args.output_root.resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
        raise RuntimeError(f"expected exact FLUX revision {MODEL_REVISION}: {snapshot}")
    dataset = ROOT / "datasets/mjhq_5000_samples.json"
    calibration = formal / "calibration_dataset/caches"
    rotation = formal / ROTATION_REL
    hessian = formal / "quantized_cache/hessians_n3200_l456.pt"
    required = [dataset, rotation, hessian]
    for relative in SCHEMES.values():
        required.append(formal / relative)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    samples = load_samples(dataset, 32)
    expected = {f"{info['category']}/{image_id}.png" for image_id, info in samples}

    output.mkdir(parents=True, exist_ok=True)
    write_serialized_estimate(formal, output)
    cache_root = output / "quantized_cache"
    cache_root.mkdir(exist_ok=True)
    # apply_dirotq keys the read-only Hessian by the quantized-cache parent.
    # Link the already validated formal Hessian so this experiment cannot
    # silently recollect 3,200 calibration calls.
    hessian_link = cache_root / hessian.name
    if hessian_link.exists():
        if hessian_link.resolve() != hessian.resolve():
            raise RuntimeError(f"unexpected Hessian provenance at {hessian_link}")
    else:
        hessian_link.symlink_to(hessian)
    records_path = output / "run_manifest.json"
    records = []
    if records_path.is_file():
        records = json.loads(records_path.read_text()).get("records", [])

    immutable_paths = [dataset, rotation, hessian] + [
        formal / relative for relative in SCHEMES.values()
    ]
    before = {str(path): stat_record(path) for path in immutable_paths}
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "NVIDIA_TF32_OVERRIDE": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })

    def write_manifest() -> None:
        after = {str(path): stat_record(path) for path in immutable_paths}
        payload = {
            "schema": "dirotq.flux_rank64_legacy_nvfp4_pilot32",
            "version": 1,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "model_revision": MODEL_REVISION,
            "dataset_sha256": sha256_file(dataset),
            "contract": {
                "schemes": list(SCHEMES),
                "high_rank_hidden": 64,
                "weight_format": "legacy DiRotQ E2M1 group-16 GPTQ",
                "activation_format": "legacy DiRotQ E2M1 group-16",
                "runtime": "BF16 reconstructed fake quant; no packed Ada NVFP4 kernel",
                "batch_size": 1,
                "images": 32,
                "steps": 25,
                "guidance_scale": 3.5,
                "resolution": [1024, 1024],
                "official_nvfp4_skip_list": True,
            },
            "immutable_before": before,
            "immutable_after": after,
            "immutable_unchanged": before == after,
            "records": records,
        }
        records_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for scheme, relative in SCHEMES.items():
        basis = formal / relative
        cache = cache_root / f"{scheme}-r64-nvfp4-g16-gptq.pt"
        common = common_command(snapshot, dataset, calibration, rotation, basis, cache)

        if args.stage in {"build", "all"} and not cache.is_file():
            record = run_logged(
                common + ["--no-generate"],
                choose_log(output / "logs" / f"{scheme}-build.log"),
                env, args.physical_gpu,
                allow_foreign=args.allow_foreign_processes,
            )
            if not cache.is_file():
                raise RuntimeError(f"{scheme}: build completed without cache")
            record.update({"scheme": scheme, "stage": "build", "cache": stat_record(cache)})
            records.append(record)
            write_manifest()

        if args.stage in {"generate", "all"}:
            if not cache.is_file():
                raise FileNotFoundError(cache)
            image_dir = output / "pilot32" / scheme
            actual = {
                str(path.relative_to(image_dir)) for path in image_dir.rglob("*.png")
            } if image_dir.exists() else set()
            if not actual.issubset(expected):
                raise RuntimeError(f"{scheme}: unexpected PNGs: {actual - expected}")
            if actual != expected:
                attempt = 1
                while True:
                    suffix = "" if attempt == 1 else f"-attempt{attempt}"
                    log = output / "logs" / f"{scheme}-generate{suffix}.log"
                    if not log.exists():
                        break
                    attempt += 1
                record = run_logged(
                    common + [
                        "--batch-size", "1", "--max-images", "32",
                        "--output-dir", str(image_dir),
                    ],
                    log, env, args.physical_gpu,
                    allow_foreign=args.allow_foreign_processes,
                )
                actual_after = {
                    str(path.relative_to(image_dir)) for path in image_dir.rglob("*.png")
                }
                if actual_after != expected:
                    raise RuntimeError(
                        f"{scheme}: missing={expected-actual_after}, extra={actual_after-expected}"
                    )
                validate_images([(scheme, image_dir)], samples)
                record.update({
                    "scheme": scheme, "stage": "generate",
                    "cache": stat_record(cache), "image_count": len(actual_after),
                    "new_images": len(actual_after - actual),
                })
                records.append(record)
                write_manifest()

        if args.stage in {"memory", "all"}:
            if not cache.is_file():
                raise FileNotFoundError(cache)
            result_path = output / "memory" / f"{scheme}.json"
            if not result_path.exists():
                result_path.parent.mkdir(parents=True, exist_ok=True)
                record = run_logged(
                    common + [
                        "--no-generate",
                        "--flux-transformer-only-memory-output", str(result_path),
                        "--flux-transformer-only-warmup", "2",
                        "--flux-transformer-only-repeats", "4",
                    ],
                    choose_log(output / "logs" / f"{scheme}-memory.log"),
                    env, args.physical_gpu,
                    allow_foreign=args.allow_foreign_processes,
                )
                measurement = json.loads(result_path.read_text())
                record.update({
                    "scheme": scheme, "stage": "memory", "cache": stat_record(cache),
                    "measurement": measurement,
                })
                records.append(record)
                write_manifest()

    write_manifest()
    print(json.dumps(json.loads(records_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
