#!/usr/bin/env python3
"""Measure five FLUX shared-basis arms with real W4A16 modulators at B=1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL = Path("/tmp/dirotq_flux_shared_pca.iQPi5A/formal")
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
SCHEMES = {
    "per-layer-pca": "U-flux-dev-per-layer-norot-down.pt",
    "shared-width": "bases/U-flux-dev-shared-width.pt",
    "shared-operator": "bases/U-flux-dev-shared-operator.pt",
    "shared-operator-stage4": "bases/U-flux-dev-shared-operator-stage4.pt",
    "representative-operator": "bases/U-flux-dev-representative-operator.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class ProcessMemorySampler:
    def __init__(self, physical_gpu: int, pid: int, interval: float = 0.2):
        self.physical_gpu = physical_gpu
        self.pid = pid
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi", f"--id={self.physical_gpu}",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True, capture_output=True,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    if len(fields) == 2 and fields[0] == str(self.pid):
                        self.peak_mib = max(self.peak_mib, int(fields[1]))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def run_logged(command: list[str], log: Path, env: dict[str, str], gpu: int) -> dict:
    if log.exists():
        raise FileExistsError(f"refusing to overwrite log: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            text=True,
        )
        sampler = ProcessMemorySampler(gpu, process.pid)
        sampler.start()
        returncode = process.wait()
        sampler.stop()
    if returncode:
        raise RuntimeError(f"command failed ({returncode}); see {log}")
    return {
        "command": command,
        "log": str(log),
        "wall_seconds": time.time() - started,
        "process_peak_mib": sampler.peak_mib,
    }


def parse_measurements(log: Path) -> dict:
    text = log.read_text()
    storage_match = re.search(r"Real INT4 persistent storage: (\{.*\})", text)
    inference_match = re.search(
        r"Inference-only peak CUDA memory: allocated=(\d+) bytes, reserved=(\d+) bytes",
        text,
    )
    if not storage_match or not inference_match:
        raise RuntimeError(f"missing memory records in {log}")
    return {
        "persistent": json.loads(storage_match.group(1)),
        "peak_allocated_bytes": int(inference_match.group(1)),
        "peak_reserved_bytes": int(inference_match.group(2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--stage", choices=("build", "generate", "all"), default="all")
    parser.add_argument("--wait-for-free-mib", type=int, default=0)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    args = parser.parse_args()

    formal = args.formal_root.resolve()
    snapshot = args.model_snapshot.resolve()
    output = (args.output_root or (formal / "real_quant_w4a16_modulators_b1")).resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
        raise RuntimeError(f"expected exact FLUX revision {MODEL_REVISION}: {snapshot}")
    dataset = ROOT / "datasets/mjhq_5000_samples.json"
    rotation = formal / "R-flux-dev.pt"
    w4a16_cache = output / "packed_cache" / "flux-modulators-w4a16-g64-bf16.pt"
    existing_packed_root = formal / "real_quant" / "packed_cache"
    for path in (dataset, rotation):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "NVIDIA_TF32_OVERRIDE": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })

    def command_for(scheme: str) -> list[str]:
        return [
            sys.executable, str(ROOT / "apply_dirotq.py"),
            "--model", "flux-dev",
            "--model-id", str(snapshot),
            "--dataset", str(dataset),
            "--rotation-path", str(rotation),
            "--calib-dir", str(formal / "calibration_dataset/caches"),
            "--gptq", "--gptq-calib-files", "3200", "--gptq-batch-size", "8",
            "--gptq-rtn-layers", ".net.2", "proj_out.linears.1",
            "--real-int4", "--real-w4a16-modulators",
            "--real-w4a16-cache", str(w4a16_cache),
            "--batch-size", "1", "--max-images", "1",
            "--basis-path", str(formal / SCHEMES[scheme]),
            "--quantized-cache", str(formal / "quantized_cache" / f"{scheme}.pt"),
            "--real-int4-cache", str(existing_packed_root / f"{scheme}.packed-int4.pt"),
        ]

    records: list[dict] = []
    if args.stage in {"build", "all"} and not w4a16_cache.exists():
        from utils.flux_w4a16_modulators import (
            build_w4a16_cache_from_safetensors,
            save_w4a16_cache,
            w4a16_provenance,
        )

        started = time.time()
        cache, report = build_w4a16_cache_from_safetensors(snapshot / "transformer")
        cache_manifest = save_w4a16_cache(
            cache, report, w4a16_cache, provenance=w4a16_provenance(str(snapshot))
        )
        records.append({
            "stage": "build",
            "wall_seconds": time.time() - started,
            "source": "direct immutable HF safetensors",
            "cache_manifest": cache_manifest,
        })
    if args.stage in {"generate", "all"}:
        if not w4a16_cache.is_file():
            raise FileNotFoundError(w4a16_cache)
        wait_started = time.time()
        required_free = max(12_000, args.wait_for_free_mib)
        while True:
            free_result = subprocess.run(
                ["nvidia-smi", f"--id={args.physical_gpu}", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True, capture_output=True,
            )
            free_mib = int(free_result.stdout.strip()) if free_result.returncode == 0 else 0
            if free_mib >= required_free:
                print(
                    f"GPU {args.physical_gpu} has {free_mib} MiB free; starting formal run.",
                    flush=True,
                )
                break
            elapsed = time.time() - wait_started
            if not args.max_wait_seconds or elapsed >= args.max_wait_seconds:
                break
            print(
                f"GPU {args.physical_gpu}: {free_mib} MiB free; waiting for "
                f"{required_free} MiB ({elapsed:.0f}s elapsed).",
                flush=True,
            )
            time.sleep(min(30, args.max_wait_seconds - elapsed))
        if free_mib < required_free:
            raise RuntimeError(
                f"GPU {args.physical_gpu} has only {free_mib} MiB free; "
                "refusing to interfere with another workload"
            )
        for scheme in SCHEMES:
            packed = existing_packed_root / f"{scheme}.packed-int4.pt"
            packed_manifest = packed.with_suffix(packed.suffix + ".manifest.json")
            if not packed.is_file() or not packed_manifest.is_file():
                raise FileNotFoundError(f"missing packed cache/manifest for {scheme}")
            provenance = json.loads(packed_manifest.read_text())["provenance"]
            image_dir = output / "images" / scheme
            log = output / "logs" / f"{scheme}-generate.log"
            record = run_logged(
                command_for(scheme) + [
                    "--real-int4-fake-cache-sha256",
                    provenance["fake_quant_cache_sha256"],
                    "--real-int4-hessian-sha256", provenance["hessian_sha256"],
                    "--output-dir", str(image_dir),
                ],
                log, env, args.physical_gpu,
            )
            pngs = sorted(image_dir.glob("*.png"))
            if len(pngs) != 1:
                raise RuntimeError(f"{scheme}: expected one PNG, found {len(pngs)}")
            record.update(parse_measurements(log))
            record["scheme"] = scheme
            record["image"] = str(pngs[0])
            records.append(record)

    manifest = {
        "schema": "dirotq.flux_shared_pca_w4a16_memory",
        "version": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "model_revision": MODEL_REVISION,
        "dataset_sha256": sha256_file(dataset),
        "rotation_sha256": sha256_file(rotation),
        "w4a16_cache": str(w4a16_cache),
        "w4a16_cache_sha256": sha256_file(w4a16_cache),
        "records": records,
    }
    manifest_path = output / f"run_manifest_{args.stage}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
