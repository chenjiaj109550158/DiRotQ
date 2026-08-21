#!/usr/bin/env python3
"""Matched transformer-only memory benchmark for five FLUX shared-PCA arms."""

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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stat_record(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def compute_processes(physical_gpu: int) -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={physical_gpu}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi process query failed")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, used = (part.strip() for part in line.split(","))
        rows.append({"pid": int(pid), "used_memory_mib": int(used)})
    return rows


class ProcessMemorySampler:
    def __init__(self, physical_gpu: int, pid: int, interval: float = 0.2):
        self.physical_gpu = physical_gpu
        self.pid = pid
        self.interval = interval
        self.peak_mib = 0
        self.samples = 0
        self.foreign_pids_seen: set[int] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            for row in compute_processes(self.physical_gpu):
                if row["pid"] == self.pid:
                    self.peak_mib = max(self.peak_mib, row["used_memory_mib"])
                    self.samples += 1
                else:
                    self.foreign_pids_seen.add(row["pid"])
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def run_one(
    command: list[str], *, log: Path, env: dict[str, str], physical_gpu: int
) -> dict:
    if log.exists():
        raise FileExistsError(f"refusing to overwrite {log}")
    foreign = compute_processes(physical_gpu)
    if foreign:
        raise RuntimeError(
            f"GPU {physical_gpu} is not idle; refusing contaminated measurement: {foreign}"
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            text=True,
        )
        sampler = ProcessMemorySampler(physical_gpu, process.pid)
        sampler.start()
        returncode = process.wait()
        sampler.stop()
    if returncode:
        raise RuntimeError(f"benchmark failed ({returncode}); see {log}")
    if sampler.foreign_pids_seen:
        raise RuntimeError(
            "foreign GPU processes appeared during the measurement: "
            f"{sorted(sampler.foreign_pids_seen)}"
        )
    return {
        "command": command,
        "log": str(log),
        "wall_seconds": time.time() - started,
        "process_peak_mib": sampler.peak_mib,
        "process_memory_samples": sampler.samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    args = parser.parse_args()

    formal = args.formal_root.resolve()
    snapshot = args.model_snapshot.resolve()
    output = args.output_root.resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
        raise RuntimeError(f"expected exact FLUX revision {MODEL_REVISION}: {snapshot}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")

    wait_started = time.time()
    while True:
        foreign = compute_processes(args.physical_gpu)
        if not foreign:
            break
        elapsed = time.time() - wait_started
        if elapsed >= args.max_wait_seconds:
            raise RuntimeError(
                f"GPU {args.physical_gpu} did not become idle within "
                f"{args.max_wait_seconds}s: {foreign}"
            )
        print(
            f"GPU {args.physical_gpu} busy with {foreign}; waiting "
            f"({elapsed:.0f}/{args.max_wait_seconds}s)",
            flush=True,
        )
        time.sleep(min(30, args.max_wait_seconds - elapsed))
    output.mkdir(parents=True)

    dataset = ROOT / "datasets/mjhq_5000_samples.json"
    rotation = formal / "R-flux-dev.pt"
    packed_root = formal / "real_quant" / "packed_cache"
    w4a16 = (
        formal / "real_quant_w4a16_modulators_b1" / "packed_cache" /
        "flux-modulators-w4a16-g64-bf16.pt"
    )
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "NVIDIA_TF32_OVERRIDE": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })

    source_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    records = []
    immutable_before: dict[str, dict] = {}
    for scheme, relative_basis in SCHEMES.items():
        basis = formal / relative_basis
        fake = formal / "quantized_cache" / f"{scheme}.pt"
        packed = packed_root / f"{scheme}.packed-int4.pt"
        packed_manifest = packed.with_suffix(packed.suffix + ".manifest.json")
        for path in (dataset, rotation, basis, fake, packed, packed_manifest, w4a16):
            if not path.is_file():
                raise FileNotFoundError(path)
            immutable_before.setdefault(str(path), stat_record(path))
        provenance = json.loads(packed_manifest.read_text())["provenance"]
        result_json = output / "results" / f"{scheme}.json"
        command = [
            sys.executable, str(ROOT / "apply_dirotq.py"),
            "--model", "flux-dev",
            "--model-id", str(snapshot),
            "--dataset", str(dataset),
            "--rotation-path", str(rotation),
            "--calib-dir", str(formal / "calibration_dataset/caches"),
            "--gptq", "--gptq-calib-files", "3200", "--gptq-batch-size", "8",
            "--gptq-rtn-layers", ".net.2", "proj_out.linears.1",
            "--real-int4", "--real-w4a16-modulators",
            "--real-w4a16-cache", str(w4a16),
            "--basis-path", str(basis),
            "--quantized-cache", str(fake),
            "--real-int4-cache", str(packed),
            "--real-int4-fake-cache-sha256", provenance["fake_quant_cache_sha256"],
            "--real-int4-hessian-sha256", provenance["hessian_sha256"],
            "--no-generate",
            "--flux-transformer-only-memory-output", str(result_json),
            "--flux-transformer-only-warmup", "2",
            "--flux-transformer-only-repeats", "4",
        ]
        result_json.parent.mkdir(parents=True, exist_ok=True)
        run = run_one(
            command,
            log=output / "logs" / f"{scheme}.log",
            env=env,
            physical_gpu=args.physical_gpu,
        )
        measurement = json.loads(result_json.read_text())
        records.append({"scheme": scheme, **run, **measurement})

    immutable_after = {
        path: stat_record(Path(path)) for path in immutable_before
    }
    if immutable_before != immutable_after:
        raise RuntimeError("an immutable input cache changed during the benchmark")
    manifest = {
        "schema": "dirotq.flux_shared_pca_transformer_only_memory",
        "version": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_diff_sha256": sha256_bytes(source_diff),
        "model_revision": MODEL_REVISION,
        "physical_gpu": args.physical_gpu,
        "immutable_inputs_before": immutable_before,
        "immutable_inputs_after": immutable_after,
        "records": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
