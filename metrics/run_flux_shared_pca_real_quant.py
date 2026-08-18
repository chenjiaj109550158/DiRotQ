#!/usr/bin/env python3
"""Build and run the five matched FLUX shared-basis packed-INT4 arms.

This orchestrator is intentionally boring: one process and one model arm at a
time, immutable formal inputs, separate logs, and no overwrite of the existing
fake-quant experiment.  CUDA memory is reported both by ``apply_dirotq.py``
(``torch.cuda.max_memory_allocated``) and by a physical-GPU nvidia-smi sampler.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL = Path("/tmp/dirotq_flux_shared_pca.iQPi5A/formal")
SCHEMES = {
    "per-layer-pca": "U-flux-dev-per-layer-norot-down.pt",
    "shared-width": "bases/U-flux-dev-shared-width.pt",
    "shared-operator": "bases/U-flux-dev-shared-operator.pt",
    "shared-operator-stage4": "bases/U-flux-dev-shared-operator-stage4.pt",
    "representative-operator": "bases/U-flux-dev-representative-operator.pt",
}
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class NvidiaMemorySampler:
    def __init__(self, physical_index: int, interval: float = 0.2):
        self.physical_index = physical_index
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.physical_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                try:
                    self.peak_mib = max(self.peak_mib, int(result.stdout.strip()))
                except ValueError:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=5)


def run_logged(command: list[str], log: Path, env: dict[str, str], gpu: int) -> dict:
    if log.exists():
        raise FileExistsError(f"refusing to overwrite log: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w") as handle, NvidiaMemorySampler(gpu) as sampler:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    record = {
        "command": command,
        "log": str(log),
        "returncode": result.returncode,
        "wall_seconds": time.time() - started,
        "nvidia_smi_total_peak_mib": sampler.peak_mib,
    }
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--stage", choices=("build", "generate", "all"), default="all")
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=32)
    args = parser.parse_args()

    formal = args.formal_root.resolve()
    output = (args.output_root or (formal / "real_quant")).resolve()
    snapshot = args.model_snapshot.resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
        raise RuntimeError(
            f"model snapshot must be exact revision {MODEL_REVISION}: {snapshot}"
        )
    gpu_probe = subprocess.run(
        ["nvidia-smi", f"--id={args.physical_gpu}", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    if gpu_probe.returncode:
        raise RuntimeError("CUDA preflight failed: " + gpu_probe.stderr.strip())

    dataset = ROOT / "datasets/mjhq_5000_samples.json"
    rotation = formal / "R-flux-dev.pt"
    calibration = formal / "calibration_dataset/caches"
    hessian = formal / "quantized_cache/hessians_n3200_l456.pt"
    required = [dataset, rotation, calibration, hessian]
    if any(not path.exists() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.exists()])
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
            "NVIDIA_TF32_OVERRIDE": "0",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    common = [
        sys.executable,
        str(ROOT / "apply_dirotq.py"),
        "--model", "flux-dev",
        "--model-id", str(snapshot),
        "--dataset", str(dataset),
        "--rotation-path", str(rotation),
        "--calib-dir", str(calibration),
        "--gptq",
        "--gptq-calib-files", "3200",
        "--gptq-batch-size", "8",
        "--gptq-rtn-layers", ".net.2", "proj_out.linears.1",
        "--real-int4",
        "--batch-size", "1",
        "--max-images", str(args.max_images),
    ]
    records = []
    for scheme, relative_basis in SCHEMES.items():
        basis = formal / relative_basis
        dense = formal / "quantized_cache" / f"{scheme}.pt"
        sidecar = output / "packed_cache" / f"{scheme}.packed-int4.pt"
        images = output / "pilot32" / scheme
        for path in (basis, dense):
            if not path.is_file():
                raise FileNotFoundError(path)
        arm = common + [
            "--basis-path", str(basis),
            "--quantized-cache", str(dense),
            "--real-int4-cache", str(sidecar),
            "--output-dir", str(images),
        ]
        if args.stage in {"build", "all"} and not sidecar.exists():
            records.append(
                {
                    "scheme": scheme,
                    "stage": "build",
                    **run_logged(
                        arm + ["--real-int4-build", "--no-generate"],
                        output / "logs" / f"{scheme}-build.log",
                        env,
                        args.physical_gpu,
                    ),
                }
            )
        if args.stage in {"generate", "all"}:
            if not sidecar.is_file():
                raise FileNotFoundError(f"missing packed sidecar for generation: {sidecar}")
            records.append(
                {
                    "scheme": scheme,
                    "stage": "generate",
                    **run_logged(
                        arm,
                        output / "logs" / f"{scheme}-generate.log",
                        env,
                        args.physical_gpu,
                    ),
                }
            )

    manifest = {
        "schema": "dirotq.flux_shared_pca_real_quant_run",
        "version": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "model_revision": MODEL_REVISION,
        "dataset_sha256": sha256_file(dataset),
        "rotation_sha256": sha256_file(rotation),
        "hessian_sha256": sha256_file(hessian),
        "records": records,
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

