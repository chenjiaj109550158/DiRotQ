#!/usr/bin/env python3
"""Run the matched FLUX.1-schnell shared-width-r64 fake-INT4 Pilot32.

All large artifact locations are command-line arguments so the run can be
reproduced on a machine whose model and quantization caches are not mounted at
the producer's original paths.  See
``docs/flux_schnell_shared_width_r64/README.md`` for the packed-real command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"


def gpu_rows() -> list[dict[str, int | str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, name, used, free, util = (part.strip() for part in line.split(","))
        rows.append(
            {
                "index": int(index),
                "name": name,
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_percent": int(util),
            }
        )
    return rows


def wait_for_gpu(candidates: tuple[int, ...], minimum_free_mib: int) -> dict:
    consecutive: dict[int, int] = {index: 0 for index in candidates}
    while True:
        rows = {int(row["index"]): row for row in gpu_rows()}
        stamp = datetime.now(timezone.utc).astimezone().isoformat()
        print(
            stamp,
            " ".join(
                f"gpu{index}:free={rows[index]['free_mib']}MiB"
                for index in candidates
            ),
            flush=True,
        )
        for index in candidates:
            if rows[index]["free_mib"] >= minimum_free_mib:
                consecutive[index] += 1
            else:
                consecutive[index] = 0
            if consecutive[index] >= 3:
                return rows[index]
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing U/R and the saved qdiff calibration cache",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Directory containing the dense GPTQ and packed W4A16 caches",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/mjhq_5000_samples.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--candidate-gpus", default="4,5,6")
    parser.add_argument("--minimum-free-mib", type=int, default=33000)
    args = parser.parse_args()
    candidates = tuple(int(value) for value in args.candidate_gpus.split(","))

    snapshot = args.model_snapshot.resolve()
    run_root = args.run_root.resolve()
    cache_root = args.cache_root.resolve()
    dataset = args.dataset.resolve()
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError(
            f"expected FLUX.1-schnell revision {MODEL_REVISION}, got {snapshot.name}"
        )
    basis = run_root / "U-flux-schnell-shared-width.pt"
    rotation = run_root / "R-flux-schnell-shared-width-r64.pt"
    calibration = (
        run_root
        / "calibration/torch.bfloat16/flux.1-schnell/fmeuler4-g0/qdiff/s128/caches"
    )
    output_dir = (args.output_dir or run_root / "dirotq_shared_width_r64_int4").resolve()
    log_path = (args.log_path or run_root / "logs/dirotq-int4-w4a16.log").resolve()
    dense_cache = cache_root / "int4_g64_gptq_model.pt"
    w4a16_cache = cache_root / "flux-schnell-adaptive-norm-int4-g64-bf16.pt"
    for path in (snapshot, dataset, basis, rotation, calibration, dense_cache, w4a16_cache):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    selected = wait_for_gpu(candidates, args.minimum_free_mib)
    gpu = int(selected["index"])
    command = [
        sys.executable,
        str(ROOT / "apply_dirotq.py"),
        "--model", "flux-schnell",
        "--model-id", str(snapshot),
        "--dataset", str(dataset),
        "--basis-path", str(basis),
        "--rotation-path", str(rotation),
        "--calib-dir", str(calibration),
        "--gptq",
        "--gptq-calib-files", "512",
        "--gptq-batch-size", "4",
        "--gptq-rtn-layers", ".net.2", "proj_out.linears.1",
        "--quantized-cache", str(dense_cache),
        "--real-w4a16-modulators",
        "--real-w4a16-cache", str(w4a16_cache),
        "--generate",
        "--batch-size", "1",
        "--max-images", str(args.max_images),
        "--output-dir", str(output_dir),
    ]
    env = os.environ.copy()
    env.update(
        {
            "NVIDIA_TF32_OVERRIDE": "0",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            # The 380 small/medium Hessians consume ~13.7 GiB when resident
            # on GPU.  The previously validated Schnell run accumulates them
            # on CPU and uses the GPU only for each X^T X partial.
            "DIROTQ_GPU_HESSIAN_MAX_D": "0",
        }
    )
    print("COMMAND", " ".join(command), flush=True)
    print(f"PHYSICAL_GPU={gpu} LOGICAL_GPU=0", flush=True)

    stop = threading.Event()
    peak_used_mib = int(selected["used_mib"])

    def sample_memory() -> None:
        nonlocal peak_used_mib
        while not stop.wait(1):
            try:
                row = next(row for row in gpu_rows() if row["index"] == gpu)
                peak_used_mib = max(peak_used_mib, int(row["used_mib"]))
            except Exception as error:  # monitoring must not alter the run
                print(f"VRAM_MONITOR_WARNING={error}", flush=True)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).astimezone().isoformat()
    with log_path.open("w") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    wall_seconds = time.monotonic() - started
    stop.set()
    monitor.join(timeout=3)
    png_count = len(list(output_dir.rglob("*.png")))
    manifest = {
        "schema": "dirotq.flux_schnell_shared_width_r64_int4_matched32",
        "version": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": command,
        "environment": {
            "NVIDIA_TF32_OVERRIDE": "0",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "DIROTQ_GPU_HESSIAN_MAX_D": "0",
            "physical_gpu": gpu,
            "logical_gpu": 0,
        },
        "gpu_at_selection": selected,
        "peak_board_used_mib": peak_used_mib,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "wall_seconds": wall_seconds,
        "returncode": returncode,
        "png_count": png_count,
        "output_dir": str(output_dir),
        "dense_cache": str(dense_cache),
        "w4a16_cache": str(w4a16_cache),
        "log": str(log_path),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
