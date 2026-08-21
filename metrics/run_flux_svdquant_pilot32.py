#!/usr/bin/env python3
"""Generate a matched MJHQ-32 FLUX.1-dev SVDQuant/Nunchaku pilot.

This runner deliberately shares the image-id seed and generation contract used
by ``apply_dirotq.py``.  The quantized transformer is the existing Nunchaku
INT4 SVDQuant checkpoint; text encoders and the VAE come from the exact local
FLUX snapshot and are CPU-offloaded to keep the run safe on a shared GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

from PIL import Image, ImageStat
import torch
import diffusers
from diffusers import FluxPipeline
from diffusers.training_utils import set_seed

# The CUDA extension is installed in the dedicated deepcompressor environment,
# but its Python package can be appended (not prepended) to a matching PyTorch
# environment.  Appending preserves the caller's Diffusers version, which is
# essential when comparing against images produced by that same pipeline.
if site := os.environ.get("NUNCHAKU_SITE_PACKAGES"):
    sys.path.append(site)
from nunchaku import NunchakuFluxTransformer2dModel
import nunchaku


MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"


def hash_str_to_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % (10**8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_file(path: Path, *, with_sha: bool = True) -> dict:
    stat = path.stat()
    record = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if with_sha:
        record["sha256"] = sha256_file(path)
    return record


def query_process_mib(pid: int) -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            return int(fields[1])
    return None


def monitor_memory(stop: threading.Event, samples: list[int]) -> None:
    pid = os.getpid()
    while not stop.wait(0.25):
        value = query_process_mib(pid)
        if value is not None:
            samples.append(value)


def load_samples(path: Path, count: int) -> list[tuple[str, dict]]:
    data = json.loads(path.read_text())
    samples = list(data.items())[:count]
    if len(samples) != count:
        raise RuntimeError(f"requested {count} samples, found {len(samples)}")
    return samples


def validate_images(output: Path, samples: list[tuple[str, dict]]) -> dict:
    expected = {
        output / info["category"] / f"{image_id}.png"
        for image_id, info in samples
    }
    actual = set(output.rglob("*.png"))
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise RuntimeError(f"image set mismatch; missing={missing}, extra={extra}")
    for path in sorted(expected):
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1024, 1024):
                raise RuntimeError(f"invalid image contract: {path}: {image.mode} {image.size}")
            extrema = ImageStat.Stat(image).extrema
            if all(lo == hi for lo, hi in extrema):
                raise RuntimeError(f"flat image: {path}")
    return {"count": len(expected), "valid_rgb_1024_nonflat": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--minimum-free-mib", type=int, default=12288)
    args = parser.parse_args()

    snapshot = args.model_snapshot.resolve()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    output = args.output_dir.resolve()
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError(f"expected FLUX revision {MODEL_REVISION}, got {snapshot.name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError(f"expected Ada sm89, got {torch.cuda.get_device_capability(0)}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < args.minimum_free_mib * 2**20:
        raise RuntimeError(
            f"only {free_bytes / 2**20:.0f} MiB free; require {args.minimum_free_mib} MiB"
        )
    samples = load_samples(dataset, args.max_images)
    output.mkdir(parents=True, exist_ok=True)
    existing = sum(
        (output / info["category"] / f"{image_id}.png").is_file()
        for image_id, info in samples
    )
    print(f"Found {existing}/{len(samples)} matched images; generating the remainder.", flush=True)

    immutable_before = {
        "transformer_blocks": stat_file(checkpoint / "transformer_blocks.safetensors"),
        "unquantized_layers": stat_file(checkpoint / "unquantized_layers.safetensors"),
        "checkpoint_config": stat_file(checkpoint / "config.json"),
        "dataset": stat_file(dataset),
    }
    stop = threading.Event()
    board_samples: list[int] = []
    monitor = threading.Thread(target=monitor_memory, args=(stop, board_samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    generation_seconds = 0.0
    try:
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            str(checkpoint), torch_dtype=torch.bfloat16, device="cuda",
            precision="int4", offload=False,
        ).eval()
        pipeline = FluxPipeline.from_pretrained(
            str(snapshot), transformer=transformer, torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        # Keep the persistent Nunchaku transformer on CUDA.  Sequentially
        # offload ordinary Diffusers submodules (especially the large T5) so
        # this accuracy run can safely coexist with an unrelated GPU process.
        pipeline.enable_sequential_cpu_offload(gpu_id=0)
        pipeline.set_progress_bar_config(disable=True)
        torch.cuda.reset_peak_memory_stats()
        for image_id, info in samples:
            path = output / info["category"] / f"{image_id}.png"
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            seed = hash_str_to_int(image_id)
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            item_started = time.perf_counter()
            with torch.inference_mode():
                image = pipeline(
                    info["prompt"], generator=generator,
                    num_inference_steps=25, guidance_scale=3.5,
                    height=1024, width=1024, max_sequence_length=512,
                ).images[0]
            generation_seconds += time.perf_counter() - item_started
            image.save(path)
            print(f"saved {path.relative_to(output)} seed={seed}", flush=True)
    finally:
        stop.set()
        monitor.join(timeout=5)

    integrity = validate_images(output, samples)
    immutable_after = {
        "transformer_blocks": stat_file(checkpoint / "transformer_blocks.safetensors"),
        "unquantized_layers": stat_file(checkpoint / "unquantized_layers.safetensors"),
        "checkpoint_config": stat_file(checkpoint / "config.json"),
        "dataset": stat_file(dataset),
    }
    if immutable_before != immutable_after:
        raise RuntimeError("checkpoint or dataset changed during generation")
    manifest = {
        "schema": "svdquant.nunchaku_flux_mjhq32",
        "model_revision": MODEL_REVISION,
        "checkpoint": str(checkpoint),
        "checkpoint_provenance": immutable_before,
        "dataset": str(dataset),
        "dataset_sha256": immutable_before["dataset"]["sha256"],
        "contract": {
            "quantization": "official Nunchaku INT4 SVDQuant r32",
            "dtype": "BF16 non-transformer modules",
            "batch_size": 1,
            "steps": 25,
            "guidance_scale": 3.5,
            "height": 1024,
            "width": 1024,
            "max_sequence_length": 512,
            "seed": "sha256(image_id) modulo 1e8",
            "cpu_offload": "Diffusers sequential CPU offload; Nunchaku transformer resident CUDA",
        },
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "nunchaku_module": str(Path(nunchaku.__file__).resolve()),
        },
        "integrity": integrity,
        "runtime": {
            "generation_seconds_this_invocation": generation_seconds,
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_peak_mib": max(board_samples) if board_samples else None,
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
