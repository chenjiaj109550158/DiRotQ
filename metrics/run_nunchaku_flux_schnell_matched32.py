#!/usr/bin/env python3
"""Run official Nunchaku INT4 FLUX.1-schnell on the matched MJHQ subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time

import diffusers
from diffusers import FluxPipeline
from PIL import Image, ImageStat
import torch

from nunchaku import NunchakuFluxTransformer2dModel
import nunchaku


MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"


def hash_str_to_int(value: str) -> int:
    """Match DiRotQ, DeepCompressor, and the official Nunchaku evaluator."""
    modulus = 10**9 + 7
    result = 0
    for character in value:
        result = (result * 31 + ord(character)) % modulus
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def query_process_mib(pid: int) -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            return int(fields[1])
    return None


def monitor_memory(stop: threading.Event, samples: list[int]) -> None:
    pid = os.getpid()
    while not stop.wait(0.25):
        if (value := query_process_mib(pid)) is not None:
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
        raise RuntimeError(
            f"image mismatch: missing={sorted(map(str, expected-actual))}, "
            f"extra={sorted(map(str, actual-expected))}"
        )
    for path in expected:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1024, 1024):
                raise RuntimeError(f"invalid image {path}: {image.mode} {image.size}")
            if all(low == high for low, high in ImageStat.Stat(image).extrema):
                raise RuntimeError(f"flat image: {path}")
    return {"count": len(expected), "valid_rgb_1024_nonflat": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--minimum-free-mib", type=int, default=16384)
    args = parser.parse_args()

    snapshot = args.model_snapshot.resolve()
    checkpoint = args.checkpoint_dir.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    dataset = args.dataset.resolve()
    output = args.output_dir.resolve()
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError(f"expected FLUX revision {MODEL_REVISION}, got {snapshot.name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError(f"expected Ada sm89, got {torch.cuda.get_device_capability(0)}")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < args.minimum_free_mib * 2**20:
        raise RuntimeError(
            f"only {free_bytes / 2**20:.0f} MiB free; require {args.minimum_free_mib} MiB"
        )

    samples = load_samples(dataset, args.max_images)
    output.mkdir(parents=True, exist_ok=True)
    immutable_paths = [
        source_checkpoint,
        checkpoint / "transformer_blocks.safetensors",
        checkpoint / "unquantized_layers.safetensors",
        checkpoint / "config.json",
        checkpoint / "split_manifest.json",
        dataset,
    ]
    immutable_before = {path.name: file_record(path) for path in immutable_paths}
    split_manifest = json.loads((checkpoint / "split_manifest.json").read_text())
    if split_manifest["source_sha256"] != immutable_before[source_checkpoint.name]["sha256"]:
        raise RuntimeError("legacy split does not match the official source checkpoint")

    stop = threading.Event()
    board_samples: list[int] = []
    monitor = threading.Thread(target=monitor_memory, args=(stop, board_samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    generation_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    try:
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            str(checkpoint),
            torch_dtype=torch.bfloat16,
            device="cuda",
            precision="int4",
            offload=False,
        ).eval()
        pipeline = FluxPipeline.from_pretrained(
            str(snapshot),
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        pipeline.set_progress_bar_config(disable=True)
        for image_id, info in samples:
            path = output / info["category"] / f"{image_id}.png"
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            seed = hash_str_to_int(image_id)
            generator = torch.Generator().manual_seed(seed)
            item_started = time.perf_counter()
            with torch.inference_mode():
                image = pipeline(
                    info["prompt"],
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    generator=generator,
                    height=1024,
                    width=1024,
                ).images[0]
            generation_seconds += time.perf_counter() - item_started
            image.save(path)
            print(f"saved {path.relative_to(output)} seed={seed}", flush=True)
    finally:
        stop.set()
        monitor.join(timeout=5)

    immutable_after = {path.name: file_record(path) for path in immutable_paths}
    if immutable_after != immutable_before:
        raise RuntimeError("a checkpoint or dataset changed during generation")
    manifest = {
        "schema": "svdquant.nunchaku_flux_schnell_matched32",
        "model_revision": MODEL_REVISION,
        "official_checkpoint": immutable_before[source_checkpoint.name],
        "legacy_lossless_split": split_manifest,
        "dataset": immutable_before[dataset.name],
        "contract": {
            "quantization": "official Nunchaku INT4 SVDQuant r32",
            "steps": 4,
            "guidance_scale": 0.0,
            "batch_size": 1,
            "resolution": [1024, 1024],
            "seed": "rolling base-31 hash(image_id) modulo 1e9+7",
        },
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "nunchaku": str(Path(nunchaku.__file__).resolve()),
        },
        "integrity": validate_images(output, samples),
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
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
