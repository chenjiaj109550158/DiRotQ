"""SVDQuant NVFP4 (svdq-fp4_r32) FLUX.1-dev generation on MJHQ with the official
Nunchaku kernels, replicating deepcompressor's DiffusionEvalConfig._generate
protocol exactly (same dataset sampling, same per-filename hash seeds, same
batch handling, num_steps=50, guidance_scale=3.5, default 1024x1024).

Also records transformer-only memory usage and per-forward latency.
Run from ~/deepcompressor/examples/diffusion.
"""

import argparse
import json
import os
import time

import torch
import diffusers
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from deepcompressor.app.diffusion.dataset.data import get_dataset
from deepcompressor.utils.common import hash_str_to_int

STATS = {"lat_ms": [], "entry_alloc": [], "peak_alloc": []}


def attach_probes(transformer):
    ev = {}

    def pre(module, args, kwargs):
        torch.cuda.synchronize()
        STATS["entry_alloc"].append(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        ev["start"] = s
        return None

    def post(module, args, kwargs, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        e.synchronize()
        STATS["lat_ms"].append(ev["start"].elapsed_time(e))
        STATS["peak_alloc"].append(torch.cuda.max_memory_allocated())
        return None

    transformer.register_forward_pre_hook(pre, with_kwargs=True)
    transformer.register_forward_hook(post, with_kwargs=True)


def used_bytes():
    free, total = torch.cuda.mem_get_info()
    return total - free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=0.0)
    ap.add_argument("--base-model", type=str, default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--weight-repo", type=str, default="mit-han-lab/nunchaku-flux.1-schnell")
    ap.add_argument("--weight-file", type=str, default="svdq-fp4_r32-flux.1-schnell.safetensors")
    ap.add_argument("--weight-path", type=str, default=None,
                    help="local safetensors path; overrides --weight-repo/--weight-file")
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--stats-out", type=str, required=True)
    ap.add_argument("--benchmark", type=str, default="MJHQ",
                    help="benchmark name or a prompts .yaml path (e.g. prompts/qdiff.yaml)")
    args = ap.parse_args()

    torch.cuda.init()
    dev_used_base = used_bytes()
    alloc_base = torch.cuda.memory_allocated()

    if args.weight_path:
        weight_path = args.weight_path
        assert os.path.exists(weight_path), weight_path
    else:
        weight_path = hf_hub_download(args.weight_repo, args.weight_file)
    from nunchaku import NunchakuFluxTransformer2dModel

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        weight_path, device="cuda", torch_dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    dev_used_after_tf = used_bytes()
    alloc_after_tf = torch.cuda.memory_allocated()
    transformer_mem = {
        "device_used_delta_gib": (dev_used_after_tf - dev_used_base) / 2**30,
        "torch_alloc_delta_gib": (alloc_after_tf - alloc_base) / 2**30,
    }
    print("transformer load memory:", json.dumps(transformer_mem))

    pipe = FluxPipeline.from_pretrained(
        args.base_model, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    attach_probes(pipe.transformer)

    dataset = get_dataset(args.benchmark, max_dataset_size=args.num_samples, repeat=1)
    if args.benchmark.endswith((".yaml", ".yml")):
        # mirrors DiffusionEvalConfig.generate's dirpath convention
        bname = os.path.splitext(os.path.basename(args.benchmark))[0]
        dirpath = os.path.join(args.out_root, "samples", "YAML", f"{bname}-{dataset._unchunk_size}")
    else:
        dirpath = os.path.join(args.out_root, "samples", args.benchmark,
                               f"{args.benchmark}-{dataset._unchunk_size}")
    os.makedirs(dirpath, exist_ok=True)
    print(f"MJHQ has {len(dataset)} samples -> {dirpath}")

    image_wall_s = []
    # generation loop mirrors DiffusionEvalConfig._generate (num_gpus=1, rank=0)
    for batch in tqdm(
        dataset.iter(batch_size=args.batch_size, drop_last_batch=False),
        total=(len(dataset) + args.batch_size - 1) // args.batch_size,
        desc="MJHQ",
        dynamic_ncols=True,
    ):
        filenames = batch["filename"]
        if all(os.path.exists(os.path.join(dirpath, f"{f}.png")) for f in filenames):
            continue
        prompts = batch["prompt"]
        seeds = [hash_str_to_int(name) for name in filenames]
        diffusers.training_utils.set_seed(seeds[0])
        generators = [torch.Generator().manual_seed(seed) for seed in seeds]
        t0 = time.perf_counter()
        output = pipe(
            prompts,
            generator=generators,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )
        torch.cuda.synchronize()
        image_wall_s.append(time.perf_counter() - t0)
        for filename, image in zip(filenames, output.images, strict=True):
            image.save(os.path.join(dirpath, f"{filename}.png"))

    lat = STATS["lat_ms"]
    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    summary = {
        "transformer_load_mem": transformer_mem,
        "num_forwards": n,
        "lat_ms_median": lat_sorted[n // 2] if n else None,
        "lat_ms_mean": (sum(lat) / n) if n else None,
        "entry_alloc_gib_max": max(STATS["entry_alloc"]) / 2**30 if n else None,
        "peak_alloc_gib_max": max(STATS["peak_alloc"]) / 2**30 if n else None,
        "forward_act_overhead_gib_max": (
            max(p - e for p, e in zip(STATS["peak_alloc"], STATS["entry_alloc"])) / 2**30 if n else None
        ),
        "image_wall_s_mean": (sum(image_wall_s) / len(image_wall_s)) if image_wall_s else None,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "batch_size": args.batch_size,
    }
    print("NVFP4_TRANSFORMER_STATS:", json.dumps(summary, indent=2))
    with open(args.stats_out, "w") as f:
        json.dump(
            {"summary": summary, "lat_ms": lat, "entry_alloc": STATS["entry_alloc"],
             "peak_alloc": STATS["peak_alloc"], "image_wall_s": image_wall_s},
            f,
        )


if __name__ == "__main__":
    main()
