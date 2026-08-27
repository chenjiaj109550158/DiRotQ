"""Collect per-channel activation abs-max at the DiRotQ-absorb-basis hook
points (same 8 input sites as collect_cov.py) for SmoothQuant factors.

Output: dict {cov_key: amax [3072] float32} saved to
models/flux-schnell/basis/absorb_act_amax.pt
"""

import argparse
import glob
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from absorb_basis.collect_cov import build_hook_specs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--calib-dir", default="models/flux-schnell/calibration_dataset/caches")
    ap.add_argument("--out", default="models/flux-schnell/basis/absorb_act_amax.pt")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-files", type=int, default=-1)
    args = ap.parse_args()

    cache_files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    assert cache_files
    if args.max_files > 0:
        cache_files = cache_files[: args.max_files]

    from diffusers import FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    transformer.eval()
    transformer.requires_grad_(False)

    specs = build_hook_specs(transformer, "double") + build_hook_specs(transformer, "single")
    amax = {key: torch.zeros(3072, dtype=torch.float32, device="cuda")
            for key, _, _ in specs}
    hooks = []

    def make_hook(key, slice_end):
        def hook(module, hargs):
            x = hargs[0]
            if slice_end is not None:
                x = x[..., :slice_end]
            a = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
            torch.maximum(amax[key], a, out=amax[key])
        return hook

    for key, mod, slice_end in specs:
        hooks.append(mod.register_forward_pre_hook(make_hook(key, slice_end)))

    model_dtype = torch.bfloat16
    file_batches = [cache_files[i:i + args.batch_size]
                    for i in range(0, len(cache_files), args.batch_size)]

    def load_batch(files):
        return [torch.load(f, map_location="cpu", weights_only=False) for f in files]

    def forward_batch(batch_data):
        latents = torch.cat(
            [d["input_args"][0].to(model_dtype) for d in batch_data], dim=0
        ).to("cuda")
        kwargs = {}
        for k, v in batch_data[0]["input_kwargs"].items():
            if isinstance(v, torch.Tensor):
                vals = [d["input_kwargs"][k] for d in batch_data]
                if v.ndim >= 1 and v.shape[0] == 1:
                    kwargs[k] = torch.cat(
                        [x.to(model_dtype) if x.is_floating_point() else x for x in vals],
                        dim=0,
                    ).to("cuda")
                else:
                    kwargs[k] = (v.to(model_dtype) if v.is_floating_point() else v).to("cuda")
            else:
                kwargs[k] = v
        transformer(latents, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        io_queue = deque()
        it = iter(file_batches)
        for _ in range(2):
            try:
                io_queue.append(pool.submit(load_batch, next(it)))
            except StopIteration:
                break
        for _ in tqdm(range(len(file_batches)), desc="act-amax", dynamic_ncols=True):
            try:
                io_queue.append(pool.submit(load_batch, next(it)))
            except StopIteration:
                pass
            forward_batch(io_queue.popleft().result())

    for h in hooks:
        h.remove()

    out = {k: v.cpu() for k, v in amax.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {len(out)} amax vectors -> {args.out}")


if __name__ == "__main__":
    main()
