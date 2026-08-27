"""Collect input covariances for the down-projection layers (K = 12288):
double-block mlp_fc2 (ff.net.2) / mlp_context_fc2 (ff_context.net.2) and the
MLP half of the single-block fused proj_out.

12288^2 fp32 accumulators are 604 MB each, so the 76 keys are collected in
chunks of --keys-per-pass (default 6, ~3.6 GB GPU) with one forward sweep over
the calibration caches per chunk. Each covariance is saved as its own file:
  models/flux-schnell/basis/absorb_cov_down/{key}.pt   (H = 2/n X^T X, fp32)

TF32 is enabled for the accumulation matmuls (same as utils/gptq_utils
collect_hessians); the 1% Hessian damping downstream dwarfs TF32 rounding.
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


def down_specs(transformer):
    """[(key, module, col_slice)] for all K=12288 down-proj inputs."""
    specs = []
    for i, blk in enumerate(transformer.transformer_blocks):
        specs.append((f"layer.{i}.img_ffn.down", blk.ff.net[2], None))
        specs.append((f"layer.{i}.txt_ffn.down", blk.ff_context.net[2], None))
    for i, blk in enumerate(transformer.single_transformer_blocks):
        specs.append((f"single.{i}.mlp.down", blk.proj_out, (3072, 15360)))
    return specs


@torch.no_grad()
def run_chunk(transformer, cache_files, specs, batch_size, device):
    D = 12288
    H = {key: torch.zeros(D, D, dtype=torch.float32, device=device)
         for key, _, _ in specs}
    cnt = {key: 0 for key, _, _ in specs}
    hooks = []

    def make_hook(key, col_slice):
        def hook(module, hargs):
            x = hargs[0]
            if col_slice is not None:
                x = x[..., col_slice[0]:col_slice[1]]
            x = x.reshape(-1, x.shape[-1]).float()
            H[key].addmm_(x.t(), x)
            cnt[key] += x.shape[0]
        return hook

    for key, mod, col_slice in specs:
        hooks.append(mod.register_forward_pre_hook(make_hook(key, col_slice)))

    model_dtype = torch.bfloat16
    file_batches = [cache_files[i:i + batch_size]
                    for i in range(0, len(cache_files), batch_size)]

    def load_batch(files):
        return [torch.load(f, map_location="cpu", weights_only=False) for f in files]

    def forward_batch(batch_data):
        latents = torch.cat(
            [d["input_args"][0].to(model_dtype) for d in batch_data], dim=0
        ).to(device)
        kwargs = {}
        for k, v in batch_data[0]["input_kwargs"].items():
            if isinstance(v, torch.Tensor):
                vals = [d["input_kwargs"][k] for d in batch_data]
                if v.ndim >= 1 and v.shape[0] == 1:
                    kwargs[k] = torch.cat(
                        [x.to(model_dtype) if x.is_floating_point() else x for x in vals],
                        dim=0,
                    ).to(device)
                else:
                    kwargs[k] = (v.to(model_dtype) if v.is_floating_point() else v).to(device)
            else:
                kwargs[k] = v
        transformer(latents, **kwargs)

    desc = f"{specs[0][0]}..{specs[-1][0]} ({len(specs)} keys)"
    with ThreadPoolExecutor(max_workers=2) as pool:
        io_queue = deque()
        it = iter(file_batches)
        for _ in range(2):
            try:
                io_queue.append(pool.submit(load_batch, next(it)))
            except StopIteration:
                break
        for _ in tqdm(range(len(file_batches)), desc=desc, dynamic_ncols=True):
            try:
                io_queue.append(pool.submit(load_batch, next(it)))
            except StopIteration:
                pass
            forward_batch(io_queue.popleft().result())

    for h in hooks:
        h.remove()

    out = {}
    for key in H:
        assert cnt[key] > 0, f"no data for {key}"
        out[key] = (H[key] * (2.0 / cnt[key])).cpu()
        H[key] = None
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--calib-dir", default="models/flux-schnell/calibration_dataset/caches")
    ap.add_argument("--out-dir", default="models/flux-schnell/basis/absorb_cov_down")
    ap.add_argument("--keys-per-pass", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-files", type=int, default=-1)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cache_files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    assert cache_files
    if args.max_files > 0:
        cache_files = cache_files[: args.max_files]
    os.makedirs(args.out_dir, exist_ok=True)

    from diffusers import FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    transformer.eval()
    transformer.requires_grad_(False)

    specs = down_specs(transformer)
    todo = [sp for sp in specs
            if not os.path.exists(os.path.join(args.out_dir, f"{sp[0]}.pt"))]
    print(f"{len(todo)}/{len(specs)} covariances to collect "
          f"({len(cache_files)} caches, {args.keys_per_pass} keys/pass)")

    for c0 in range(0, len(todo), args.keys_per_pass):
        chunk = todo[c0:c0 + args.keys_per_pass]
        covs = run_chunk(transformer, cache_files, chunk, args.batch_size, "cuda")
        for key, Hc in covs.items():
            torch.save(Hc, os.path.join(args.out_dir, f"{key}.pt"))
        del covs
    print("done")


if __name__ == "__main__":
    main()
