"""Collect per-layer input covariances for DiRotQ-absorb-basis on PixArt-Sigma,
from deepcompressor's qdiff calibration caches (their collect.calib output).

Hook points per block (28 blocks, hidden 1152, ffn 4608):
  attn1_qkv  <- attn1.to_q input   (shared by to_q/to_k/to_v)
  attn1_out  <- attn1.to_out.0 input
  attn2_q    <- attn2.to_q input
  attn2_kv   <- attn2.to_k input   (projected caption, shared by to_k/to_v)
  attn2_out  <- attn2.to_out.0 input
  ffn_up     <- ff.net.0.proj input
  ffn_down   <- ff.net.2 input     (4608, down-absorb)

All accumulators fit on GPU in one pass (~3.3 GB fp32).
Output: {key: H [d,d] float32} (H = 2/n X^T X), single .pt file (~3.3 GB).

Run in the svdquant env from deepcompressor/examples/diffusion:
  python collect_cov_pixart.py --calib-dir datasets/<dtype>/pixart-sigma/<protocol>/qdiff/s128
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def hook_specs(transformer):
    specs = []
    for i, blk in enumerate(transformer.transformer_blocks):
        specs += [
            (f"block.{i}.attn1_qkv", blk.attn1.to_q),
            (f"block.{i}.attn1_out", blk.attn1.to_out[0]),
            (f"block.{i}.attn2_q", blk.attn2.to_q),
            (f"block.{i}.attn2_kv", blk.attn2.to_k),
            (f"block.{i}.attn2_out", blk.attn2.to_out[0]),
            (f"block.{i}.ffn_up", blk.ff.net[0].proj),
            (f"block.{i}.ffn_down", blk.ff.net[2]),
        ]
    return specs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=-1)
    args = ap.parse_args()

    from deepcompressor.app.diffusion.dataset.base import DiffusionDataset
    from diffusers import PixArtTransformer2DModel

    transformer = PixArtTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float16
    ).to("cuda")
    transformer.eval()
    transformer.requires_grad_(False)

    specs = hook_specs(transformer)
    H = {}
    cnt = {}
    for key, mod in specs:
        d = mod.in_features
        H[key] = torch.zeros(d, d, dtype=torch.float32, device="cuda")
        cnt[key] = 0

    hooks = []

    def make_hook(key):
        def hook(module, hargs):
            x = hargs[0].reshape(-1, hargs[0].shape[-1]).float()
            H[key].addmm_(x.t(), x)
            cnt[key] += x.shape[0]
        return hook

    for key, mod in specs:
        hooks.append(mod.register_forward_pre_hook(make_hook(key)))

    from deepcompressor.utils.common import tree_map

    # batch_size=1 with a squeeze keeps each cache's original shapes (incl. the
    # CFG batch dim) and lets DiffusionDataset resolve latent/text-emb file refs
    # and encoder_hidden_states padding exactly as deepcompressor does.
    dataset = DiffusionDataset(args.calib_dir, num_samples=args.num_samples)
    loader = dataset.build_loader(batch_size=1, shuffle=False, num_workers=4)
    print(f"{len(dataset)} calibration caches")

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            x = x.squeeze(0)  # undo collate's batch-of-1 dim
            return x.to("cuda", torch.float16) if x.is_floating_point() else x.to("cuda")
        return x

    for data in tqdm(loader, desc="cov", dynamic_ncols=True):
        input_args = tree_map(to_dev, data["input_args"])
        input_kwargs = tree_map(to_dev, data["input_kwargs"])
        transformer(*input_args, **input_kwargs)

    for h in hooks:
        h.remove()

    out = {}
    for key in H:
        assert cnt[key] > 0, key
        out[key] = (H[key] * (2.0 / cnt[key])).cpu()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {len(out)} covariances -> {args.out}")


if __name__ == "__main__":
    main()
