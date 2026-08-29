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
    ap.add_argument("--act-quant", action="store_true",
                    help="accumulate H over NVFP4-quantized inputs Q(x/s) "
                         "(PLAN_ROUND2 G: act-aware GPTQ)")
    ap.add_argument("--smooth-pt", default=None,
                    help="SVDQuant smooth.pt; with --gains, hooks whose mean "
                         "layer gain exceeds --threshold collect in the "
                         "smoothed domain x/s (PLAN_ROUND2 S+G)")
    ap.add_argument("--gains", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
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

    # per-hook smooth vectors (ones when the hook is not selected)
    hook_smooth = {}
    if args.smooth_pt:
        import json as _json

        from absorb_basis.pixart.build_pixart_sim import layer_table as _lt
        sm = torch.load(args.smooth_pt, map_location="cpu", weights_only=False)
        gains = _json.load(open(args.gains))
        by_hook = {}
        for lp, ck in _lt(len(transformer.transformer_blocks)):
            skey = lp.replace(".to_k", ".to_q").replace(".to_v", ".to_q") \
                if ".attn1" in lp else lp
            if lp in gains:
                by_hook.setdefault(ck, []).append((gains[lp], skey))
        for ck, gl in by_hook.items():
            mean_gain = sum(g for g, _ in gl) / len(gl)
            if mean_gain > args.threshold:
                hook_smooth[ck] = sm[gl[0][1]].float().cuda()
        print(f"smoothed hooks: {len(hook_smooth)}/{len(by_hook)}")

    if args.act_quant:
        from absorb_basis.pixart.run_pixart_sim_generate import act_fp4_sim

    hooks = []

    def make_hook(key):
        s = hook_smooth.get(key)

        def hook(module, hargs):
            x = hargs[0].reshape(-1, hargs[0].shape[-1])
            if s is not None:
                x = x.float() / s
            if args.act_quant:
                x = act_fp4_sim(x)
            x = x.float()
            H[key].addmm_(x.t(), x)
            cnt[key] += x.shape[0]
        return hook

    for key, mod in specs:
        hooks.append(mod.register_forward_pre_hook(make_hook(key)))

    import glob

    from deepcompressor.utils.common import tree_map

    # deepcompressor's collect saves .pt caches with fully materialized tensors
    # (no text-emb file refs for PixArt; encoder_hidden_states already padded
    # to the attention-mask length) — load and forward them directly.
    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    if args.num_samples > 0:
        files = files[: args.num_samples]
    assert files, f"no .pt caches in {args.calib_dir}"
    print(f"{len(files)} calibration caches")

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to("cuda", torch.float16) if x.is_floating_point() else x.to("cuda")
        return x

    for f in tqdm(files, desc="cov", dynamic_ncols=True):
        data = torch.load(f, map_location="cpu", weights_only=False)
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
