"""Collect input covariances for DiRotQ-absorb-basis on SDXL-Turbo.

Two parts (GPU memory: linear covs ~8 GB, conv im2col covs ~13.5 GB):
  --part linear : 6 hook points per transformer block (70 blocks; attn1_qkv,
                  attn1_out, attn2_q, attn2_out, ffn_up, ffn_down)
  --part conv   : per-conv im2col covariance (34 resnet conv1/conv2; the
                  input is unfolded with the conv's own padding/stride so
                  H lives in the (ic*9)-dim patch domain)

Supports --act-quant / --smooth-pt / --gains like the pixart collector
(linear part only; convs deploy as fp16 so their H stays raw).

Run in the svdquant env from deepcompressor/examples/diffusion:
  python collect_cov_sdxl.py --calib-dir <caches> --out <cov.pt> --part linear
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_ID = "stabilityai/sdxl-turbo"


def linear_hook_specs(unet):
    specs = []
    for name, mod in unet.named_modules():
        if re.search(r"transformer_blocks\.\d+$", name):
            specs += [
                (f"{name}#attn1_qkv", mod.attn1.to_q),
                (f"{name}#attn1_out", mod.attn1.to_out[0]),
                (f"{name}#attn2_q", mod.attn2.to_q),
                (f"{name}#attn2_out", mod.attn2.to_out[0]),
                (f"{name}#ffn_up", mod.ff.net[0].proj),
                (f"{name}#ffn_down", mod.ff.net[2]),
            ]
    return specs


def conv_specs(unet):
    specs = []
    for name, mod in unet.named_modules():
        if re.search(r"resnets\.\d+\.conv[12]$", name):
            specs.append((name, mod))
    return specs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=["linear", "conv"], required=True)
    ap.add_argument("--num-samples", type=int, default=-1)
    ap.add_argument("--act-quant", action="store_true")
    ap.add_argument("--smooth-pt", default=None)
    ap.add_argument("--gains", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    from deepcompressor.utils.common import tree_map
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    unet.eval()
    unet.requires_grad_(False)

    H, cnt, hooks = {}, {}, []

    if args.part == "linear":
        specs = linear_hook_specs(unet)
        hook_smooth = {}
        if args.smooth_pt:
            sm = torch.load(args.smooth_pt, map_location="cpu", weights_only=False)
            gains = json.load(open(args.gains))
            for key, mod in specs:
                blk, point = key.split("#")
                lp = {"attn1_qkv": f"{blk}.attn1.to_q", "attn1_out": f"{blk}.attn1.to_out.0",
                      "attn2_q": f"{blk}.attn2.to_q", "attn2_out": f"{blk}.attn2.to_out.0",
                      "ffn_up": f"{blk}.ff.net.0.proj", "ffn_down": f"{blk}.ff.net.2"}[point]
                if gains.get(lp, -1e9) > args.threshold and lp in sm:
                    hook_smooth[key] = sm[lp].float().cuda()
            print(f"smoothed hooks: {len(hook_smooth)}/{len(specs)}")
        if args.act_quant:
            from absorb_basis.pixart.run_pixart_sim_generate import act_fp4_sim
        for key, mod in specs:
            d = mod.in_features
            H[key] = torch.zeros(d, d, dtype=torch.float32, device="cuda")
            cnt[key] = 0

            def mk(key):
                s = hook_smooth.get(key) if args.smooth_pt else None

                def hook(m, a):
                    x = a[0].reshape(-1, a[0].shape[-1])
                    if s is not None:
                        x = x.float() / s
                    if args.act_quant:
                        x = act_fp4_sim(x)
                    x = x.float()
                    H[key].addmm_(x.t(), x)
                    cnt[key] += x.shape[0]
                return hook
            hooks.append(mod.register_forward_pre_hook(mk(key)))
    else:
        for name, mod in conv_specs(unet):
            d = mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
            H[name] = torch.zeros(d, d, dtype=torch.float32, device="cuda")
            cnt[name] = 0

            def mk(name, mod):
                def hook(m, a):
                    x = a[0].float()
                    cols = F.unfold(x, kernel_size=mod.kernel_size,
                                    padding=mod.padding, stride=mod.stride,
                                    dilation=mod.dilation)  # [B, ic*k*k, L]
                    r = cols.permute(0, 2, 1).reshape(-1, cols.shape[1])
                    H[name].addmm_(r.t(), r)
                    cnt[name] += r.shape[0]
                return hook
            hooks.append(mod.register_forward_pre_hook(mk(name, mod)))

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    if args.num_samples > 0:
        files = files[: args.num_samples]
    assert files
    print(f"{len(files)} calibration caches, {len(H)} hook points")

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to("cuda", torch.float16) if x.is_floating_point() else x.to("cuda")
        return x

    for f in tqdm(files, desc=f"cov-{args.part}", dynamic_ncols=True):
        data = torch.load(f, map_location="cpu", weights_only=False)
        unet(*tree_map(to_dev, data["input_args"]), **tree_map(to_dev, data["input_kwargs"]))

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
