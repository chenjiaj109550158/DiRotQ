"""Collect per-layer input covariances for DiRotQ-absorb-basis on SANA-1.6B
(ch5632 bf16 variant), from deepcompressor's qdiff calibration caches.

Hook points per block (20 blocks, hidden 2240, GLUMBConv ffn 5632):
  attn1_qkv  <- attn1.to_q input      (shared by to_q/to_k/to_v; linear attn)
  attn1_out  <- attn1.to_out.0 input
  attn2_q    <- attn2.to_q input      (cross-attn; KV skipped = attn_add)
  attn2_out  <- attn2.to_out.0 input
  ffn_up     <- ff.conv_inverted input  (1x1 conv: [B,C,H,W] -> rows)
  ffn_down   <- ff.conv_point input     (1x1 conv, 5632, down-absorb)

Output: {key: H [d,d] float32} (H = 2/n X^T X), single .pt file.

Run in the svdquant env from deepcompressor/examples/diffusion:
  python collect_cov_sana.py --calib-dir datasets/torch.bfloat16/<model>/<proto>/qdiff/s128 --out <cov.pt>
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_ID = "Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632"


def hook_specs(transformer):
    specs = []
    for i, blk in enumerate(transformer.transformer_blocks):
        specs += [
            (f"block.{i}.attn1_qkv", blk.attn1.to_q),
            (f"block.{i}.attn1_out", blk.attn1.to_out[0]),
            (f"block.{i}.attn2_q", blk.attn2.to_q),
            (f"block.{i}.attn2_out", blk.attn2.to_out[0]),
            (f"block.{i}.ffn_up", blk.ff.conv_inverted),
            (f"block.{i}.ffn_down", blk.ff.conv_point),
        ]
    return specs


def in_dim(mod):
    if isinstance(mod, torch.nn.Conv2d):
        assert mod.kernel_size == (1, 1) and mod.groups == 1, mod
        return mod.in_channels
    return mod.in_features


def rows(x):
    """Flatten a linear input [.., d] or a 1x1-conv input [B,C,H,W] to [N,d]."""
    if x.dim() == 4:  # conv: channels-first
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return x.reshape(-1, x.shape[-1])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=-1)
    ap.add_argument("--act-quant", action="store_true",
                    help="accumulate H over NVFP4-quantized inputs Q(x/s)")
    ap.add_argument("--smooth-pt", default=None)
    ap.add_argument("--gains", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    from deepcompressor.utils.common import tree_map
    from diffusers import SanaTransformer2DModel

    transformer = SanaTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        variant="bf16",
    ).to("cuda")
    transformer.eval()
    transformer.requires_grad_(False)

    specs = hook_specs(transformer)
    H, cnt = {}, {}
    for key, mod in specs:
        d = in_dim(mod)
        H[key] = torch.zeros(d, d, dtype=torch.float32, device="cuda")
        cnt[key] = 0

    hook_smooth = {}
    if args.smooth_pt:
        import json as _json

        from absorb_basis.sana.build_sana_kernel import layer_table as _lt
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
            x = rows(hargs[0])
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

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    if args.num_samples > 0:
        files = files[: args.num_samples]
    assert files, f"no .pt caches in {args.calib_dir}"
    print(f"{len(files)} calibration caches")

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to("cuda", torch.bfloat16) if x.is_floating_point() else x.to("cuda")
        return x

    for f in tqdm(files, desc="cov", dynamic_ncols=True):
        data = torch.load(f, map_location="cpu", weights_only=False)
        transformer(*tree_map(to_dev, data["input_args"]), **tree_map(to_dev, data["input_kwargs"]))

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
