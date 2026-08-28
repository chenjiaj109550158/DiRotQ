"""Convert SVDQuant's own calibrated PixArt-Sigma quantization (deepcompressor
`--save-model` dump: model.pt + scale.pt + smooth.pt + branch.pt) into the
same per-layer SVDQW4A4Linear kernel checkpoint format that
build_pixart_kernel.py emits, so both methods run on the identical nunchaku
fp4 kernel path.

Notes
-----
- SVDQuant's PixArt low-rank branch is stored for the FUSED qkv (a: [32, ic],
  b: [3*oc, 32]).  A fused rank-32 lora splits EXACTLY into three per-layer
  rank-32 loras sharing the same down-projection (y_q = b[:oc] @ (a @ x)), so
  the split checkpoint is mathematically identical and structurally matches
  our per-layer deployment (fair head-to-head).
- Their out_proj smoothing is folded into the value projection when
  `proj.fuse_when_possible` (smooth_fused): kernel smooth = ones there.
- Uses deepcompressor's OWN converter
  (convert_to_nunchaku_w4x4y16_linear_state_dict), which asserts the dumped
  weights are exactly on the two-level NVFP4 grid.

Run in the svdquant env from the repo root:
  python absorb_basis/pixart/build_pixart_kernel_from_svdq.py \
      --dump models/pixart-sigma/svdq_model_dump --out <kernel.pt>
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict


def block_layer_specs(prefix):
    """(layer path, smooth key, branch key, branch row slice, is_out_proj)"""
    return [
        (f"{prefix}.attn1.to_q", f"{prefix}.attn1.to_q", f"{prefix}.attn1.to_q", 0, False),
        (f"{prefix}.attn1.to_k", f"{prefix}.attn1.to_q", f"{prefix}.attn1.to_q", 1, False),
        (f"{prefix}.attn1.to_v", f"{prefix}.attn1.to_q", f"{prefix}.attn1.to_q", 2, False),
        (f"{prefix}.attn1.to_out.0", f"{prefix}.attn1.to_out.0", f"{prefix}.attn1.to_out.0", 0, True),
        (f"{prefix}.attn2.to_q", f"{prefix}.attn2.to_q", f"{prefix}.attn2.to_q", 0, False),
        (f"{prefix}.attn2.to_out.0", f"{prefix}.attn2.to_out.0", f"{prefix}.attn2.to_out.0", 0, True),
        (f"{prefix}.ff.net.0.proj", f"{prefix}.ff.net.0.proj", f"{prefix}.ff.net.0.proj", 0, False),
        (f"{prefix}.ff.net.2", f"{prefix}.ff.net.2", f"{prefix}.ff.net.2", 0, False),
    ]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-blocks", type=int, default=28)
    args = ap.parse_args()

    sd = torch.load(os.path.join(args.dump, "model.pt"), map_location="cpu", weights_only=False)
    sc = torch.load(os.path.join(args.dump, "scale.pt"), map_location="cpu", weights_only=False)
    sm = torch.load(os.path.join(args.dump, "smooth.pt"), map_location="cpu", weights_only=False)
    br = torch.load(os.path.join(args.dump, "branch.pt"), map_location="cpu", weights_only=False)
    fuse_flag = bool(sm.get("proj.fuse_when_possible", True))
    print(f"proj.fuse_when_possible = {fuse_flag}")

    out = {}
    specs = []
    for bi in range(args.num_blocks):
        specs += block_layer_specs(f"transformer_blocks.{bi}")
    for lpath, smkey, brkey, slot, is_out in tqdm(specs, dynamic_ncols=True):
        W = sd[f"{lpath}.weight"]
        oc = W.shape[0]
        bias = sd.get(f"{lpath}.bias")
        scale = sc[f"{lpath}.weight.scale.0"]
        subscale = sc[f"{lpath}.weight.scale.1"]
        zero = sc.get(f"{lpath}.weight.zero")
        assert zero is None or float(zero) == 0.0, f"nonzero zero-point at {lpath}"
        smooth = sm[smkey]
        a = br[brkey]["a.weight"]          # [r, ic]
        b = br[brkey]["b.weight"]          # [k*oc, r]
        assert b.shape[0] % oc == 0
        b_slice = b[slot * oc:(slot + 1) * oc]
        smooth_fused = is_out and fuse_flag
        conv = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=W, scale=scale, bias=bias, smooth=smooth,
            lora=(a, b_slice), shift=None, smooth_fused=smooth_fused,
            float_point=True, subscale=subscale,
        )
        assert "wtscale" in conv, f"{lpath}: expected per-tensor top scale, got {list(conv)}"
        out[lpath] = {
            "qweight": conv["qweight"],
            "wscales": conv["wscales"],
            "smooth": conv["smooth"].half(),
            "smooth_orig": conv["smooth_orig"].half(),
            "lora_down": conv["lora_down"].half(),
            "lora_up": conv["lora_up"].half(),
            "wtscale": float(conv["wtscale"].float()),
            "bias": conv["bias"].half() if conv.get("bias") is not None else None,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"converted {len(out)} layers -> {args.out}")


if __name__ == "__main__":
    main()
