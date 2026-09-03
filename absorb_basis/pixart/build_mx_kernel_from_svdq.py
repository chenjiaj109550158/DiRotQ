"""Convert SVDQuant's MX-recalibrated PixArt dump (PLAN_MX; deepcompressor
--save-model with configs/svdquant/mxfp4.yaml) into the MXW4A4Linear packed
kernel format, mirroring build_pixart_kernel_from_svdq.py's semantics:
fused-qkv branch row-slicing, lora un-smoothing (their lora runs on raw x
after dividing a by the smooth factors), and out_proj smooth fusing.

Their dumped residual weights sit on the MX grid with THEIR chosen E8M0
exponents (scale.pt `.weight.scale.0`); we pack with those exact exponents.

Run: python absorb_basis/pixart/build_mx_kernel_from_svdq.py \
       --dump models/pixart-sigma/svdq_mx_dump --out <kernel.pt>
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from absorb_basis.mx_quant import BLOCK, E2M1_POS
from absorb_basis.pixart.build_pixart_kernel_from_svdq import block_layer_specs


def pack_with_scales(W: torch.Tensor, s: torch.Tensor):
    """Pack on-grid W with EXPLICIT per-group scales s [oc, ic/32]."""
    oc, ic = W.shape
    g = W.float().reshape(oc, ic // BLOCK, BLOCK) / s.unsqueeze(-1)
    pos = E2M1_POS.to(W.device)
    d = (g.abs().unsqueeze(-1) - pos.view(1, 1, 1, -1)).abs()
    assert float(d.min(dim=-1).values.max()) < 1e-4, "weights not on MX grid"
    idx = d.argmin(dim=-1)
    nib = (idx | ((g < 0).long() << 3)).reshape(oc, ic).to(torch.uint8)
    codes = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    exps = torch.log2(s.float()).round().to(torch.int8).contiguous()
    return codes, exps


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
        W = sd[f"{lpath}.weight"].float()
        oc, ic = W.shape
        scale = sc[f"{lpath}.weight.scale.0"].float().reshape(oc, ic // BLOCK)
        assert f"{lpath}.weight.scale.1" not in sc, "unexpected second scale level (not MX)"
        zero = sc.get(f"{lpath}.weight.zero")
        assert zero is None or float(zero) == 0.0, f"nonzero zero-point at {lpath}"
        smooth = sm[smkey].float()
        a = br[brkey]["a.weight"].float()            # [r, ic] (smoothed-x domain)
        b = br[brkey]["b.weight"].float()            # [k*oc, r]
        assert b.shape[0] % oc == 0
        b_slice = b[slot * oc:(slot + 1) * oc]
        smooth_fused = is_out and fuse_flag
        codes, exps = pack_with_scales(W, scale)
        if smooth_fused:
            s_vec, a_raw = None, a               # producer already smoothed
        else:
            s_vec = smooth.half()
            a_raw = a / smooth.unsqueeze(0)      # lora reads raw x
        out[lpath] = {
            "codes": codes, "exps": exps,
            "lora_down": a_raw.t().half(),       # [ic, r]
            "lora_up": b_slice.half(),           # [oc, r]
            "smooth": (s_vec if s_vec is not None
                       else torch.ones(ic, dtype=torch.float16)),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"converted {len(out)} layers -> {args.out}")


if __name__ == "__main__":
    main()
