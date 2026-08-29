"""Convert SVDQuant's calibrated SANA-1.6B dump (model.pt + scale.pt +
smooth.pt + branch.pt) into the pad-first SVDQW4A4Linear kernel checkpoint
format of build_sana_kernel.py, so both methods run the identical kernel path.

Notes
-----
- Their attn1 branch is fused-qkv (a [32,2240], b [6720,32]) — split exactly
  into three per-layer rank-32 loras sharing the down projection.
- Their out_proj smoothing is folded into the value/preceding weights when
  `proj.fuse_when_possible` (smooth_fused): kernel smooth = ones there,
  lora_down NOT unsmoothed; other layers get lora_down / smooth.
- model.pt weights are the on-grid dequant residuals in the smoothed domain
  (eff grid = scale.0 x scale.1); we pad to 128-aligned dims BEFORE packing
  with our own pack chain (their converter's internal padding is inconsistent
  with the python SVDQW4A4Linear shapes).

Run in the svdquant env from the repo root:
  python absorb_basis/sana/build_sana_kernel_from_svdq.py \
      --dump models/sana-1.6b/svdq_model_dump --out <kernel.pt>
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from absorb_basis.build_checkpoint import pack_layer, pack_perm_vector
from absorb_basis.sana.build_sana_kernel import pad128


def block_layer_specs(prefix):
    """(layer path, smooth/branch key, branch row slot, is_out_proj)"""
    return [
        (f"{prefix}.attn1.to_q", f"{prefix}.attn1.to_q", 0, False),
        (f"{prefix}.attn1.to_k", f"{prefix}.attn1.to_q", 1, False),
        (f"{prefix}.attn1.to_v", f"{prefix}.attn1.to_q", 2, False),
        (f"{prefix}.attn1.to_out.0", f"{prefix}.attn1.to_out.0", 0, True),
        (f"{prefix}.attn2.to_q", f"{prefix}.attn2.to_q", 0, False),
        (f"{prefix}.attn2.to_out.0", f"{prefix}.attn2.to_out.0", 0, True),
        (f"{prefix}.ff.conv_inverted", f"{prefix}.ff.conv_inverted", 0, False),
        (f"{prefix}.ff.conv_point", f"{prefix}.ff.conv_point", 0, False),
    ]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-blocks", type=int, default=20)
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
    for lpath, bkey, slot, is_out in tqdm(specs, dynamic_ncols=True):
        W = sd[f"{lpath}.weight"].float().cuda()
        if W.dim() == 4:
            assert W.shape[2] == W.shape[3] == 1, lpath
            W = W.view(W.shape[0], W.shape[1])
        oc, ic = W.shape
        top = sc[f"{lpath}.weight.scale.0"].float().view(()).cuda()
        micro = sc[f"{lpath}.weight.scale.1"].float().view(oc, -1).cuda()  # [oc, ng]
        zero = sc.get(f"{lpath}.weight.zero")
        assert zero is None or float(zero) == 0.0, f"nonzero zero-point at {lpath}"
        assert micro.shape[1] * 16 == ic, (lpath, micro.shape, ic)
        s = sm[bkey].float().cuda()
        a = br[bkey]["a.weight"].float().cuda()          # [r, ic] (smoothed domain)
        b = br[bkey]["b.weight"].float().cuda()          # [k*oc, r]
        assert b.shape[0] % oc == 0
        b_slice = b[slot * oc:(slot + 1) * oc]
        smooth_fused = is_out and fuse_flag
        if smooth_fused:
            s_kernel = torch.ones_like(s)
            a_raw = a
        else:
            s_kernel = s
            a_raw = a / s.unsqueeze(0)  # lora runs on RAW x in the kernel

        oc_p, ic_p = pad128(oc), pad128(ic)
        W_p = F.pad(W, (0, ic_p - ic, 0, oc_p - oc))
        micro_p = F.pad(micro, (0, (ic_p - ic) // 16, 0, oc_p - oc), value=1.0)
        a_p = F.pad(a_raw, (0, ic_p - ic))
        b_p = F.pad(b_slice, (0, 0, 0, oc_p - oc))
        s_p = F.pad(s_kernel, (0, ic_p - ic), value=1.0)

        packed = pack_layer(
            W_p.to(torch.bfloat16), top, micro_p,
            a_p.to(torch.bfloat16), b_p.to(torch.bfloat16),
            s_p.to(torch.bfloat16), "plain", "cuda",
        )
        bias = sd.get(f"{lpath}.bias")
        bias_packed = None
        if bias is not None:
            bpad = F.pad(bias.detach().float().view(-1).cuda(), (0, oc_p - oc))
            bias_packed = bpad[pack_perm_vector(oc_p).cuda()].to(torch.bfloat16).cpu()
        out[lpath] = {
            "qweight": packed["qweight"],
            "wscales": packed["wscales"],
            "smooth": packed["smooth"],
            "smooth_orig": packed["smooth_orig"],
            "lora_down": packed["lora_down"],
            "lora_up": packed["lora_up"],
            "wtscale": float(packed["wtscale"].float()),
            "bias": bias_packed,
            "oc": oc, "ic": ic, "oc_p": oc_p, "ic_p": ic_p,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"converted {len(out)} layers -> {args.out}")


if __name__ == "__main__":
    main()
