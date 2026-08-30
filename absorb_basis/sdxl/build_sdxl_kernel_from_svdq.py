"""Convert SVDQuant's calibrated SDXL-Turbo dump into our kernel checkpoint
format (same deployment path as build_sdxl_kernel.py).

- 420 transformer linears (560 after fused-qkv split): shared-down-proj
  branch split, smooth_fused handling per `proj.fuse_when_possible`; packed
  with our chain (dims are 128-multiples, no padding).
- 34 resnet convs: their recipe has NO conv branch/smooth — model.pt holds
  the dequantized quantized conv weight directly; up-block concat convs are
  stored as ConcatConv splits (convs.0/convs.1, no shift keys, biases
  unchanged) and are concatenated back along the in-dim.

Run in the svdquant env from the repo root:
  python absorb_basis/sdxl/build_sdxl_kernel_from_svdq.py \
      --dump models/sdxl-turbo/svdq_model_dump --out <kernel.pt>
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from absorb_basis.build_checkpoint import pack_layer, pack_perm_vector


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = torch.load(os.path.join(args.dump, "model.pt"), map_location="cpu", weights_only=False)
    sc = torch.load(os.path.join(args.dump, "scale.pt"), map_location="cpu", weights_only=False)
    sm = torch.load(os.path.join(args.dump, "smooth.pt"), map_location="cpu", weights_only=False)
    br = torch.load(os.path.join(args.dump, "branch.pt"), map_location="cpu", weights_only=False)
    fuse_flag = bool(sm.get("proj.fuse_when_possible", True))
    print(f"proj.fuse_when_possible = {fuse_flag}")

    blocks = sorted({m.group(0) for k in br
                     for m in [re.match(r".*transformer_blocks\.\d+", k)] if m})
    specs = []
    for b in blocks:
        specs += [
            (f"{b}.attn1.to_q", f"{b}.attn1.to_q", 0, False),
            (f"{b}.attn1.to_k", f"{b}.attn1.to_q", 1, False),
            (f"{b}.attn1.to_v", f"{b}.attn1.to_q", 2, False),
            (f"{b}.attn1.to_out.0", f"{b}.attn1.to_out.0", 0, True),
            (f"{b}.attn2.to_q", f"{b}.attn2.to_q", 0, False),
            (f"{b}.attn2.to_out.0", f"{b}.attn2.to_out.0", 0, True),
            (f"{b}.ff.net.0.proj", f"{b}.ff.net.0.proj", 0, False),
            (f"{b}.ff.net.2", f"{b}.ff.net.2", 0, False),
        ]
    out = {}
    for lpath, bkey, slot, is_out in tqdm(specs, desc="linear", dynamic_ncols=True):
        W = sd[f"{lpath}.weight"].float().cuda()
        oc, ic = W.shape
        top = sc[f"{lpath}.weight.scale.0"].float().view(()).cuda()
        micro = sc[f"{lpath}.weight.scale.1"].float().view(oc, -1).cuda()
        zero = sc.get(f"{lpath}.weight.zero")
        assert zero is None or float(zero) == 0.0, lpath
        s = sm[bkey].float().cuda()
        a = br[bkey]["a.weight"].float().cuda()
        b = br[bkey]["b.weight"].float().cuda()
        assert b.shape[0] % oc == 0
        b_slice = b[slot * oc:(slot + 1) * oc]
        smooth_fused = is_out and fuse_flag
        if smooth_fused:
            s_kernel, a_raw = torch.ones_like(s), a
        else:
            s_kernel, a_raw = s, a / s.unsqueeze(0)
        packed = pack_layer(W.half(), top, micro,
                            a_raw.half(), b_slice.half(), s_kernel.half(),
                            "plain", "cuda")
        bias = sd.get(f"{lpath}.bias")
        bias_packed = None
        if bias is not None:
            bp = bias.detach().float().view(-1).cuda()
            bias_packed = bp[pack_perm_vector(oc).cuda()].half().cpu()
        out[lpath] = {
            "type": "linear",
            "qweight": packed["qweight"], "wscales": packed["wscales"],
            "smooth": packed["smooth"].half(), "smooth_orig": packed["smooth_orig"].half(),
            "lora_down": packed["lora_down"].half(), "lora_up": packed["lora_up"].half(),
            "wtscale": float(packed["wtscale"].float()),
            "bias": bias_packed,
        }

    n_conv = 0
    for key in sd:
        if re.search(r"resnets\.\d+\.conv[12]\.weight$", key):
            cpath = key[: -len(".weight")]
            out[cpath] = {"type": "conv", "weight": sd[key].half()}
            n_conv += 1
    # their pipeline splits up-block concat convs into ConcatConv(convs.0/1);
    # no shift keys in this recipe, biases unchanged -> concat along in-dim
    for key in sd:
        m = re.search(r"(.*resnets\.\d+\.conv[12])\.convs\.0\.weight$", key)
        if m:
            cpath = m.group(1)
            w = torch.cat([sd[f"{cpath}.convs.0.weight"],
                           sd[f"{cpath}.convs.1.weight"]], dim=1)
            out[cpath] = {"type": "conv", "weight": w.half()}
            n_conv += 1
    print(f"converted {len(out) - n_conv} linears + {n_conv} convs")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
