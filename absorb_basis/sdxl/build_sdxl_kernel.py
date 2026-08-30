"""Build the DiRotQ-absorb-basis kernel checkpoint for SDXL-Turbo.

- 560 transformer linears (attn1 q/k/v/out, attn2 q/out with cross-KV kept
  fp16 per SVDQuant's attn_add skip, GEGLU ff up/down): H-SVD rank-32 +
  GPTQ NVFP4, packed for SVDQW4A4Linear (all dims are 128-multiples, no
  padding needed). Supports --smooth-pt/--gains (PLAN_ROUND2 S) and
  --gptq-cov like the pixart builder.
- 34 resnet conv1/conv2: H-SVD rank-32 + GPTQ NVFP4 in the im2col domain;
  deployment mirrors nunchaku's SDXL semantics (fp16 conv executing the
  dequantized-grid + lora-corrected weight), so the output stores the fused
  fp16 conv weight W_hat = (Q(W_res) + lora).reshape(oc,ic,3,3).

Run in the svdquant env from the repo root:
  python absorb_basis/sdxl/build_sdxl_kernel.py --cov <lin.pt> --cov-conv <conv.pt> \
      --out <kernel.pt> --hsvd-damping 0.01
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from absorb_basis.build_checkpoint import hsvd_basis, pack_layer, quantize_residual
from absorb_basis.sdxl.collect_cov_sdxl import MODEL_ID


def linear_table(unet):
    t = []
    for name, mod in unet.named_modules():
        if re.search(r"transformer_blocks\.\d+$", name):
            t += [
                (f"{name}.attn1.to_q", f"{name}#attn1_qkv"),
                (f"{name}.attn1.to_k", f"{name}#attn1_qkv"),
                (f"{name}.attn1.to_v", f"{name}#attn1_qkv"),
                (f"{name}.attn1.to_out.0", f"{name}#attn1_out"),
                (f"{name}.attn2.to_q", f"{name}#attn2_q"),
                (f"{name}.attn2.to_out.0", f"{name}#attn2_out"),
                (f"{name}.ff.net.0.proj", f"{name}#ffn_up"),
                (f"{name}.ff.net.2", f"{name}#ffn_down"),
            ]
    return t


def conv_table(unet):
    return [name for name, _ in unet.named_modules()
            if re.search(r"resnets\.\d+\.conv[12]$", name)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--cov-conv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--hsvd-damping", type=float, default=0.01)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--smooth-pt", default=None)
    ap.add_argument("--gains", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--gptq-cov", default=None)
    ap.add_argument("--gain-k", default=None)
    args = ap.parse_args()

    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    )
    sd = unet.state_dict()
    lt = linear_table(unet)
    ct = conv_table(unet)
    del unet

    cov = torch.load(args.cov, map_location="cpu", weights_only=False)
    cov_c = torch.load(args.cov_conv, map_location="cpu", weights_only=False)
    cov_g = (torch.load(args.gptq_cov, map_location="cpu", weights_only=False)
             if args.gptq_cov else None)
    gain_k = (torch.load(args.gain_k, map_location="cpu", weights_only=False)
              if args.gain_k else None)

    smoothed = {}
    if args.smooth_pt:
        sm = torch.load(args.smooth_pt, map_location="cpu", weights_only=False)
        gains = json.load(open(args.gains))
        by_hook = {}
        for lp, ck in lt:
            if lp in gains and lp in sm:
                by_hook.setdefault(ck, []).append((gains[lp], lp))
        for ck, gl in by_hook.items():
            if sum(g for g, _ in gl) / len(gl) > args.threshold:
                smoothed[ck] = sm[gl[0][1]].float()
        print(f"smoothed hooks: {len(smoothed)}/{len(set(ck for _, ck in lt))}")

    out, qsnrs = {}, {}
    t0 = time.time()
    for wpath, ckey in tqdm(lt, desc="linear", dynamic_ncols=True):
        W = sd[f"{wpath}.weight"].to("cuda", torch.float32)
        H = cov[ckey]
        D, lora_up = hsvd_basis(W, H, args.rank, "cuda", damping=args.hsvd_damping)
        W_res = W - lora_up @ D
        s = smoothed.get(ckey)
        if s is not None:
            s = s.to("cuda", torch.float32)
            W_res = W_res * s.unsqueeze(0)
            Hg = (cov_g[ckey].to("cuda", torch.float32) if cov_g is not None
                  else H.to("cuda", torch.float32) / (s.unsqueeze(1) * s.unsqueeze(0)))
        else:
            Hg = (cov_g[ckey] if cov_g is not None else H).to("cuda", torch.float32)
        W_q, top, micro = quantize_residual(
            W_res, Hg, "plain", args.group_size, "cuda",
            gptq=True, damp_pct=args.damp, block_size=args.block_size,
        )
        s_pack = (s if s is not None
                  else torch.ones(W.shape[1], device="cuda", dtype=torch.float32))
        inv_k = None
        if gain_k is not None and wpath in gain_k:
            inv_k = (1.0 / gain_k[wpath].to("cuda", torch.float32))
        lu_pack = lora_up * inv_k.unsqueeze(1) if inv_k is not None else lora_up
        packed = pack_layer(W_q.half(), top, micro, D.half(), lu_pack.half(),
                            s_pack.half(), "plain", "cuda")
        out[wpath] = {
            "type": "linear",
            "qweight": packed["qweight"], "wscales": packed["wscales"],
            "smooth": packed["smooth"].half(), "smooth_orig": packed["smooth_orig"].half(),
            "lora_down": packed["lora_down"].half(), "lora_up": packed["lora_up"].half(),
            "wtscale": float(packed["wtscale"].float()),
        }
        if inv_k is not None:
            from absorb_basis.build_checkpoint import pack_perm_vector
            perm = pack_perm_vector(inv_k.shape[0]).cuda()
            out[wpath]["wcscales"] = inv_k[perm].half().cpu()
        W_hat = (W_q / s.unsqueeze(0) if s is not None else W_q) + lora_up @ D
        err = (W_hat - W).pow(2).sum()
        qsnrs[wpath] = (10.0 * torch.log10(W.pow(2).sum() / err.clamp(min=1e-20))).item()

    for cpath in tqdm(ct, desc="conv", dynamic_ncols=True):
        Wc = sd[f"{cpath}.weight"].to("cuda", torch.float32)
        oc, ic, kh, kw = Wc.shape
        W = Wc.reshape(oc, ic * kh * kw)
        H = cov_c[cpath]
        D, lora_up = hsvd_basis(W, H, args.rank, "cuda", damping=args.hsvd_damping)
        W_res = W - lora_up @ D
        W_q, top, micro = quantize_residual(
            W_res, H.to("cuda", torch.float32), "plain", args.group_size, "cuda",
            gptq=True, damp_pct=args.damp, block_size=args.block_size,
        )
        W_hat = (W_q + lora_up @ D).reshape(oc, ic, kh, kw)
        out[cpath] = {"type": "conv", "weight": W_hat.half().cpu()}
        err = (W_hat.reshape(oc, -1) - W).pow(2).sum()
        qsnrs[cpath] = (10.0 * torch.log10(W.pow(2).sum() / err.clamp(min=1e-20))).item()

    print(f"packed {len(out)} layers in {time.time()-t0:.0f}s")
    worst = sorted(qsnrs.items(), key=lambda kv: kv[1])
    for k, v in worst[:5]:
        print(f"  {v:6.2f} dB  {k}")
    print(f"median weight-QSNR: {worst[len(worst)//2][1]:.2f} dB")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    with open(args.out + ".qsnr.json", "w") as f:
        json.dump(qsnrs, f, indent=2)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
