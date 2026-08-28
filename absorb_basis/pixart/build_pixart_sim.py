"""Build DiRotQ-absorb-basis (H-SVD, rank 32, GPTQ, NVFP4 two-level grid,
down-absorb) simulated-quantization weights for PixArt-Sigma.

Quantized layers = the 224 block linears (attn1 q/k/v/out, attn2 q/out,
ff up/down), matching SVDQuant's pixart-sigma skips exactly (adaln_single,
caption_projection, proj_out, all embeds AND cross-attn to_k/to_v stay fp16;
the last is their "attn_add" skip, verified from their quantization log).

Output: dict {diffusers_module_path: {"W_q": fp16 [oc,ic] (dequantized residual
on the NVFP4 grid), "lora_down": fp16 [ic,32], "lora_up": fp16 [oc,32]}}.

Run in the svdquant env:
  python build_pixart_sim.py --cov <cov.pt> --out <sim.pt>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from absorb_basis.build_checkpoint import hsvd_basis, quantize_residual


def layer_table(num_blocks: int):
    t = []
    for i in range(num_blocks):
        p = f"transformer_blocks.{i}"
        c = f"block.{i}"
        t += [
            (f"{p}.attn1.to_q", f"{c}.attn1_qkv"),
            (f"{p}.attn1.to_k", f"{c}.attn1_qkv"),
            (f"{p}.attn1.to_v", f"{c}.attn1_qkv"),
            (f"{p}.attn1.to_out.0", f"{c}.attn1_out"),
            # NOTE: attn2.to_k / attn2.to_v (cross-attn KV over the projected
            # caption) are NOT quantized by SVDQuant on PixArt (their
            # "attn_add" skip) — verified from their quantization log
            # (224 = 28 x 8 layers). Keep them fp16 here for alignment.
            (f"{p}.attn2.to_q", f"{c}.attn2_q"),
            (f"{p}.attn2.to_out.0", f"{c}.attn2_out"),
            (f"{p}.ff.net.0.proj", f"{c}.ffn_up"),
            (f"{p}.ff.net.2", f"{c}.ffn_down"),
        ]
    return t


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    ap.add_argument("--cov", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()

    from diffusers import PixArtTransformer2DModel

    model = PixArtTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float16
    )
    sd = model.state_dict()
    num_blocks = len(model.transformer_blocks)
    del model

    cov = torch.load(args.cov, map_location="cpu", weights_only=False)

    out, qsnrs = {}, {}
    t0 = time.time()
    for wpath, ckey in tqdm(layer_table(num_blocks), dynamic_ncols=True):
        W = sd[f"{wpath}.weight"].to("cuda", torch.float32)
        H = cov[ckey]
        D, lora_up = hsvd_basis(W, H, args.rank, "cuda")
        W_res = W - lora_up @ D
        Hs = H.to("cuda", torch.float32)
        W_q, _, _ = quantize_residual(
            W_res, Hs, "plain", args.group_size, "cuda",
            gptq=True, damp_pct=args.damp, block_size=args.block_size,
        )
        out[wpath] = {
            "W_q": W_q.half().cpu(),
            "lora_down": D.t().half().cpu(),  # [ic, r]
            "lora_up": lora_up.half().cpu(),  # [oc, r]
        }
        W_hat = W_q + lora_up @ D
        err = (W_hat - W).pow(2).sum()
        qsnrs[wpath] = (10.0 * torch.log10(W.pow(2).sum() / err.clamp(min=1e-20))).item()

    print(f"quantized {len(out)} layers in {time.time()-t0:.0f}s")
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
