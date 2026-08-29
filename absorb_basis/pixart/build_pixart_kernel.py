"""Build a REAL nunchaku-kernel checkpoint for DiRotQ-absorb-basis on
PixArt-Sigma (H-SVD basis, rank 32, GPTQ, NVFP4 two-level grid).

Unlike build_pixart_sim.py (which stores dequantized fp16 residuals for
fake-quant simulation), this packs every quantized layer into the exact
tensor layout consumed by nunchaku's SVDQW4A4Linear (fp4 fused
act-quant + lora + GEMM kernel) — verified bit-compatible on all PixArt
dims (1152/3456/4608).

Quantized layers = the same 224 block linears as the sim build (SVDQuant
skip alignment: adaln_single, caption_projection, proj_out, embeds AND
cross-attn to_k/to_v stay fp16). No smoothing (smooth = ones).

Output: dict {diffusers_module_path: {qweight int8 [oc,ic/2],
wscales e4m3 [ic/16,oc], smooth/smooth_orig fp16 [ic], lora_down fp16
[ic,32], lora_up fp16 [oc,32], wtscale fp32 scalar}}.

Run in the svdquant env from the repo root:
  python absorb_basis/pixart/build_pixart_kernel.py --cov <cov.pt> \
      --out <kernel.pt> --hsvd-damping 0.1
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

from absorb_basis.build_checkpoint import hsvd_basis, pack_layer, quantize_residual
from absorb_basis.pixart.build_pixart_sim import layer_table


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    ap.add_argument("--cov", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--hsvd-damping", type=float, default=0.1,
                    help="PixArt PLAN-B winner default")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--clip-search", action="store_true",
                    help="per-group Hessian-diag-weighted clip-ratio search "
                         "for the micro-scales (finer grid preserves small "
                         "high-frequency weight components)")
    ap.add_argument("--clip-grid", type=float, nargs=3, default=[0.8, 1.0, 21],
                    metavar=("LO", "HI", "N"))
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
        D, lora_up = hsvd_basis(W, H, args.rank, "cuda", damping=args.hsvd_damping)
        W_res = W - lora_up @ D
        Hg = H.to("cuda", torch.float32)
        clip_kw = {}
        if args.clip_search:
            lo, hi, n = args.clip_grid
            clip_kw = {"hdiag": Hg.diagonal().clone(),
                       "clip_ratios": torch.linspace(lo, hi, int(n))}
        W_q, top, micro = quantize_residual(
            W_res, Hg, "plain", args.group_size, "cuda",
            gptq=True, damp_pct=args.damp, block_size=args.block_size,
            **clip_kw,
        )
        packed = pack_layer(
            W_q.half(), top, micro, D.half(), lora_up.half(),
            torch.ones(W.shape[1], device="cuda", dtype=torch.float16),
            "plain", "cuda",
        )
        # pack_layer emits bf16 side tensors (FLUX convention); PixArt runs fp16
        out[wpath] = {
            "qweight": packed["qweight"],
            "wscales": packed["wscales"],
            "smooth": packed["smooth"].half(),
            "smooth_orig": packed["smooth_orig"].half(),
            "lora_down": packed["lora_down"].half(),
            "lora_up": packed["lora_up"].half(),
            "wtscale": float(packed["wtscale"].float()),
        }
        W_hat = W_q + lora_up @ D
        err = (W_hat - W).pow(2).sum()
        qsnrs[wpath] = (10.0 * torch.log10(W.pow(2).sum() / err.clamp(min=1e-20))).item()

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
