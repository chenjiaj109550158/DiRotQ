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
    ap.add_argument("--smooth-pt", default=None,
                    help="PLAN_ROUND2 S: SVDQuant smooth.pt; hooks whose mean "
                         "layer gain (--gains json) exceeds --threshold are "
                         "quantized in the smoothed domain")
    ap.add_argument("--gains", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--gptq-cov", default=None,
                    help="PLAN_ROUND2 G: alternate covariance file (collected "
                         "in the act-quantized input domain) used for GPTQ; "
                         "the H-SVD basis always uses the raw --cov")
    ap.add_argument("--gain-k", default=None,
                    help="PLAN_ROUND2 C: per-layer per-channel gain file from "
                         "fold_gain_correction.py; folds 1/k into lora_up "
                         "(pre-pack) and wcscales (post-pack)")
    ap.add_argument("--per-channel-top", action="store_true",
                    help="PLAN_ROUND3: per-output-channel top scale "
                         "(wcscales) instead of per-tensor (wtscale)")
    ap.add_argument("--smooth-alpha", type=float, default=1.0,
                    help="PLAN_ROUND3: smoothing strength s^alpha")
    ap.add_argument("--basis-diag", action="store_true",
                    help="THEORY Prop 4' ablation: diagonal basis metric "
                         "(GPTQ keeps the full H)")
    ap.add_argument("--grid", choices=["nvfp4", "mx"], default="nvfp4",
                    help="PLAN_MX: quantization grid. 'mx' = OCP MXFP4e2 "
                         "(group-32 E8M0), packed for the Triton "
                         "MXW4A4Linear runtime")
    args = ap.parse_args()
    assert not (args.per_channel_top and args.gain_k), "flags conflict on wcscales"

    from diffusers import PixArtTransformer2DModel

    model = PixArtTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float16
    )
    sd = model.state_dict()
    num_blocks = len(model.transformer_blocks)
    del model

    cov = torch.load(args.cov, map_location="cpu", weights_only=False)
    cov_g = (torch.load(args.gptq_cov, map_location="cpu", weights_only=False)
             if args.gptq_cov else None)
    gain_k = (torch.load(args.gain_k, map_location="cpu", weights_only=False)
              if args.gain_k else None)

    table = layer_table(num_blocks)
    smoothed_hooks = {}
    if args.smooth_pt:
        sm = torch.load(args.smooth_pt, map_location="cpu", weights_only=False)
        gains = json.load(open(args.gains))
        by_hook = {}
        for lp, ck in table:
            skey = lp.replace(".to_k", ".to_q").replace(".to_v", ".to_q") \
                if ".attn1" in lp else lp
            if lp in gains:
                by_hook.setdefault(ck, []).append((gains[lp], skey))
        for ck, gl in by_hook.items():
            if sum(g for g, _ in gl) / len(gl) > args.threshold:
                smoothed_hooks[ck] = sm[gl[0][1]].float()
        print(f"smoothed hooks: {len(smoothed_hooks)}/{len(by_hook)}")

    out, qsnrs = {}, {}
    t0 = time.time()
    for wpath, ckey in tqdm(table, dynamic_ncols=True):
        W = sd[f"{wpath}.weight"].to("cuda", torch.float32)
        H = cov[ckey]
        Hb = torch.diag(H.diagonal()) if args.basis_diag else H
        D, lora_up = hsvd_basis(W, Hb, args.rank, "cuda", damping=args.hsvd_damping)
        W_res = W - lora_up @ D
        s = smoothed_hooks.get(ckey)
        if s is not None:
            s = s.to("cuda", torch.float32).pow(args.smooth_alpha)
            W_res = W_res * s.unsqueeze(0)
            if cov_g is not None:  # already collected in the Q(x/s) domain
                Hg = cov_g[ckey].to("cuda", torch.float32)
            else:
                Hg = H.to("cuda", torch.float32) / (s.unsqueeze(1) * s.unsqueeze(0))
        else:
            Hg = (cov_g[ckey] if cov_g is not None else H).to("cuda", torch.float32)
        clip_kw = {}
        if args.clip_search:
            lo, hi, n = args.clip_grid
            clip_kw = {"hdiag": Hg.diagonal().clone(),
                       "clip_ratios": torch.linspace(lo, hi, int(n))}
        if args.grid == "mx":
            from absorb_basis.mx_quant import mx_pack_weight, quantize_residual_mx
            W_q = quantize_residual_mx(W_res, Hg, "cuda", gptq=True,
                                       damp_pct=args.damp,
                                       block_size=args.block_size)
            codes, exps = mx_pack_weight(W_q)
            s_pack = (s if s is not None
                      else torch.ones(W.shape[1], device="cuda",
                                      dtype=torch.float32))
            out[wpath] = {
                "codes": codes.cpu(), "exps": exps.cpu(),
                "lora_down": D.t().half().cpu(),   # [ic, r]
                "lora_up": lora_up.half().cpu(),   # [oc, r]
                "smooth": s_pack.half().cpu(),
            }
            W_hat = (W_q / s.unsqueeze(0) if s is not None else W_q) + lora_up @ D
            err = (W_hat - W).pow(2).sum()
            qsnrs[wpath] = (10.0 * torch.log10(
                W.pow(2).sum() / err.clamp(min=1e-20))).item()
            continue
        kind = "qkv" if args.per_channel_top else "plain"
        W_q, top, micro = quantize_residual(
            W_res, Hg, kind, args.group_size, "cuda",
            gptq=True, damp_pct=args.damp, block_size=args.block_size,
            **clip_kw,
        )
        s_pack = (s if s is not None
                  else torch.ones(W.shape[1], device="cuda", dtype=torch.float32))
        inv_k = None
        if gain_k is not None and wpath in gain_k:
            inv_k = (1.0 / gain_k[wpath].to("cuda", torch.float32))
        lu_pack = lora_up * inv_k.unsqueeze(1) if inv_k is not None else lora_up
        packed = pack_layer(
            W_q.half(), top, micro, D.half(), lu_pack.half(),
            s_pack.half(), kind, "cuda",
        )
        if inv_k is not None:
            from absorb_basis.build_checkpoint import pack_perm_vector
            perm = pack_perm_vector(inv_k.shape[0]).cuda()
            packed["wcscales"] = inv_k[perm].half().cpu()
        # pack_layer emits bf16 side tensors (FLUX convention); PixArt runs fp16
        out[wpath] = {
            "qweight": packed["qweight"],
            "wscales": packed["wscales"],
            "smooth": packed["smooth"].half(),
            "smooth_orig": packed["smooth_orig"].half(),
            "lora_down": packed["lora_down"].half(),
            "lora_up": packed["lora_up"].half(),
            "wtscale": (float(packed["wtscale"].float())
                        if "wtscale" in packed else 1.0),
        }
        if args.per_channel_top:
            # fp16 wcscales underflow at raw top magnitudes (~1e-4): store the
            # normalized split wcscales = top/mean(top), wtscale = mean(top)
            from absorb_basis.build_checkpoint import pack_perm_vector
            mean_top = float(top.mean())
            out[wpath]["wcscales"] = (top / mean_top)[
                pack_perm_vector(top.shape[0]).cuda()].half().cpu()
            out[wpath]["wtscale"] = mean_top
        elif "wcscales" in packed:
            out[wpath]["wcscales"] = packed["wcscales"].half()
        W_hat = (W_q / s.unsqueeze(0) if s is not None else W_q) + lora_up @ D
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
