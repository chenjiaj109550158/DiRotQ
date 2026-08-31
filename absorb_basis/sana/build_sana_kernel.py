"""Build a REAL nunchaku-kernel checkpoint for DiRotQ-absorb-basis on
SANA-1.6B (ch5632 bf16): H-SVD basis (rank 32) + GPTQ NVFP4 residual,
packed into SVDQW4A4Linear layout.

SANA dims are not all multiples of the packer's 128-lane warp (hidden 2240),
so every layer is zero-padded to the aligned shape BEFORE quantization
("pad-first", validated bit-compatible on all SANA shapes); the runtime
wrapper pads the input and slices the output.

Quantized layers = 160 = 20 blocks x 8 (attn1 q/k/v/out, attn2 q/out with
cross-attn KV kept bf16 per SVDQuant's attn_add skip, GLUMBConv
conv_inverted/conv_point as 1x1-conv linears; conv_depth stays bf16).

Run in the svdquant env from the repo root:
  python absorb_basis/sana/build_sana_kernel.py --cov <cov.pt> --out <kernel.pt> \
      --hsvd-damping 0.01
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from absorb_basis.build_checkpoint import hsvd_basis, pack_layer, quantize_residual
from absorb_basis.sana.collect_cov_sana import MODEL_ID


def layer_table(num_blocks: int):
    t = []
    for i in range(num_blocks):
        p, c = f"transformer_blocks.{i}", f"block.{i}"
        t += [
            (f"{p}.attn1.to_q", f"{c}.attn1_qkv"),
            (f"{p}.attn1.to_k", f"{c}.attn1_qkv"),
            (f"{p}.attn1.to_v", f"{c}.attn1_qkv"),
            (f"{p}.attn1.to_out.0", f"{c}.attn1_out"),
            (f"{p}.attn2.to_q", f"{c}.attn2_q"),
            (f"{p}.attn2.to_out.0", f"{c}.attn2_out"),
            (f"{p}.ff.conv_inverted", f"{c}.ffn_up"),
            (f"{p}.ff.conv_point", f"{c}.ffn_down"),
        ]
    return t


def pad128(n: int) -> int:
    return (n + 127) // 128 * 128


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--hsvd-damping", type=float, default=0.01)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--smooth-pt", default=None,
                    help="PLAN_ROUND2 S: selective smoothing (see pixart builder)")
    ap.add_argument("--gains", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--gptq-cov", default=None,
                    help="PLAN_ROUND2 G: act-quant-domain covariances for GPTQ")
    ap.add_argument("--gain-k", default=None,
                    help="PLAN_ROUND2 C: per-layer per-channel gain file; "
                         "folds 1/k into lora_up (pre-pack) + wcscales")
    ap.add_argument("--per-channel-top", action="store_true",
                    help="PLAN_ROUND3: per-output-channel top scale")
    ap.add_argument("--smooth-alpha", type=float, default=1.0,
                    help="PLAN_ROUND3: smoothing strength s^alpha")
    args = ap.parse_args()
    assert not (args.per_channel_top and args.gain_k), "flags conflict on wcscales"

    from diffusers import SanaTransformer2DModel

    model = SanaTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        variant="bf16",
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
        if W.dim() == 4:  # 1x1 conv
            assert W.shape[2] == W.shape[3] == 1, wpath
            W = W.view(W.shape[0], W.shape[1])
        oc, ic = W.shape
        H = cov[ckey]
        D, lora_up = hsvd_basis(W, H, args.rank, "cuda", damping=args.hsvd_damping)
        W_res = W - lora_up @ D
        s = smoothed_hooks.get(ckey)
        if s is not None:
            s = s.to("cuda", torch.float32).pow(args.smooth_alpha)
            W_res = W_res * s.unsqueeze(0)
            if cov_g is not None:
                Hq = cov_g[ckey].to("cuda", torch.float32)
            else:
                Hq = H.to("cuda", torch.float32) / (s.unsqueeze(1) * s.unsqueeze(0))
        else:
            Hq = (cov_g[ckey] if cov_g is not None else H).to("cuda", torch.float32)
        oc_p, ic_p = pad128(oc), pad128(ic)
        W_res_p = F.pad(W_res, (0, ic_p - ic, 0, oc_p - oc))
        H_p = torch.zeros(ic_p, ic_p, dtype=torch.float32, device="cuda")
        H_p[:ic, :ic] = Hq
        H_p[range(ic, ic_p), range(ic, ic_p)] = float(Hq.diagonal().mean())
        kind = "qkv" if args.per_channel_top else "plain"
        W_qp, top, micro = quantize_residual(
            W_res_p, H_p, kind, args.group_size, "cuda",
            gptq=True, damp_pct=args.damp, block_size=args.block_size,
        )
        D_p = F.pad(D, (0, ic_p - ic))
        lu_p = F.pad(lora_up, (0, 0, 0, oc_p - oc))
        s_pack = (F.pad(s, (0, ic_p - ic), value=1.0) if s is not None
                  else torch.ones(ic_p, device="cuda", dtype=torch.float32))
        inv_k = None
        if gain_k is not None and wpath in gain_k:
            inv_k = F.pad(1.0 / gain_k[wpath].to("cuda", torch.float32),
                          (0, oc_p - oc), value=1.0)
            lu_p = lu_p * inv_k.unsqueeze(1)
        packed = pack_layer(
            W_qp.to(torch.bfloat16), top, micro,
            D_p.to(torch.bfloat16), lu_p.to(torch.bfloat16),
            s_pack.to(torch.bfloat16), kind, "cuda",
        )
        if inv_k is not None:
            from absorb_basis.build_checkpoint import pack_perm_vector
            packed["wcscales"] = inv_k[pack_perm_vector(oc_p).cuda()].to(torch.bfloat16).cpu()
        out[wpath] = {
            "qweight": packed["qweight"],
            "wscales": packed["wscales"],
            "smooth": packed["smooth"],
            "smooth_orig": packed["smooth_orig"],
            "lora_down": packed["lora_down"],
            "lora_up": packed["lora_up"],
            "wtscale": (float(packed["wtscale"].float())
                        if "wtscale" in packed else 1.0),
            "oc": oc, "ic": ic, "oc_p": oc_p, "ic_p": ic_p,
        }
        if args.per_channel_top:
            from absorb_basis.build_checkpoint import pack_perm_vector
            real_top = top.clone()
            mean_top = float(real_top[:oc].mean())  # exclude zero pad rows
            out[wpath]["wcscales"] = (real_top / mean_top)[
                pack_perm_vector(real_top.shape[0]).cuda()].to(torch.bfloat16).cpu()
            out[wpath]["wtscale"] = mean_top
        elif "wcscales" in packed:
            out[wpath]["wcscales"] = packed["wcscales"]
        Wq_raw = W_qp[:oc, :ic] / s.unsqueeze(0) if s is not None else W_qp[:oc, :ic]
        W_hat = Wq_raw + lora_up @ D
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
