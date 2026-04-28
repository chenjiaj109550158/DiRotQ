"""
speedup/block_latency_hadamard.py

Same as `speedup/block_latency_pixart.py` but with two additional configs:

  4) **W4A16 + Hadamard** — replace the dense PCA `U @ x` with structured
       FWHT (`x_low ⊙ sf` then `FWHT`) on the largest power-of-2 ≤ K. Same
       fp16 tail / W4A16 weight on the low region.
  5) **W4A4 + Hadamard** — same Hadamard rotation, but with the Triton W4A4
       kernel on the low region.

The dense-rotation configs (W4A16 with PCA U, W4A4 with PCA U) are kept for
side-by-side comparison.

The block layout still follows pixart-sigma. Per-head layers are flattened
to a global rotation with hlen=128 — same approximation as block_latency_pixart.

Usage:
    python -m speedup.block_latency_hadamard
    python -m speedup.block_latency_hadamard --per-layer
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_DIROTQ_ROOT = _HERE.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))

from speedup.kernels import torch_int4 as ti4  # noqa: E402
from speedup.kernels import triton_w4a4 as tt4  # noqa: E402
from speedup.utils import gpu_name, run_timed, write_json  # noqa: E402
from speedup.hadamard_layer import (  # noqa: E402
    prepare_hadamard_layer, forward_hadamard_w4a4, forward_hadamard_w4a16,
)
from speedup.kernels.fused_w4a4_pca import fused_w4a4_pca_forward  # noqa: E402
from speedup.kernels.fused_w4a16 import w4a16_fused_forward  # noqa: E402


# ---------------------------------------------------------------------------
# Layer specs (same as block_latency_pixart.py)
# ---------------------------------------------------------------------------

GROUPSIZE = 64


def _block_layers(m_img: int, m_txt: int):
    return [
        ("attn1.to_q",       1152, 1152, m_img, "rot",       192),
        ("attn1.to_k",       1152, 1152, m_img, "rot",       192),
        ("attn1.to_v",       1152, 1152, m_img, "rot",       192),
        ("attn1.to_out.0",   1152, 1152, m_img, "rot",       128),  # per-head approx
        ("attn2.to_q",       1152, 1152, m_img, "rot",       192),
        ("attn2.to_k",       1152, 1152, m_txt, "skipped",   None),
        ("attn2.to_v",       1152, 1152, m_txt, "skipped",   None),
        ("attn2.to_out.0",   1152, 1152, m_img, "rot",       128),  # per-head approx
        ("ff.net.0.proj",    1152, 4608, m_img, "rot",       192),
        ("ff.net.2",         4608, 1152, m_img, "act_skip",  None),
    ]


# ---------------------------------------------------------------------------
# Setup — produces tensors for all 5 configs in one go
# ---------------------------------------------------------------------------

def setup_layer(K, N, M, kind, hlen, *, dtype, device, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=dtype, device=device) * 0.5
    W = torch.randn(N, K, dtype=dtype, device=device) * 0.05
    bias = torch.randn(N, dtype=dtype, device=device) * 0.01

    state = {"x": x, "W": W, "bias": bias, "K": K, "N": N, "M": M,
             "kind": kind, "hlen_cfg": hlen}

    # Configs 2 & 3 (PCA-style dense rotation) — original block_latency_pixart logic
    if kind == "rot":
        Q, _ = torch.linalg.qr(torch.randn(K, K, device=device))
        state["U"] = Q.to(dtype)
        n_low = K - hlen
        assert n_low % GROUPSIZE == 0
        W_low = W[:, :n_low].contiguous()
        W_tail = W[:, n_low:].contiguous() if hlen > 0 else None
        state["n_low"] = n_low
        state["hlen"] = hlen
        state["W_tail"] = W_tail
        state["w4a16_low_pack"] = ti4.pack_weight_int4(W_low, GROUPSIZE)
        w_p, w_s = tt4.pack_w4(W_low, GROUPSIZE)
        state["w4a4_w_packed"] = w_p.contiguous()
        state["w4a4_w_scales"] = w_s.contiguous()

        # Configs 4 & 5 — Hadamard variant
        state["had"] = prepare_hadamard_layer(W, group_size=GROUPSIZE, seed=seed)

    elif kind == "act_skip":
        state["n_low"] = K
        state["hlen"] = 0
        state["w4a16_full_pack"] = ti4.pack_weight_int4(W.contiguous(), GROUPSIZE)
        # Also pack for the fused Triton W4A16 kernel.
        w_p, w_s = tt4.pack_w4(W.contiguous(), GROUPSIZE)
        state["w4a16_fused_packed"] = w_p.contiguous()
        state["w4a16_fused_scales"] = w_s.contiguous()
        # ff_down (in pixart) is act-skipped — no rotation to begin with, so
        # Hadamard config is *also* W4A16 (no rotation).
        state["had"] = None

    elif kind == "skipped":
        # Cross-attn K/V — fp16 in all configs
        state["had"] = None

    return state


# ---------------------------------------------------------------------------
# Forward variants (5 configs)
# ---------------------------------------------------------------------------

def fwd_fp16(L) -> torch.Tensor:
    return F.linear(L["x"], L["W"], L["bias"])


def fwd_w4a16_pca(L) -> torch.Tensor:
    """Dense PCA U + W4A16 main + fp16 tail."""
    if L["kind"] == "skipped":
        return F.linear(L["x"], L["W"], L["bias"])
    if L["kind"] == "act_skip":
        return ti4.int4_gemm(L["x"], L["w4a16_full_pack"]) + L["bias"]
    x_rot = L["x"] @ L["U"]
    n_low, hlen = L["n_low"], L["hlen"]
    x_low = x_rot[:, :n_low].contiguous()
    y_low = ti4.int4_gemm(x_low, L["w4a16_low_pack"])
    if hlen > 0:
        x_high = x_rot[:, n_low:].contiguous()
        y_low = y_low + (x_high @ L["W_tail"].t())
    return y_low + L["bias"]


def fwd_w4a4_pca(L) -> torch.Tensor:
    """Dense PCA U + Triton W4A4 main + fp16 tail."""
    if L["kind"] == "skipped":
        return F.linear(L["x"], L["W"], L["bias"])
    if L["kind"] == "act_skip":
        return ti4.int4_gemm(L["x"], L["w4a16_full_pack"]) + L["bias"]
    x_rot = L["x"] @ L["U"]
    n_low, hlen, M, N = L["n_low"], L["hlen"], L["M"], L["N"]
    x_low = x_rot[:, :n_low].contiguous()
    a_p, a_s = tt4.quantize_act_int4(x_low, GROUPSIZE)
    y_low = tt4.triton_w4a4_gemm(
        a_p, a_s, L["w4a4_w_packed"], L["w4a4_w_scales"],
        M=M, N=N, K=n_low, group_size=GROUPSIZE, out_dtype=L["x"].dtype,
    )
    if hlen > 0:
        x_high = x_rot[:, n_low:].contiguous()
        y_low = y_low + (x_high @ L["W_tail"].t())
    return y_low + L["bias"]


def fwd_w4a16_hadamard(L) -> torch.Tensor:
    """FWHT + W4A16 main + fp16 tail."""
    if L["kind"] == "skipped":
        return F.linear(L["x"], L["W"], L["bias"])
    if L["kind"] == "act_skip":
        return ti4.int4_gemm(L["x"], L["w4a16_full_pack"]) + L["bias"]
    return forward_hadamard_w4a16(L["x"], L["had"], L["bias"])


def fwd_w4a4_hadamard(L) -> torch.Tensor:
    """FWHT + Triton W4A4 main + fp16 tail."""
    if L["kind"] == "skipped":
        return F.linear(L["x"], L["W"], L["bias"])
    if L["kind"] == "act_skip":
        return ti4.int4_gemm(L["x"], L["w4a16_full_pack"]) + L["bias"]
    return forward_hadamard_w4a4(L["x"], L["had"], L["bias"])


def fwd_w4a4_pca_fused(L) -> torch.Tensor:
    """Dense PCA U + FUSED Triton kernel (rotation by cuBLAS, then everything
    else in one Triton kernel pass: int4 quant + W4A4 mma + fp16 tail + bias).

    For act_skip layers (ff.net.2): use the FUSED W4A16 Triton kernel
    instead of torch._weight_int4pack_mm.
    """
    if L["kind"] == "skipped":
        return F.linear(L["x"], L["W"], L["bias"])
    if L["kind"] == "act_skip":
        return w4a16_fused_forward(
            L["x"], L["w4a16_fused_packed"], L["w4a16_fused_scales"],
            L["bias"], gs=GROUPSIZE,
        )
    return fused_w4a4_pca_forward(
        L["x"], L["U"],
        L["w4a4_w_packed"], L["w4a4_w_scales"],
        L["W_tail"] if L["hlen"] > 0 else None,
        L["bias"],
        gs=GROUPSIZE,
    )


CONFIGS = [
    ("fp16",                fwd_fp16),
    ("W4A16+PCA",           fwd_w4a16_pca),
    ("W4A4+PCA",            fwd_w4a4_pca),
    ("W4A16+Hadamard",      fwd_w4a16_hadamard),
    ("W4A4+Hadamard",       fwd_w4a4_hadamard),
    ("W4A4+PCA fused",      fwd_w4a4_pca_fused),
]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_block(layers, fwd, *, warmup, repeats, ignore_ratio):
    def run():
        out = None
        for L in layers:
            out = fwd(L)
        return out.sum()
    return run_timed(run, warmup=warmup, repeats=repeats, ignore_ratio=ignore_ratio)


def _time_layer(L, fwd, *, warmup, repeats, ignore_ratio):
    return run_timed(lambda: fwd(L).sum(), warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Pixart block latency: fp16 vs PCA-U vs Hadamard")
    p.add_argument("--m-img", type=int, default=4096)
    p.add_argument("--m-txt", type=int, default=120)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--ignore-ratio", type=float, default=0.2)
    p.add_argument("--per-layer", action="store_true")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device

    print("=== PixArt-Sigma block latency: 5-way comparison ===")
    print(f"GPU       : {gpu_name()}")
    print(f"M (image) : {args.m_img}")
    print(f"M (text)  : {args.m_txt}")
    print(f"dtype     : {args.dtype}, gs={GROUPSIZE}")
    print(f"warmup={args.warmup}, repeats={args.repeats}, ignore_ratio={args.ignore_ratio}")
    print()

    layer_specs = _block_layers(args.m_img, args.m_txt)
    layers = [setup_layer(K, N, M, kind, hlen, dtype=dtype, device=device, seed=42 + i)
              for i, (_, K, N, M, kind, hlen) in enumerate(layer_specs)]

    # ---- Per-layer ----
    if args.per_layer:
        hdr_cfgs = [c[0] for c in CONFIGS]
        col_w = 11
        header = f"{'Layer':<22}{'shape':>15}{'M':>6}{'kind':>10} " + \
                 "".join(f"{c+' ms':>{col_w}}" for c in hdr_cfgs) + " | speedup vs fp16"
        print(header)
        print("-" * len(header))
        for spec, L in zip(layer_specs, layers):
            name, K, N, M, kind, hlen = spec
            stats = []
            for cfg_name, fn in CONFIGS:
                stats.append(_time_layer(L, fn, warmup=args.warmup,
                                         repeats=args.repeats,
                                         ignore_ratio=args.ignore_ratio))
            t_fp16 = stats[0]["mean"]
            row = f"{name:<22}{K}x{N:>9}{M:>6}{kind:>10} "
            for s in stats:
                row += f"{s['mean']*1000:>{col_w}.3f}"
            row += " |"
            for s in stats[1:]:
                row += f" {t_fp16/s['mean']:>5.2f}x"
            print(row)
        print()

    # ---- Block-level ----
    print("Block-level (full sweep through all 10 layers):")
    print()
    block_stats = {}
    for cfg_name, fn in CONFIGS:
        s = _time_block(layers, fn, warmup=args.warmup, repeats=args.repeats,
                        ignore_ratio=args.ignore_ratio)
        block_stats[cfg_name] = s

    t_fp16 = block_stats["fp16"]["mean"]
    print(f"{'Config':<22}{'Block latency':>22}{'Speedup vs fp16':>18}")
    print("-" * 62)
    for cfg_name, s in block_stats.items():
        ms = s["mean"] * 1000
        std = s["std"] * 1000
        sp = t_fp16 / s["mean"]
        print(f"{cfg_name:<22}{f'{ms:.3f} ± {std:.3f} ms':>22}{sp:>17.2f}x")

    # Persist
    out = {
        "gpu": gpu_name(),
        "model": "pixart-sigma",
        "m_img": args.m_img,
        "m_txt": args.m_txt,
        "dtype": args.dtype,
        "groupsize": GROUPSIZE,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "block": {
            cfg: {"mean_ms": s["mean"]*1000, "std_ms": s["std"]*1000,
                  "speedup_vs_fp16": t_fp16 / s["mean"]}
            for cfg, s in block_stats.items()
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = out_dir / f"block_latency_hadamard_M{args.m_img}{tag}.json"
    write_json(fname, out)
    print(f"\nWrote {fname}")


if __name__ == "__main__":
    main()
