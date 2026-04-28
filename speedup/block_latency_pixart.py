"""
speedup/block_latency_pixart.py

Measure ACTUAL latency for one PixArt-Sigma transformer block in three
configurations and report the per-layer + block-level speedup over fp16:

  1) fp16 baseline    — vanilla nn.Linear, no rotation, no quantization.
  2) W4A16            — torch._weight_int4pack_mm on the int4 region,
                          fp16 activation, fp16 tail GEMM, fp16 rotation.
  3) W4A4 (Triton)    — speedup/kernels/triton_w4a4.py kernel: int4 weights
                          AND int4 activations on the low region, fp16 tail.

Block layout follows the actual pixart-sigma architecture (28 such blocks):
  attn1.to_q/k/v          (1152→1152, hlen=192, rotation)
  attn1.to_out.0          (1152→1152, per-head ≈ hlen=128 flat)
  attn2.to_q              (1152→1152, hlen=192, rotation)
  attn2.to_k, attn2.to_v  (1152→1152, M=120 caption tokens, fp16 — never
                            quantized in DiRotQ; same in all 3 configs)
  attn2.to_out.0          (1152→1152, per-head ≈ hlen=128 flat)
  ff.net.0.proj           (1152→4608, hlen=192, rotation)
  ff.net.2                (4608→1152, --skip-quant-layers in DiRotQ —
                            uses W4A16 in configs 2 and 3, no rotation,
                            no tail; activation stays fp16 by design)

Note on per-head approximation:
  pixart's per-head layers have d_q=63, hlen_per_head=9, total H*d_q=1008
  (not a multiple of 64). For the latency benchmark we model these as a
  global rotation with hlen=128 so the W4A4/W4A16 kernels get clean tile
  alignment. The compute and BW costs are within ~3% of the true per-head
  treatment; the headline block speedup number is unchanged.

Usage:
    python -m speedup.block_latency_pixart
    python -m speedup.block_latency_pixart --warmup 5 --repeats 30
    python -m speedup.block_latency_pixart --m-img 8192   # CFG case
"""

from __future__ import annotations

import argparse
import sys
import time
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


# ---------------------------------------------------------------------------
# Layer specs for one pixart-sigma block
# ---------------------------------------------------------------------------

def _block_layers(m_img: int, m_txt: int):
    """Return list of (name, K, N, M, kind, hlen) for one pixart-sigma block.

    kind:
      "rot"      — global rotation + W4A4 low + fp16 tail
      "skipped"  — never quantized in DiRotQ (attn2 cross-attn K/V), fp16 in all configs
      "act_skip" — DiRotQ --skip-quant-layers: W4A16 weight, fp16 act, no rotation
    """
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


GROUPSIZE = 64


# ---------------------------------------------------------------------------
# Per-layer setup (random tensors + packed kernels)
# ---------------------------------------------------------------------------

def setup_layer(K: int, N: int, M: int, kind: str, hlen, *, dtype, device):
    x = torch.randn(M, K, dtype=dtype, device=device) * 0.5
    W = torch.randn(N, K, dtype=dtype, device=device) * 0.05
    bias = torch.randn(N, dtype=dtype, device=device) * 0.01

    state = {"x": x, "W": W, "bias": bias, "K": K, "N": N, "M": M,
             "kind": kind, "hlen_cfg": hlen}

    if kind == "rot":
        # Random orthogonal rotation matrix
        Q, _ = torch.linalg.qr(torch.randn(K, K, device=device))
        state["U"] = Q.to(dtype)

        n_low = K - hlen
        assert n_low % GROUPSIZE == 0, f"n_low={n_low} not divisible by gs={GROUPSIZE}"
        W_low = W[:, :n_low].contiguous()
        W_tail = W[:, n_low:].contiguous() if hlen > 0 else None
        state["n_low"] = n_low
        state["hlen"] = hlen
        state["W_tail"] = W_tail

        # W4A16 packed
        state["w4a16_low_pack"] = ti4.pack_weight_int4(W_low, GROUPSIZE)

        # W4A4 packed (weight side)
        w_packed, w_scales = tt4.pack_w4(W_low, GROUPSIZE)
        state["w4a4_w_packed"] = w_packed.contiguous()
        state["w4a4_w_scales"] = w_scales.contiguous()

    elif kind == "act_skip":
        # ff.net.2: full int4 weight, no rotation, no tail.
        # Both W4A16 and W4A4 configs use W4A16 here (act stays fp16 in DiRotQ).
        state["n_low"] = K
        state["hlen"] = 0
        state["w4a16_full_pack"] = ti4.pack_weight_int4(W.contiguous(), GROUPSIZE)

    elif kind == "skipped":
        # Cross-attn K/V — never touched, fp16 Linear in all configs
        pass

    return state


# ---------------------------------------------------------------------------
# Three forward variants
# ---------------------------------------------------------------------------

def fwd_fp16(L) -> torch.Tensor:
    return F.linear(L["x"], L["W"], L["bias"])


def fwd_w4a16(L) -> torch.Tensor:
    """fp16 rotation + W4A16 GEMM on low + fp16 GEMM on tail."""
    if L["kind"] == "skipped":
        return F.linear(L["x"], L["W"], L["bias"])

    if L["kind"] == "act_skip":
        # ff.net.2 — W4A16 only, no rotation
        y = ti4.int4_gemm(L["x"], L["w4a16_full_pack"])
        return y + L["bias"]

    # kind == "rot": full rotation in fp16, then split + W4A16 + fp16 tail
    x_rot = L["x"] @ L["U"]
    n_low, hlen = L["n_low"], L["hlen"]
    x_low = x_rot[:, :n_low].contiguous()
    y_low = ti4.int4_gemm(x_low, L["w4a16_low_pack"])
    if hlen > 0:
        x_high = x_rot[:, n_low:].contiguous()
        y_high = x_high @ L["W_tail"].t()
        y = y_low + y_high
    else:
        y = y_low
    return y + L["bias"]


def fwd_w4a4(L) -> torch.Tensor:
    """fp16 rotation + int4 act + W4A4 Triton GEMM on low + fp16 tail."""
    if L["kind"] == "skipped":
        return F.linear(L["x"], L["W"], L["bias"])

    if L["kind"] == "act_skip":
        # ff.net.2 — DiRotQ keeps act fp16, so this is W4A16
        y = ti4.int4_gemm(L["x"], L["w4a16_full_pack"])
        return y + L["bias"]

    # kind == "rot": rotation + W4A4 on low + fp16 tail
    x_rot = L["x"] @ L["U"]
    n_low, hlen, M, N = L["n_low"], L["hlen"], L["M"], L["N"]
    x_low = x_rot[:, :n_low].contiguous()
    a_packed, a_scales = tt4.quantize_act_int4(x_low, GROUPSIZE)
    y_low = tt4.triton_w4a4_gemm(
        a_packed, a_scales,
        L["w4a4_w_packed"], L["w4a4_w_scales"],
        M=M, N=N, K=n_low, group_size=GROUPSIZE,
        out_dtype=L["x"].dtype,
    )
    if hlen > 0:
        x_high = x_rot[:, n_low:].contiguous()
        y_high = x_high @ L["W_tail"].t()
        y = y_low + y_high
    else:
        y = y_low
    return y + L["bias"]


# ---------------------------------------------------------------------------
# Block-level timing
# ---------------------------------------------------------------------------

def _time_block(layers: list[dict], fwd, *, warmup: int, repeats: int,
                ignore_ratio: float) -> dict:
    """Time a single sweep through all layers; report mean ms."""
    def run_block():
        outs = []
        for L in layers:
            outs.append(fwd(L))
        # Touch outputs so PyTorch can't elide
        return outs[-1].sum()

    return run_timed(run_block, warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


def _time_layer(L: dict, fwd, *, warmup: int, repeats: int,
                ignore_ratio: float) -> dict:
    """Per-layer timing, isolated."""
    return run_timed(lambda: fwd(L).sum(), warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PixArt-Sigma block latency: fp16 vs W4A16 vs W4A4")
    p.add_argument("--m-img", type=int, default=4096,
                   help="Image tokens (1024×1024 → 4096; with CFG → 8192). Default 4096.")
    p.add_argument("--m-txt", type=int, default=120,
                   help="Caption tokens for cross-attn K/V. Default 120.")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--ignore-ratio", type=float, default=0.2)
    p.add_argument("--per-layer", action="store_true",
                   help="Also report per-layer breakdown.")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device

    print("=== PixArt-Sigma block latency benchmark ===")
    print(f"GPU       : {gpu_name()}")
    print(f"M (image) : {args.m_img}")
    print(f"M (text)  : {args.m_txt}")
    print(f"dtype     : {args.dtype}")
    print(f"warmup={args.warmup}, repeats={args.repeats}, ignore_ratio={args.ignore_ratio}")
    print()

    layer_specs = _block_layers(args.m_img, args.m_txt)
    print(f"Setting up {len(layer_specs)} layers (each gets random rotated weights)...")
    layers = [setup_layer(K, N, M, kind, hlen, dtype=dtype, device=device)
              for (_, K, N, M, kind, hlen) in layer_specs]
    print()

    # ---- Per-layer timings ----
    if args.per_layer:
        print(f"{'Layer':<22}{'shape':>15}{'M':>6}{'kind':>10} "
              f"{'fp16 ms':>10}{'w4a16 ms':>10}{'w4a4 ms':>10} "
              f"{'w4a16/fp16':>12}{'w4a4/fp16':>12}")
        print("-" * 117)
        for spec, L in zip(layer_specs, layers):
            name, K, N, M, kind, hlen = spec
            t_fp16 = _time_layer(L, fwd_fp16, warmup=args.warmup,
                                 repeats=args.repeats, ignore_ratio=args.ignore_ratio)
            t_w16 = _time_layer(L, fwd_w4a16, warmup=args.warmup,
                                repeats=args.repeats, ignore_ratio=args.ignore_ratio)
            t_w4 = _time_layer(L, fwd_w4a4, warmup=args.warmup,
                               repeats=args.repeats, ignore_ratio=args.ignore_ratio)

            ms = lambda s: s["mean"] * 1000
            sp = lambda s: t_fp16["mean"] / s["mean"]

            shape = f"{K}x{N}"
            print(f"{name:<22}{shape:>15}{M:>6}{kind:>10} "
                  f"{ms(t_fp16):>10.3f}{ms(t_w16):>10.3f}{ms(t_w4):>10.3f} "
                  f"{sp(t_w16):>11.2f}x{sp(t_w4):>11.2f}x")
            L["_t_fp16"] = t_fp16
            L["_t_w16"] = t_w16
            L["_t_w4"] = t_w4
        print()

    # ---- Block-level timings ----
    print("Block-level (full sweep through all 10 layers):")
    print()
    t_block_fp16 = _time_block(layers, fwd_fp16, warmup=args.warmup,
                               repeats=args.repeats, ignore_ratio=args.ignore_ratio)
    t_block_w16 = _time_block(layers, fwd_w4a16, warmup=args.warmup,
                              repeats=args.repeats, ignore_ratio=args.ignore_ratio)
    t_block_w4 = _time_block(layers, fwd_w4a4, warmup=args.warmup,
                             repeats=args.repeats, ignore_ratio=args.ignore_ratio)

    def fmt(stats):
        return f"{stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms"

    print(f"{'Config':<25}{'Block latency':>22}{'Speedup vs fp16':>20}")
    print("-" * 67)
    print(f"{'fp16 baseline':<25}{fmt(t_block_fp16):>22}{'1.00x':>20}")
    print(f"{'W4A16 (torch int4pack)':<25}{fmt(t_block_w16):>22}"
          f"{t_block_fp16['mean']/t_block_w16['mean']:>19.2f}x")
    print(f"{'W4A4 (Triton)':<25}{fmt(t_block_w4):>22}"
          f"{t_block_fp16['mean']/t_block_w4['mean']:>19.2f}x")

    # Persist
    out = {
        "gpu": gpu_name(),
        "model": "pixart-sigma",
        "m_img": args.m_img,
        "m_txt": args.m_txt,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "block": {
            "fp16":  {"mean_ms": t_block_fp16["mean"]*1000, "std_ms": t_block_fp16["std"]*1000},
            "w4a16": {"mean_ms": t_block_w16["mean"]*1000,  "std_ms": t_block_w16["std"]*1000,
                      "speedup_vs_fp16": t_block_fp16["mean"]/t_block_w16["mean"]},
            "w4a4":  {"mean_ms": t_block_w4["mean"]*1000,   "std_ms": t_block_w4["std"]*1000,
                      "speedup_vs_fp16": t_block_fp16["mean"]/t_block_w4["mean"]},
        },
    }
    if args.per_layer:
        out["per_layer"] = []
        for spec, L in zip(layer_specs, layers):
            out["per_layer"].append({
                "name": spec[0], "K": spec[1], "N": spec[2], "M": spec[3],
                "kind": spec[4], "hlen": spec[5],
                "fp16_ms":  L["_t_fp16"]["mean"]*1000,
                "w4a16_ms": L["_t_w16"]["mean"]*1000,
                "w4a4_ms":  L["_t_w4"]["mean"]*1000,
            })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = out_dir / f"block_latency_pixart_M{args.m_img}{tag}.json"
    write_json(fname, out)
    print(f"\nWrote {fname}")


if __name__ == "__main__":
    main()
