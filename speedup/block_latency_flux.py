"""
speedup/block_latency_flux.py

Block-level latency benchmark for flux-dev. Mirrors block_latency_pixart but
adapted to flux's shapes and adds the int8-rotation config that wins at
flux's larger K.

flux-dev specs:
  hidden=3072, intermediate=12288, num_heads=24, head_dim=128
  num_double_layers=19, num_single_layers=38
  high_fraction=0.125 → high_len_hidden=384, high_len_down=1536

We benchmark ONE double block + ONE single block (representative). The full
flux-dev transformer has 19 double + 38 single = 57 such blocks per timestep.

Layer M (token count): 4608 (= 4096 image + 512 text). Cross-attn-style K/V
M=120 doesn't apply because flux uses joint attention (single concatenated
sequence).

Configs:
  1. fp16 baseline (cuBLAS)
  2. W4A16+PCA unfused (5 kernels per layer, current original DiRotQ)
  3. W4A4+PCA fused (cuBLAS bf16 rotation + 1 Triton fused W4A4 kernel —
     current best for pixart)
  4. W4A4+PCA fused with int8 rotation (NEW: replaces cuBLAS rotation
     with a tuned Triton int8 mma kernel; same fused W4A4 kernel after).

Skip layers in flux that DiRotQ keeps bf16 (modulator linears,
final norm/proj_out): NOT included in this block benchmark — they
sit OUTSIDE the rotated layers, would benefit from the W4A16 fix
similar to pixart's ff.net.2 but separately.
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
from speedup.kernels.fused_w4a4_pca import fused_w4a4_pca_forward  # noqa: E402
from speedup.kernels.fused_w4a4_pca_int8rot import (  # noqa: E402
    fused_w4a4_pca_int8rot_forward,
)
from speedup.kernels.int8_rotation import (  # noqa: E402
    quantize_U_int8, compute_scale_x,
)
from speedup.utils import gpu_name, run_timed, write_json  # noqa: E402


GROUPSIZE = 64
M_TOKENS = 4608  # 4096 image + 512 text


# ---------------------------------------------------------------------------
# Layer specs for flux-dev (one double block + one single block)
# ---------------------------------------------------------------------------

def _block_layers(m: int = M_TOKENS):
    """All quantized linear layers in one double + one single transformer block."""
    return [
        # Double block (image + text streams, 12 quantized linears)
        ("d.attn.to_q (img QKV)",      3072,  3072, m, "rot",  384),
        ("d.attn.to_k",                3072,  3072, m, "rot",  384),
        ("d.attn.to_v",                3072,  3072, m, "rot",  384),
        ("d.attn.add_q_proj (txt QKV)", 3072,  3072, m, "rot",  384),
        ("d.attn.add_k_proj",          3072,  3072, m, "rot",  384),
        ("d.attn.add_v_proj",          3072,  3072, m, "rot",  384),
        ("d.attn.to_out.0",            3072,  3072, m, "rot",  384),
        ("d.attn.to_add_out",          3072,  3072, m, "rot",  384),
        ("d.ff.net.0.proj (ff_up)",    3072, 12288, m, "rot",  384),
        ("d.ff.net.2 (ff_down)",      12288,  3072, m, "rot", 1536),
        ("d.ff_context.net.0.proj",    3072, 12288, m, "rot",  384),
        ("d.ff_context.net.2",        12288,  3072, m, "rot", 1536),
        # Single block (parallel attn + MLP, 6 quantized linears)
        ("s.attn.to_q",                3072,  3072, m, "rot",  384),
        ("s.attn.to_k",                3072,  3072, m, "rot",  384),
        ("s.attn.to_v",                3072,  3072, m, "rot",  384),
        ("s.proj_mlp",                 3072, 12288, m, "rot",  384),
        ("s.proj_out.linears.0",       3072,  3072, m, "rot",  384),
        ("s.proj_out.linears.1",      12288,  3072, m, "rot", 1536),
    ]


def setup_layer(K, N, M, kind, hlen, *, dtype, device, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=dtype, device=device) * 0.5
    W = torch.randn(N, K, dtype=dtype, device=device) * 0.05
    bias = torch.randn(N, dtype=dtype, device=device) * 0.01

    state = {"x": x, "W": W, "bias": bias, "K": K, "N": N, "M": M,
             "kind": kind, "hlen_cfg": hlen}

    # PCA rotation matrix (random orthogonal for benchmark)
    Q, _ = torch.linalg.qr(torch.randn(K, K, device=device))
    U = Q.to(dtype).contiguous()
    state["U"] = U

    n_low = K - hlen
    assert n_low % GROUPSIZE == 0, f"n_low={n_low} not divisible by gs={GROUPSIZE}"
    state["n_low"] = n_low
    state["hlen"] = hlen

    # Pre-rotated weight, split into low (int4-packed) + tail (bf16)
    W_rot = W @ U
    W_low = W_rot[:, :n_low].contiguous()
    W_tail = W_rot[:, n_low:].contiguous() if hlen > 0 else None
    state["W_rot"] = W_rot
    state["W_tail"] = W_tail
    state["w4a16_low_pack"] = ti4.pack_weight_int4(W_low, GROUPSIZE)
    w_p, w_s = tt4.pack_w4(W_low, GROUPSIZE)
    state["w4a4_w_packed"] = w_p.contiguous()
    state["w4a4_w_scales"] = w_s.contiguous()

    # Quantize U to int8 once for the int8-rotation path
    U_int8, scale_U = quantize_U_int8(U)
    state["U_int8"] = U_int8
    state["scale_U_int8"] = scale_U

    return state


# ---------------------------------------------------------------------------
# Forward variants
# ---------------------------------------------------------------------------

def fwd_fp16(L) -> torch.Tensor:
    return F.linear(L["x"], L["W"], L["bias"])


def fwd_w4a16_pca(L) -> torch.Tensor:
    """Dense PCA U + W4A16 main + fp16 tail (unfused, 5 kernels)."""
    x_rot = L["x"] @ L["U"]
    n_low, hlen = L["n_low"], L["hlen"]
    x_low = x_rot[:, :n_low].contiguous()
    y_low = ti4.int4_gemm(x_low, L["w4a16_low_pack"])
    if hlen > 0:
        x_high = x_rot[:, n_low:].contiguous()
        y_low = y_low + (x_high @ L["W_tail"].t())
    return y_low + L["bias"]


def fwd_w4a4_pca(L) -> torch.Tensor:
    """Dense PCA U + Triton W4A4 main + fp16 tail (unfused)."""
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


def fwd_w4a4_pca_fused(L) -> torch.Tensor:
    """cuBLAS bf16 rotation + fused Triton kernel (current best for pixart)."""
    return fused_w4a4_pca_forward(
        L["x"], L["U"],
        L["w4a4_w_packed"], L["w4a4_w_scales"],
        L["W_tail"] if L["hlen"] > 0 else None,
        L["bias"], gs=GROUPSIZE,
    )


def fwd_w4a4_pca_fused_int8rot(L) -> torch.Tensor:
    """int8 rotation kernel + fused Triton kernel (NEW: flux-targeted)."""
    return fused_w4a4_pca_int8rot_forward(
        L["x"], L["U_int8"], L["scale_U_int8"],
        L["w4a4_w_packed"], L["w4a4_w_scales"],
        L["W_tail"] if L["hlen"] > 0 else None,
        L["bias"], gs=GROUPSIZE,
    )


def fwd_w4a4_pca_unfused_int8rot(L) -> torch.Tensor:
    """int8 rotation + UNFUSED W4A4 path (separate kernels for quant/W4A4/tail/add).

    This is the simplest plug-and-play swap: replaces cuBLAS bf16 rotation
    in the unfused path with an int8-mma rotation. Same kernels for the
    W4A4 stage as the unfused baseline.
    """
    from speedup.kernels.int8_rotation import int8_rotation_forward
    x_rot = int8_rotation_forward(L["x"], L["U_int8"], L["scale_U_int8"])
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


CONFIGS = [
    ("fp16",                            fwd_fp16),
    ("W4A16+PCA (unfused)",             fwd_w4a16_pca),
    ("W4A4+PCA unfused, bf16 rot",      fwd_w4a4_pca),
    ("W4A4+PCA unfused, int8 rot",      fwd_w4a4_pca_unfused_int8rot),
    ("W4A4+PCA fused, bf16 rot",        fwd_w4a4_pca_fused),
    ("W4A4+PCA fused, int8 rot",        fwd_w4a4_pca_fused_int8rot),
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
    return run_timed(run, warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


def _time_layer(L, fwd, *, warmup, repeats, ignore_ratio):
    return run_timed(lambda: fwd(L).sum(), warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Flux-dev block latency benchmark")
    p.add_argument("--m-tokens", type=int, default=M_TOKENS,
                   help="Total tokens (image + text). flux-dev: 4608.")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--ignore-ratio", type=float, default=0.2)
    p.add_argument("--per-layer", action="store_true")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device

    print("=== flux-dev (1 double + 1 single block) latency benchmark ===")
    print(f"GPU       : {gpu_name()}")
    print(f"M tokens  : {args.m_tokens}")
    print(f"dtype     : {args.dtype}, gs={GROUPSIZE}")
    print(f"warmup={args.warmup}, repeats={args.repeats}, ignore_ratio={args.ignore_ratio}")
    print()

    layer_specs = _block_layers(args.m_tokens)
    print(f"Setting up {len(layer_specs)} layers...")
    layers = [setup_layer(K, N, M, kind, hlen, dtype=dtype, device=device, seed=42 + i)
              for i, (_, K, N, M, kind, hlen) in enumerate(layer_specs)]

    # Sanity check: int8 rotation produces same output as bf16 rotation within noise
    print("\nCorrectness sanity (int8 rotation vs bf16 baseline):")
    for L, spec in zip(layers[:3], layer_specs[:3]):
        y_ref = fwd_fp16(L)
        y_int8 = fwd_w4a4_pca_fused_int8rot(L)
        rel_err = ((y_ref.float() - y_int8.float()).abs().mean() /
                   y_ref.float().abs().mean()).item()
        print(f"  {spec[0]:<35} rel_err vs fp16: {rel_err:.4e}")
    print()

    # Per-layer
    if args.per_layer:
        col_w = 13
        cfg_names = [c[0] for c in CONFIGS]
        header = f"{'Layer':<32}{'shape':>14}" + "".join(f"{c[:11]+'(ms)':>{col_w}}" for c in cfg_names)
        print(header)
        print('-' * len(header))
        for spec, L in zip(layer_specs, layers):
            name, K, N, M, kind, hlen = spec
            stats = []
            for cfg_name, fn in CONFIGS:
                stats.append(_time_layer(L, fn, warmup=args.warmup,
                                         repeats=args.repeats,
                                         ignore_ratio=args.ignore_ratio))
            row = f"{name:<32}{f'{K}x{N}':>14}"
            for s in stats:
                row += f"{s['mean']*1000:>{col_w}.3f}"
            print(row)
        print()

    # Block-level
    print("Block-level (one double + one single block, sum of all 18 layers):")
    print()
    block_stats = {}
    for cfg_name, fn in CONFIGS:
        s = _time_block(layers, fn, warmup=args.warmup, repeats=args.repeats,
                        ignore_ratio=args.ignore_ratio)
        block_stats[cfg_name] = s

    t_fp16 = block_stats["fp16"]["mean"]
    print(f"{'Config':<35}{'Block latency':>22}{'Speedup vs fp16':>18}")
    print('-' * 75)
    for cfg_name, s in block_stats.items():
        ms = s["mean"] * 1000
        std = s["std"] * 1000
        sp = t_fp16 / s["mean"]
        print(f"{cfg_name:<35}{f'{ms:.3f} ± {std:.3f} ms':>22}{sp:>17.2f}x")

    out = {
        "gpu": gpu_name(),
        "model": "flux-dev",
        "m_tokens": args.m_tokens,
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
    fname = out_dir / f"block_latency_flux_M{args.m_tokens}{tag}.json"
    write_json(fname, out)
    print(f"\nWrote {fname}")


if __name__ == "__main__":
    main()
