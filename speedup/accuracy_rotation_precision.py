"""
speedup/accuracy_rotation_precision.py

Per-layer SNR probe: how much accuracy do we lose if we run the LOW-REGION
rotation in int8 / int4 instead of bf16, before the existing int4 activation
quantization step?

Scheme summary (all share the same int4 activation quant + int4 weight quant
on the low region; only the rotation precision changes):

  A. bf16 rotation       (current DiRotQ — baseline)
  B. int8 rotation       (per-row x scale, per-col U scale)
  C. int8 rotation       (per-row-per-group x, per-col U)
  D. int4 rotation       (per-row x, per-col U)
  E. int4 rotation       (per-row-per-group x, per-col U)

For each, we measure SNR vs the fp16 reference matmul (no quantization
anywhere).

The high-region rotation stays bf16 in all schemes because it lands in the
fp16 output without further quantization — any precision drop there is
visible in the final y.

Reference noise floors:
  bf16 LSB at typical activation magnitude ≈ 35 dB SNR
  int8 quant ≈ 50 dB single-step
  int4 quant ≈ 17 dB single-step

So the question is whether int8/int4 rotation (which is followed by int4
activation quant) noticeably degrades the per-layer y compared to bf16
rotation followed by the same int4 act quant.
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

from speedup.kernels.triton_w4a4 import _quantize_int4_sym  # noqa: E402
from speedup.utils import gpu_name, write_json  # noqa: E402
from speedup.accuracy_qsnr import (  # noqa: E402
    make_realistic_input, make_weight, fit_pca_rotation, snr_db,
)


# ---------------------------------------------------------------------------
# Quant helpers
# ---------------------------------------------------------------------------

def _qd_int_rowwise(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row symmetric int quantize-dequantize.
       int8: maxq=127, int4: maxq=7."""
    maxq = (1 << (bits - 1)) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / maxq
    return (x / scale).round().clamp(-(maxq + 1), maxq) * scale


def _qd_int_rowwise_groupwise(x: torch.Tensor, bits: int, gs: int) -> torch.Tensor:
    """Per-row, per-group symmetric int quantize-dequantize."""
    maxq = (1 << (bits - 1)) - 1
    M, K = x.shape
    assert K % gs == 0
    Xg = x.reshape(M, K // gs, gs)
    scale = Xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / maxq
    return ((Xg / scale).round().clamp(-(maxq + 1), maxq) * scale).reshape(M, K)


def _qd_int_colwise(U: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-column symmetric int quantize-dequantize."""
    maxq = (1 << (bits - 1)) - 1
    scale = U.abs().amax(dim=0, keepdim=True).clamp(min=1e-9) / maxq
    return (U / scale).round().clamp(-(maxq + 1), maxq) * scale


def _qd_int4_groupwise(x: torch.Tensor, gs: int) -> torch.Tensor:
    """Activation int4 quant — per-row, per-group (matches DiRotQ runtime)."""
    codes, scales = _quantize_int4_sym(x, gs)
    M, K = x.shape
    return (codes.float().reshape(M, K // gs, gs) *
            scales.float().unsqueeze(-1)).reshape(M, K).to(x.dtype)


# ---------------------------------------------------------------------------
# Rotation+quant variants
# ---------------------------------------------------------------------------

def forward_split_rot(
    x: torch.Tensor, W: torch.Tensor, U: torch.Tensor,
    *, hlen: int, gs: int,
    rot_quant_low: str,           # 'bf16', 'int8', 'int4'
    x_quant_granularity: str = 'rowwise',  # 'rowwise' or 'rowwise_groupwise'
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Split-rotation forward: low-region rotation in chosen precision, high
    in bf16. Activation int4 quant of the low region happens AFTER rotation
    (same as current DiRotQ)."""
    K = x.shape[1]
    n_low = K - hlen
    U_low = U[:, :n_low]
    U_high = U[:, n_low:] if hlen > 0 else None

    # ---- High region rotation: always bf16 ----
    if hlen > 0:
        x_high_rot = x @ U_high
    else:
        x_high_rot = None

    # ---- Low region rotation in chosen precision ----
    if rot_quant_low == 'bf16':
        x_low_rot = x @ U_low
    else:
        bits = 8 if rot_quant_low == 'int8' else 4
        # Quantize x for the rotation
        if x_quant_granularity == 'rowwise':
            x_qd = _qd_int_rowwise(x.float(), bits)
        elif x_quant_granularity == 'rowwise_groupwise':
            x_qd = _qd_int_rowwise_groupwise(x.float(), bits, gs)
        else:
            raise ValueError(x_quant_granularity)
        U_low_qd = _qd_int_colwise(U_low.float(), bits)
        x_low_rot = x_qd @ U_low_qd

    # ---- Activation int4 quant on low rotated activation (DiRotQ baseline) ----
    # Cast back to compute dtype before the int4 quant + matmul so dtypes
    # are consistent across schemes (bf16 rotation produces bf16; int8/int4
    # rotation goes through fp32). For SNR what matters is the quantization
    # error, not the rotation dtype.
    x_low_q = _qd_int4_groupwise(x_low_rot.to(x.dtype).contiguous(), gs)

    # ---- Weight: precomputed in rotated basis, int4-quantized for low ----
    W_rot = W @ U
    W_low_q = _qd_int4_groupwise(W_rot[:, :n_low].contiguous(), gs)
    W_high = W_rot[:, n_low:].contiguous() if hlen > 0 else None

    # ---- Matmuls ----
    y_low = x_low_q @ W_low_q.t()
    if W_high is not None:
        y = y_low + (x_high_rot.to(x.dtype) @ W_high.t())
    else:
        y = y_low
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

# pixart-sigma layer shapes (subset that matters for the rotation question)
LAYERS = [
    ("attn1.to_q",       1152, 1152, 4096, 192),
    ("attn1.to_k",       1152, 1152, 4096, 192),
    ("attn1.to_v",       1152, 1152, 4096, 192),
    ("attn1.to_out.0",   1152, 1152, 4096, 128),
    ("attn2.to_q",       1152, 1152, 4096, 192),
    ("attn2.to_out.0",   1152, 1152, 4096, 128),
    ("ff.net.0.proj",    1152, 4608, 4096, 192),
]


SCHEMES = [
    # (name, kwargs to forward_split_rot)
    ("A: bf16 rotation (current)",  {"rot_quant_low": "bf16"}),
    ("B: int8 rot, x rowwise",       {"rot_quant_low": "int8", "x_quant_granularity": "rowwise"}),
    ("C: int8 rot, x rowgrp",        {"rot_quant_low": "int8", "x_quant_granularity": "rowwise_groupwise"}),
    ("D: int4 rot, x rowwise",       {"rot_quant_low": "int4", "x_quant_granularity": "rowwise"}),
    ("E: int4 rot, x rowgrp",        {"rot_quant_low": "int4", "x_quant_granularity": "rowwise_groupwise"}),
]


def _parse_args():
    p = argparse.ArgumentParser(description="Rotation-precision SNR probe")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gs", type=int, default=64)
    p.add_argument("--n-calib", type=int, default=2048)
    p.add_argument("--n-test", type=int, default=4096)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--outlier-frac", type=float, default=0.05)
    p.add_argument("--outlier-scale", type=float, default=4.0)
    p.add_argument("--distribution", choices=["axis-aligned", "rotated", "mixture", "all"],
                   default="rotated",
                   help="Default 'rotated' is the most representative of real activations.")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def _run_distribution(distribution: str, args, dtype, device):
    print(f"\n----- distribution: {distribution} -----")
    print(f"{'Layer':<22}{'shape':>14}{'M':>6}", end="")
    for sname, _ in SCHEMES:
        print(f"{sname.split(':')[0]:>10}", end="")
    print(f"{'  ΔB-A':>9}{'  ΔC-A':>9}{'  ΔD-A':>9}{'  ΔE-A':>9}")
    print(f"{'':<22}{'':>14}{'':>6}", end="")
    for _ in SCHEMES:
        print(f"{'(dB)':>10}", end="")
    print(f"{'  (dB)':>9}{'  (dB)':>9}{'  (dB)':>9}{'  (dB)':>9}")
    print("-" * (22 + 14 + 6 + 10 * len(SCHEMES) + 9 * 4))

    rows = []
    for layer_name, K, N, M, hlen in LAYERS:
        snr_per_scheme = {sname: [] for sname, _ in SCHEMES}
        for trial in range(args.n_trials):
            x_calib = make_realistic_input(
                args.n_calib, K, dtype=dtype, device=device,
                outlier_frac=args.outlier_frac, outlier_scale=args.outlier_scale,
                distribution=distribution, seed=1000 * trial + 1,
            )
            x_test = make_realistic_input(
                args.n_test, K, dtype=dtype, device=device,
                outlier_frac=args.outlier_frac, outlier_scale=args.outlier_scale,
                distribution=distribution, seed=1000 * trial + 2,
            )
            W = make_weight(N, K, dtype=dtype, device=device, seed=1000 * trial + 3)
            U = fit_pca_rotation(x_calib, hlen)

            y_ref = F.linear(x_test, W)

            for sname, kwargs in SCHEMES:
                y_q = forward_split_rot(x_test, W, U, hlen=hlen, gs=args.gs, **kwargs)
                snr_per_scheme[sname].append(snr_db(y_ref, y_q))

        avg = lambda lst: sum(lst) / len(lst)
        snrs = {n: avg(snr_per_scheme[n]) for n, _ in SCHEMES}

        # Print row
        shape = f"{K}x{N}"
        print(f"{layer_name:<22}{shape:>14}{M:>6}", end="")
        for sname, _ in SCHEMES:
            print(f"{snrs[sname]:>10.2f}", end="")
        # Deltas vs scheme A
        s_a = snrs[SCHEMES[0][0]]
        for sname, _ in SCHEMES[1:]:
            d = snrs[sname] - s_a
            sgn = '+' if d >= 0 else ''
            print(f"{sgn}{d:>8.2f}", end="")
        print()

        rows.append({
            "layer": layer_name, "K": K, "N": N, "M": M, "hlen": hlen,
            "snr": snrs,
            "delta_vs_A": {n: snrs[n] - s_a for n, _ in SCHEMES[1:]},
        })

    # Block average
    print("-" * (22 + 14 + 6 + 10 * len(SCHEMES) + 9 * 4))
    avg_snrs = {n: sum(r["snr"][n] for r in rows) / len(rows) for n, _ in SCHEMES}
    print(f"{'Block average':<22}{'':>14}{'':>6}", end="")
    for sname, _ in SCHEMES:
        print(f"{avg_snrs[sname]:>10.2f}", end="")
    sa = avg_snrs[SCHEMES[0][0]]
    for sname, _ in SCHEMES[1:]:
        d = avg_snrs[sname] - sa
        sgn = '+' if d >= 0 else ''
        print(f"{sgn}{d:>8.2f}", end="")
    print()

    return {"rows": rows,
            "block_avg": avg_snrs,
            "block_avg_delta": {n: avg_snrs[n] - sa for n, _ in SCHEMES[1:]}}


def main():
    args = _parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device

    print("=== Rotation-precision SNR probe ===")
    print(f"GPU       : {gpu_name()}")
    print(f"dtype     : {args.dtype}, gs={args.gs}")
    print(f"calib/test: {args.n_calib}/{args.n_test}, trials: {args.n_trials}")
    print(f"outliers  : {int(args.outlier_frac*100)}% at {args.outlier_scale}× scale")
    print(f"\nSchemes:")
    for sname, kwargs in SCHEMES:
        print(f"  {sname:<35}{kwargs}")

    distributions = (["axis-aligned", "rotated", "mixture"]
                     if args.distribution == "all" else [args.distribution])

    results = {}
    for d in distributions:
        results[d] = _run_distribution(d, args, dtype, device)

    print()
    print("How to read:")
    print("  ΔB-A = SNR(int8 rot, x rowwise) − SNR(bf16 rot)")
    print("  Δ ≥ −0.5 dB  → safe; rotation precision drop is invisible at the y-level")
    print("  Δ ∈ [−1.5, −0.5] → likely safe; recommend image-quality verification")
    print("  Δ < −1.5 dB → noticeable; need per-row-per-group or stick with bf16 rotation")

    out = {
        "gpu": gpu_name(),
        "model": "pixart-sigma",
        "dtype": args.dtype,
        "n_calib": args.n_calib,
        "n_test": args.n_test,
        "n_trials": args.n_trials,
        "outlier_frac": args.outlier_frac,
        "outlier_scale": args.outlier_scale,
        "groupsize": args.gs,
        "schemes": [{"name": n, "kwargs": k} for n, k in SCHEMES],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "by_distribution": results,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = out_dir / f"accuracy_rotation_precision{tag}.json"
    write_json(fname, out)
    print(f"\nWrote {fname}")


if __name__ == "__main__":
    main()
