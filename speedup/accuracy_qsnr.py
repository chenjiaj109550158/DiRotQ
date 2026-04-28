"""
speedup/accuracy_qsnr.py

Per-layer accuracy comparison: PCA-rotation vs Hadamard-rotation in DiRotQ's
mixed-precision (W4A4 low / fp16 tail) framework.

Why per-layer SNR is the right first metric:
  Image-quality metrics (FID, CLIP, image-reward) require running the full
  diffusion pipeline for hundreds of prompts and computing scores against
  reference images — minutes to hours of GPU time. Per-layer SNR is a
  fast proxy that catches most quantization-method regressions: if the
  per-layer output is within bf16 noise (SNR > 35 dB), the end-to-end
  pipeline almost always agrees within a fraction of an FID point. If
  per-layer SNR is degraded (< 25 dB), the end-to-end pipeline is
  definitely degraded too.

What this script does:
  1. For each pixart-sigma layer shape, draw `n_calib` calibration samples
     from a "transformer-realistic" distribution (gaussian + a few
     outlier-heavy channels) — this is the surrogate for activation traces
     from real calibration prompts.
  2. Fit the rotation matrix U:
       - PCA: U = eigenvectors of x_calib's covariance (matching DiRotQ's
              calibration recipe). hlen channels with the largest eigenvalues
              are kept fp16; the rest are int4-quantized with rotation.
       - Hadamard: U is the structured FWHT matrix (no fitting). The fp16
              tail size is determined structurally (K - largest_pow2 ≤ K).
  3. Draw `n_test` fresh samples from the same distribution and:
       - Compute fp16 reference: `y_ref = x @ W^T`
       - Compute each rotation+quantization scheme's output `y_q`
       - SNR(dB) = 20·log10( ||y_ref|| / ||y_ref − y_q|| )
  4. Aggregate across pixart-sigma layer shapes; report per-layer and per-block.

Caveats:
  - Calibration data is synthetic (gaussian + injected outliers). Real
    activations have richer correlation structure that PCA exploits — so
    PCA's advantage in real life is likely larger than what this script
    shows. Hadamard's data-independence means its SNR here is ≈ its real-life
    SNR; PCA's SNR here is a lower bound.
  - We use the SAME random sign flips per layer for Hadamard (deterministic).
    Different sign flips give slightly different SNR; we don't sweep.
  - This measures per-LAYER, not per-block. End-of-block error (after a chain
    of rotations and matmuls) accumulates differently. To extend, run a
    multi-layer chain with the same data and measure final SNR.

Usage:
    python -m speedup.accuracy_qsnr
    python -m speedup.accuracy_qsnr --outlier-scale 8.0    # heavier outliers
    python -m speedup.accuracy_qsnr --outlier-frac 0.10    # more outlier chans
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
from speedup.kernels.triton_w4a4 import _quantize_int4_sym  # noqa: E402
from speedup.utils import gpu_name, write_json  # noqa: E402
from speedup.hadamard_layer import (  # noqa: E402
    prepare_hadamard_layer, hadamard_dims,
)


# ---------------------------------------------------------------------------
# Synthetic calibration distribution
# ---------------------------------------------------------------------------

def make_realistic_input(
    M: int, K: int, *, dtype, device,
    outlier_frac: float = 0.05,
    outlier_scale: float = 4.0,
    distribution: str = "axis-aligned",
    seed: int | None = None,
):
    """Synthetic activation distribution.

    Three modes:
      "axis-aligned": specific channel indices have larger variance. Closest
            to "outlier channel" intuition; benefits no-rotation+sort.
      "rotated": draw axis-aligned outliers, then apply a random orthogonal
            rotation to mix them across all channels. Outliers now live in
            arbitrary linear combinations of channels — Hadamard's regime.
      "mixture": small fraction of *samples* (rows) have all-channel high
            variance. Different shape of outlier; tests Hadamard's
            row-spreading property.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(M, K, dtype=torch.float32, device=device, generator=gen) * 0.5
    else:
        gen = None
        x = torch.randn(M, K, dtype=torch.float32, device=device) * 0.5

    n_outliers = max(1, int(K * outlier_frac))
    if distribution == "axis-aligned":
        outlier_idx = torch.arange(K, device=device)[:n_outliers]
        x[:, outlier_idx] = x[:, outlier_idx] * outlier_scale

    elif distribution == "rotated":
        outlier_idx = torch.arange(K, device=device)[:n_outliers]
        x[:, outlier_idx] = x[:, outlier_idx] * outlier_scale
        # Random orthogonal rotation
        if gen is not None:
            R = torch.randn(K, K, device=device, generator=gen)
        else:
            R = torch.randn(K, K, device=device)
        Q, _ = torch.linalg.qr(R)
        x = x @ Q  # outliers now live in linear combinations of all channels

    elif distribution == "mixture":
        # 5% of rows have outlier_scale× variance everywhere
        n_outlier_rows = max(1, int(M * outlier_frac))
        x[:n_outlier_rows] = x[:n_outlier_rows] * outlier_scale

    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    return x.to(dtype)


def make_weight(N: int, K: int, *, dtype, device, seed: int | None = None):
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(N, K, dtype=torch.float32, device=device, generator=gen)
                * 0.05).to(dtype)
    return (torch.randn(N, K, dtype=torch.float32, device=device) * 0.05).to(dtype)


# ---------------------------------------------------------------------------
# Quant helpers — fake-quant (dequant-back-to-fp) for SNR measurement.
#
# We use fake-quant rather than the real Triton kernel because:
#   1. The Triton kernel's output exactly matches `dequant·dequant·matmul`
#      modulo bf16 mma noise (verified earlier in convert_to_int4_ckpt's
#      `verify_equivalence`), so SNR is the same.
#   2. Fake-quant lets us isolate the quantization error from kernel
#      implementation noise.
# ---------------------------------------------------------------------------

def _quantize_dequantize(x: torch.Tensor, gs: int) -> torch.Tensor:
    codes, scales = _quantize_int4_sym(x, gs)
    M, K = x.shape
    return (codes.float().reshape(M, K // gs, gs) *
            scales.float().unsqueeze(-1)).reshape(M, K).to(x.dtype)


# ---------------------------------------------------------------------------
# Rotation schemes
# ---------------------------------------------------------------------------

def fit_pca_rotation(x_calib: torch.Tensor, hlen: int) -> torch.Tensor:
    """Compute K×K PCA basis from calibration data.

    Returns U so that `x @ U` rotates x into the PCA basis. The columns are
    ordered so that the *last* `hlen` columns correspond to the highest-variance
    directions (matching DiRotQ's "fp16 tail goes at the end" convention).
    """
    K = x_calib.shape[1]
    x_c = x_calib.float()
    x_c = x_c - x_c.mean(dim=0, keepdim=True)
    cov = (x_c.t() @ x_c) / max(x_c.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # eigenvalues ascending; columns of eigvecs match. We want last `hlen`
    # columns of U to be the high-variance directions, so simply use eigvecs
    # as is (ascending → tail = highest).
    return eigvecs.to(x_calib.dtype)


def forward_fp16(x: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None = None
                 ) -> torch.Tensor:
    return F.linear(x, W, bias)


def forward_pca_w4a4(
    x: torch.Tensor, W: torch.Tensor, U: torch.Tensor,
    *, hlen: int, gs: int, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense PCA rotation + W4A4 low + fp16 tail (fake-quant)."""
    K = x.shape[1]
    n_low = K - hlen
    x_rot = x @ U
    W_rot = W @ U
    x_low_q = _quantize_dequantize(x_rot[:, :n_low].contiguous(), gs)
    W_low_q = _quantize_dequantize(W_rot[:, :n_low].contiguous(), gs)
    y_low = x_low_q @ W_low_q.t()
    if hlen > 0:
        y_high = x_rot[:, n_low:] @ W_rot[:, n_low:].t()
        y = y_low + y_high
    else:
        y = y_low
    if bias is not None:
        y = y + bias
    return y


def forward_hadamard_w4a4_fakequant(
    x: torch.Tensor, W: torch.Tensor,
    *, gs: int, bias: torch.Tensor | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Hadamard rotation + W4A4 low + fp16 tail (fake-quant).

    Builds packed dict via prepare_hadamard_layer (same calibration used by
    the latency path) and runs a fake-quant version for SNR comparison.
    """
    packed = prepare_hadamard_layer(W, group_size=gs, seed=seed,
                                     use_dense_matmul=True)
    low_dim = packed["low_dim"]
    hlen = packed["hlen"]
    H_dense = packed["H_dense"]
    sf = packed["sign_flips"]
    K = x.shape[1]

    # Apply Hadamard rotation to x (low region only)
    x_low_rot = x[:, :low_dim] @ H_dense
    # Apply same Hadamard rotation to W (low region) so x_rot @ W_rot.T = x @ W.T
    # in original basis. Recall: H_dense already includes diag(sf) on the input
    # side, i.e., H_dense = (FWHT) ⊙ sf along the "input" axis. To rotate W's
    # low region the equivalent way, multiply W[:, :low_dim] by H_dense too.
    W_low_rot = W[:, :low_dim] @ H_dense

    x_low_q = _quantize_dequantize(x_low_rot.contiguous(), gs)
    W_low_q = _quantize_dequantize(W_low_rot.contiguous(), gs)
    y_low = x_low_q @ W_low_q.t()

    if hlen > 0:
        x_high = x[:, low_dim:].contiguous()
        W_high = W[:, low_dim:].contiguous()
        y = y_low + x_high @ W_high.t()
    else:
        y = y_low
    if bias is not None:
        y = y + bias
    return y


def forward_no_rotation_w4a4(
    x: torch.Tensor, W: torch.Tensor,
    *, hlen: int, gs: int, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Baseline: NO rotation, just int4-quantize the low region (= channels
    sorted by variance to put high-variance ones in the fp16 tail).

    This shows what a DiRotQ-style mixed-precision design WITHOUT any rotation
    would achieve — useful to isolate how much of DiRotQ's accuracy gain is
    from the rotation vs from the mixed-precision split alone.
    """
    K = x.shape[1]
    n_low = K - hlen
    # Sort channels by variance, put high-var at end (the fp16 tail position).
    var_per_chan = x.float().var(dim=0)
    order = torch.argsort(var_per_chan)  # ascending → tail = highest
    x_perm = x[:, order]
    W_perm = W[:, order]

    x_low_q = _quantize_dequantize(x_perm[:, :n_low].contiguous(), gs)
    W_low_q = _quantize_dequantize(W_perm[:, :n_low].contiguous(), gs)
    y_low = x_low_q @ W_low_q.t()
    if hlen > 0:
        y_high = x_perm[:, n_low:] @ W_perm[:, n_low:].t()
        y = y_low + y_high
    else:
        y = y_low
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------

def snr_db(y_ref: torch.Tensor, y_q: torch.Tensor) -> float:
    """20 · log10( ||y_ref|| / ||y_ref − y_q|| ). Higher = closer to ref."""
    sig = y_ref.float().norm()
    err = (y_ref.float() - y_q.float()).norm()
    if err == 0:
        return float("inf")
    return float(20 * torch.log10(sig / err))


# ---------------------------------------------------------------------------
# Pixart-sigma layer specs (same as the latency benchmark)
# ---------------------------------------------------------------------------

LAYERS = [
    # (name, K, N, M, hlen_pca)
    ("attn1.to_q",       1152, 1152, 4096, 192),
    ("attn1.to_k",       1152, 1152, 4096, 192),
    ("attn1.to_v",       1152, 1152, 4096, 192),
    ("attn1.to_out.0",   1152, 1152, 4096, 128),  # per-head approximated as global
    ("attn2.to_q",       1152, 1152, 4096, 192),
    ("attn2.to_out.0",   1152, 1152, 4096, 128),
    ("ff.net.0.proj",    1152, 4608, 4096, 192),
    # ff.net.2 is act-skipped in pixart — full int4 weight, fp16 act, no rotation
    # We include it as "no rotation" both for PCA and Hadamard variants.
    ("ff.net.2",         4608, 1152, 4096, 0),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Per-layer SNR: PCA vs Hadamard rotation")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gs", type=int, default=64)
    p.add_argument("--n-calib", type=int, default=2048,
                   help="Calibration samples for PCA")
    p.add_argument("--n-test", type=int, default=4096,
                   help="Test samples (matmul M)")
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--outlier-frac", type=float, default=0.05)
    p.add_argument("--outlier-scale", type=float, default=4.0)
    p.add_argument("--distribution",
                   choices=["axis-aligned", "rotated", "mixture", "all"],
                   default="all",
                   help="Outlier distribution model. 'all' runs the comparison "
                        "across all three for a complete picture.")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def _run_one_distribution(distribution: str, args, dtype, device):
    """Run the SNR comparison for one outlier distribution. Returns rows + averages."""
    print(f"\n----- distribution: {distribution} -----")
    print(f"{'Layer':<22}{'shape':>14}{'M':>6}"
          f"{'no-rot':>10}{'PCA':>10}{'Hadamard':>10}"
          f"{'Δ (Had−PCA)':>13}")
    print(f"{'':<22}{'':>14}{'':>6}{'(dB)':>10}{'(dB)':>10}{'(dB)':>10}{'(dB)':>13}")
    print("-" * 85)

    rows = []
    for name, K, N, M, hlen_pca in LAYERS:
        snr_norot, snr_pca, snr_had = [], [], []
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

            y_ref = forward_fp16(x_test, W)

            y_norot = forward_no_rotation_w4a4(x_test, W, hlen=hlen_pca, gs=args.gs)
            snr_norot.append(snr_db(y_ref, y_norot))

            U = fit_pca_rotation(x_calib, hlen_pca)
            y_pca = forward_pca_w4a4(x_test, W, U, hlen=hlen_pca, gs=args.gs)
            snr_pca.append(snr_db(y_ref, y_pca))

            y_had = forward_hadamard_w4a4_fakequant(x_test, W, gs=args.gs,
                                                     seed=42 + trial)
            snr_had.append(snr_db(y_ref, y_had))

        avg = lambda lst: sum(lst) / len(lst)
        s_norot, s_pca, s_had = avg(snr_norot), avg(snr_pca), avg(snr_had)
        delta = s_had - s_pca
        shape = f"{K}x{N}"
        sign = "+" if delta >= 0 else ""
        print(f"{name:<22}{shape:>14}{M:>6}"
              f"{s_norot:>10.2f}{s_pca:>10.2f}{s_had:>10.2f}"
              f"{sign}{delta:>11.2f}")
        rows.append({"layer": name, "K": K, "N": N, "M": M, "hlen": hlen_pca,
                     "snr_norot_db": s_norot, "snr_pca_db": s_pca,
                     "snr_hadamard_db": s_had, "delta_db": delta})

    avg_norot = sum(r["snr_norot_db"] for r in rows) / len(rows)
    avg_pca = sum(r["snr_pca_db"] for r in rows) / len(rows)
    avg_had = sum(r["snr_hadamard_db"] for r in rows) / len(rows)
    print("-" * 85)
    print(f"{'Block average':<22}{'':>14}{'':>6}"
          f"{avg_norot:>10.2f}{avg_pca:>10.2f}{avg_had:>10.2f}"
          f"{'+' if avg_had-avg_pca >= 0 else ''}{avg_had-avg_pca:>11.2f}")
    return {"rows": rows,
            "block_average": {"no_rotation_db": avg_norot,
                              "pca_db": avg_pca,
                              "hadamard_db": avg_had}}


def main():
    args = _parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device

    print("=== Per-layer SNR: PCA-rotation vs Hadamard-rotation ===")
    print(f"GPU       : {gpu_name()}")
    print(f"dtype     : {args.dtype}, gs={args.gs}")
    print(f"calib     : {args.n_calib} samples, test : {args.n_test} samples")
    print(f"outliers  : {int(args.outlier_frac*100)}% at {args.outlier_scale}× scale")
    print(f"trials    : {args.n_trials}")

    distributions = (["axis-aligned", "rotated", "mixture"]
                     if args.distribution == "all" else [args.distribution])

    results = {}
    for d in distributions:
        results[d] = _run_one_distribution(d, args, dtype, device)
    print()
    print("How to read:")
    print("  no-rot  : sort channels by variance → fp16 tail; quantize the rest. No rotation.")
    print("  PCA     : DiRotQ's current design (data-fitted rotation).")
    print("  Hadamard: proposed change (data-independent FWHT rotation + signs).")
    print("  Δ > 0 means Hadamard is BETTER than PCA (rare; data-fitting usually wins).")
    print("  SNR > 35 dB ≈ within bf16 noise; 25-35 ≈ slight visible degradation;")
    print("  < 25 likely noticeable in image metrics.")
    print()

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
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "by_distribution": results,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = out_dir / f"accuracy_qsnr{tag}.json"
    write_json(fname, out)
    print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
