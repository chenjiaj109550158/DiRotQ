"""
speedup/layer_bench.py

Per-layer microbenchmark for DiRotQ's mixed-precision linear layer.

For each layer-shape archetype found in flux-dev (QKV-img, QKV-txt, ff_up,
ff_down, attn_out, single_attn, single_proj_mlp, single_proj_out_mlp), build
a synthetic ActQuantWrapper-equivalent setup and time:

  - fp16  : nn.Linear baseline
  - fake  : DiRotQ fake-quant fused forward (current production code)
  - torch : W4A16 path via torch._weight_int4pack_mm (real kernel)
  - triton: W4A4 path via the Triton kernel in speedup/kernels/triton_w4a4.py

Results: a printed table per layer, plus a JSON dump under speedup/results/.
This script does NOT need a quantized cache — it generates random rotated
weights on the fly. Use latency.py for end-to-end, real-checkpoint timing.

Usage:
    python -m speedup.layer_bench --model flux-dev
    python -m speedup.layer_bench --model flux-dev --shapes ff_up ff_down
    python -m speedup.layer_bench --model flux-dev --batch-tokens 4608
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_DIROTQ_ROOT = _HERE.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))

from speedup.kernels import torch_int4 as ti4  # noqa: E402
from speedup.kernels import triton_w4a4 as tt4  # noqa: E402
from speedup.utils import fmt_speedup, gpu_name, run_timed, write_json  # noqa: E402


# Shape archetypes for each model. (in_features, out_features, label)
# For flux-dev, num_heads=24, head_dim=128, hidden=3072, intermediate=12288.
SHAPE_PRESETS: dict[str, list[tuple[str, int, int]]] = {
    "flux-dev": [
        ("attn_qkv_img",      3072, 3072),   # to_q/to_k/to_v
        ("attn_qkv_txt",      3072, 3072),   # add_q/k/v_proj
        ("attn_to_out",       3072, 3072),   # per-head — falls back to fake
        ("ff_up",             3072, 12288),
        ("ff_down",          12288, 3072),
        ("single_attn_qkv",   3072, 3072),
        ("single_proj_mlp",   3072, 12288),
        ("single_proj_out_mlp", 12288, 3072),
    ],
    "pixart-sigma": [
        ("attn_qkv",          1152, 1152),
        ("attn_to_out",       1152, 1152),
        ("ff_up",             1152, 4608),
        ("ff_down",           4608, 1152),
    ],
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiRotQ per-layer microbenchmark")
    p.add_argument("--model", default="flux-dev",
                   choices=list(SHAPE_PRESETS.keys()))
    p.add_argument("--shapes", nargs="*", default=None,
                   help="Subset of layer labels to run (default: all).")
    p.add_argument("--batch-tokens", type=int, default=4608,
                   help="M = batch_size * sequence_length (flux-dev: 4608 ≈ "
                        "4096 image tokens + 512 text tokens).")
    p.add_argument("--high-fraction", type=float, default=0.125,
                   help="Fraction of input channels kept fp16 (1 - low_frac).")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--ignore-ratio", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-layer benchmark
# ---------------------------------------------------------------------------

def _bench_fp16(M: int, K: int, N: int, dtype: torch.dtype, device: str,
                warmup: int, repeats: int, ignore_ratio: float) -> dict:
    x = torch.randn(M, K, dtype=dtype, device=device)
    W = torch.randn(N, K, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device)

    def fn():
        torch.nn.functional.linear(x, W, bias)

    return run_timed(fn, warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


def _bench_fake(M: int, K: int, N: int, hlen: int, gs: int,
                dtype: torch.dtype, device: str,
                warmup: int, repeats: int, ignore_ratio: float) -> dict:
    """Fake DiRotQ path: fp16 rotation + fp16 linear + STE int4 fake-quant.

    Reuses utils.quant_utils.ActQuantWrapper. We construct a wrapped Linear
    and assign a synthetic rotation, then call patch_forward_fast.
    """
    sys.path.insert(0, str(_DIROTQ_ROOT))
    from utils.quant_utils import ActQuantWrapper, ActQuantizer  # noqa: F401
    from dirotq_fused_unrotation_fast import _fused_forward_fast

    lin = nn.Linear(K, N, bias=True).to(device=device, dtype=dtype)
    wrapper = ActQuantWrapper(lin)
    # Synthetic rotation matrix on device in compute dtype
    U = torch.randn(K, K, device=device, dtype=dtype)
    U, _ = torch.linalg.qr(U.float())
    wrapper.rotation = U.to(dtype)
    wrapper.quantizer.configure(bits=4, groupsize=gs, sym=False,
                                clip_ratio=1.0, high_bits_length=hlen,
                                quant_dtype="int")
    wrapper._unrot_fused = True

    # Bind the fast fused forward (mirrors patch_forward_fast scope-locally)
    wrapper.forward = _fused_forward_fast.__get__(wrapper, ActQuantWrapper)

    x = torch.randn(1, M, K, dtype=dtype, device=device)

    def fn():
        wrapper(x)

    return run_timed(fn, warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


def _bench_torch_w4a16(M: int, K: int, N: int, hlen: int, gs: int,
                       dtype: torch.dtype, device: str,
                       warmup: int, repeats: int, ignore_ratio: float) -> dict:
    """Real W4A16 path via torch._weight_int4pack_mm + fp16 tail."""
    n_low = K - hlen
    if n_low % gs != 0:
        return {"error": f"n_low={n_low} not divisible by gs={gs}"}

    W_low = torch.randn(N, n_low, dtype=dtype, device=device)
    W_tail = torch.randn(N, hlen, dtype=dtype, device=device) if hlen > 0 else None
    bias = torch.randn(N, dtype=dtype, device=device)
    U = torch.randn(K, K, dtype=dtype, device=device)
    U, _ = torch.linalg.qr(U.float())
    U = U.to(dtype)

    try:
        packed = ti4.pack_weight_int4(W_low, gs)
    except RuntimeError as e:
        return {"error": str(e)}

    x = torch.randn(M, K, dtype=dtype, device=device)

    def fn():
        x_rot = (x @ U)
        x_low = x_rot[:, :n_low]
        y_low = ti4.int4_gemm(x_low, packed)
        if W_tail is not None and hlen > 0:
            x_high = x_rot[:, n_low:]
            y_high = x_high @ W_tail.t()
            y = y_low.to(dtype) + y_high
        else:
            y = y_low.to(dtype)
        y = y + bias

    return run_timed(fn, warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


def _bench_triton_w4a4(M: int, K: int, N: int, hlen: int, gs: int,
                       dtype: torch.dtype, device: str,
                       warmup: int, repeats: int, ignore_ratio: float) -> dict:
    """Real W4A4 path via the Triton kernel + fp16 tail."""
    if not tt4.is_supported():
        return {"error": "triton not installed"}
    n_low = K - hlen
    if n_low % gs != 0 or n_low % 2 != 0:
        return {"error": f"n_low={n_low} incompatible with gs={gs}"}

    W_low = torch.randn(N, n_low, dtype=dtype, device=device)
    W_tail = torch.randn(N, hlen, dtype=dtype, device=device) if hlen > 0 else None
    bias = torch.randn(N, dtype=dtype, device=device)
    U = torch.randn(K, K, dtype=dtype, device=device)
    U, _ = torch.linalg.qr(U.float())
    U = U.to(dtype)

    w_packed, w_scales = tt4.pack_w4(W_low, gs)
    w_packed = w_packed.contiguous()
    w_scales = w_scales.contiguous()

    x = torch.randn(M, K, dtype=dtype, device=device)

    def fn():
        x_rot = (x @ U)
        x_low = x_rot[:, :n_low].contiguous()
        a_packed, a_scales = tt4.quantize_act_int4(x_low, gs)
        y_low = tt4.triton_w4a4_gemm(
            a_packed, a_scales, w_packed, w_scales,
            M=M, N=N, K=n_low, group_size=gs, out_dtype=dtype)
        if W_tail is not None and hlen > 0:
            x_high = x_rot[:, n_low:]
            y_high = x_high @ W_tail.t()
            y = y_low + y_high
        else:
            y = y_low
        y = y + bias

    return run_timed(fn, warmup=warmup, repeats=repeats,
                     ignore_ratio=ignore_ratio)


def main() -> None:
    args = _parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    shapes = SHAPE_PRESETS[args.model]
    if args.shapes:
        keep = set(args.shapes)
        shapes = [s for s in shapes if s[0] in keep]

    M = args.batch_tokens
    print(f"=== DiRotQ Per-Layer Benchmark ===")
    print(f"GPU: {gpu_name()}")
    print(f"Model: {args.model}, M={M}, dtype={args.dtype}, gs={args.group_size}")
    print()

    results: dict = {
        "gpu": gpu_name(),
        "model": args.model,
        "M": M,
        "dtype": args.dtype,
        "group_size": args.group_size,
        "high_fraction": args.high_fraction,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "layers": {},
    }

    # ---- Header ----
    header = (f"{'Layer':<22} {'KxN':>16} "
              f"{'fp16(ms)':>10} {'fake(ms)':>10} {'torch(ms)':>11} "
              f"{'triton(ms)':>12} "
              f"{'fake/fp16':>10} {'torch/fp16':>11} {'triton/fp16':>12}")
    print(header)
    print("-" * len(header))

    for label, K, N in shapes:
        # Align high_bits_length to group_size, same as DiRotQ does.
        high_raw = int(round(args.high_fraction * K))
        high = ((high_raw + args.group_size - 1) // args.group_size) * args.group_size
        n_low = K - high

        layer_res: dict = {"K": K, "N": N, "high": high, "n_low": n_low}

        s_fp16 = _bench_fp16(M, K, N, dtype, args.device, args.warmup,
                             args.repeats, args.ignore_ratio)
        layer_res["fp16"] = s_fp16

        s_fake = _bench_fake(M, K, N, high, args.group_size, dtype, args.device,
                             args.warmup, args.repeats, args.ignore_ratio)
        layer_res["fake"] = s_fake

        s_torch = _bench_torch_w4a16(M, K, N, high, args.group_size, dtype,
                                      args.device, args.warmup, args.repeats,
                                      args.ignore_ratio)
        layer_res["torch"] = s_torch

        s_triton = _bench_triton_w4a4(M, K, N, high, args.group_size, dtype,
                                       args.device, args.warmup, args.repeats,
                                       args.ignore_ratio)
        layer_res["triton"] = s_triton

        def _ms(s: dict) -> float | None:
            return None if "error" in s else s["mean"] * 1000

        ms_fp16 = _ms(s_fp16)
        ms_fake = _ms(s_fake)
        ms_torch = _ms(s_torch)
        ms_triton = _ms(s_triton)

        def _fmt(ms):
            return f"{ms:.3f}" if isinstance(ms, float) else "n/a"

        def _sp(ms):
            if isinstance(ms, float) and ms > 0 and ms_fp16:
                return f"{ms_fp16 / ms:.2f}x"
            return "n/a"

        print(f"{label:<22} {f'{K}x{N}':>16} "
              f"{_fmt(ms_fp16):>10} {_fmt(ms_fake):>10} "
              f"{_fmt(ms_torch):>11} {_fmt(ms_triton):>12} "
              f"{_sp(ms_fake):>10} {_sp(ms_torch):>11} {_sp(ms_triton):>12}")

        results["layers"][label] = layer_res

    # Persist
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"layer_bench_{args.model}{tag}.json"
    write_json(out_path, results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
