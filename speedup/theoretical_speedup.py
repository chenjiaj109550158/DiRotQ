"""
speedup/theoretical_speedup.py

Static FLOP / byte analysis of a DiRotQ-instrumented transformer.

For each ActQuantWrapper layer, given (in_features, out_features, high_bits_length,
group_size, has_rotation, use_hadamard) we compute:

  FP16 baseline cost:
      flops_fp16   = 2 * M * K * N
      bytes_fp16_w = 2 * K * N         (weight read)
      bytes_fp16_a = 2 * M * K         (act read)
      bytes_fp16_y = 2 * M * N         (output write)

  DiRotQ effective cost (mixed-precision low region + fp tail + rotation):
      flops_low    = 2 * M * K_low * N           in 4-bit MAC equivalent
      flops_tail   = 2 * M * K_tail * N
      flops_rot    = 2 * M * K * K           (dense rotation, U @ x)
                     OR 0 for per-head (already in W); Hadamard is O(M*K*log(K))
      bytes_low_w  = 0.5 * K_low * N         (int4 weight read)
                     + 2 * (K_low/gs) * N    (group scales fp16)
      bytes_low_a  = depends on backend:
                     W4A16: 2 * M * K_low (acts stay fp16)
                     W4A4 : 0.5 * M * K_low + 2 * M * K_low/gs (acts int4 + scales)
      bytes_tail_w = 2 * K_tail * N
      bytes_tail_a = 2 * M * K_tail
      bytes_y      = 2 * M * N
      bytes_rot    = 2 * K * K              (rotation matrix, but reused per token)

  Compute speedup (assuming int4 mma is M-bound at 2x int8 mma is 4x fp16):
      speedup_compute = flops_fp16 / (flops_low / int4_factor + flops_tail
                                       + flops_rot / fp_factor)
  Bandwidth speedup (memory-bound regime):
      speedup_bw = bytes_fp16_total / bytes_dirotq_total

  Effective speedup:
      Use min(compute, bandwidth) to model whichever is the bottleneck.

This is a *projected* speedup — it assumes peak utilization of int4/int8
tensor cores and doesn't model overheads like bin-packing, scale loads, or
launch latency. Use it for paper tables; cross-check with `latency.py` and
`layer_bench.py` for measured numbers.

Usage:
    python -m speedup.theoretical_speedup --model flux-dev --batch-tokens 4608
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DIROTQ_ROOT = _HERE.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))

from speedup.utils import gpu_name, setup_pipeline, write_json  # noqa: E402


# Hardware ratios. For a paper-style projection we use canonical numbers;
# tweak via CLI if needed for a specific GPU.
DEFAULT_RATIOS = {
    "int4_vs_fp16_compute": 4.0,   # int4 mma ≈ 2x int8 ≈ 4x fp16 on Ampere
    "int8_vs_fp16_compute": 2.0,   # int8 mma ≈ 2x fp16 (used by triton kernel)
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiRotQ theoretical speedup analysis")
    p.add_argument("--model", default="flux-dev")
    p.add_argument("--batch-tokens", type=int, default=4608,
                   help="M = total tokens per forward. flux-dev = 4096+512.")
    p.add_argument("--int4-compute-ratio", type=float,
                   default=DEFAULT_RATIOS["int4_vs_fp16_compute"],
                   help="int4 mma throughput / fp16 mma throughput (default 4x)")
    p.add_argument("--int8-compute-ratio", type=float,
                   default=DEFAULT_RATIOS["int8_vs_fp16_compute"],
                   help="int8 mma throughput / fp16 mma throughput (default 2x)")
    p.add_argument("--gptq", action="store_true")
    p.add_argument("--nvfp4", action="store_true")
    p.add_argument("--cache-path", default=None)
    p.add_argument("--no-cpu-offload", action="store_true",
                   help="Skip cpu offload while loading (for systems with VRAM headroom)")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-layer cost model
# ---------------------------------------------------------------------------

def _layer_costs(M: int, K: int, N: int, hlen: int, gs: int, *,
                 has_rotation: bool, has_per_head: bool, has_hadamard: bool,
                 head_dim: int = 0,
                 int4_ratio: float = 4.0, int8_ratio: float = 2.0) -> dict:
    K_low = max(K - hlen, 0)
    K_tail = hlen

    # FP16 baseline
    flops_fp16 = 2 * M * K * N
    bytes_fp16 = 2 * (K * N + M * K + M * N)

    # Rotation overhead (forward only — reverse is fused into W).
    if has_rotation and not has_per_head and not has_hadamard:
        # Dense KxK matmul applied to M tokens
        rot_flops = 2 * M * K * K
        rot_bytes = 2 * (K * K + M * K + M * K)  # U + x + x_rot
    elif has_per_head and head_dim > 0:
        # Block-diagonal: H × (head_dim × head_dim) matmul
        H = K // head_dim
        rot_flops = 2 * M * H * head_dim * head_dim
        rot_bytes = 2 * (H * head_dim * head_dim + 2 * M * K)
    elif has_hadamard:
        # FWHT is O(M * D log D). Roughly equivalent flops cost.
        log2D = max(1, int(math.log2(K)))
        rot_flops = M * K * log2D
        rot_bytes = 2 * (2 * M * K)
    else:
        rot_flops = 0
        rot_bytes = 0

    # ---- W4A16 (torch backend) ----
    # int4 weight read: 0.5 byte/elem; +scales overhead; act stays fp16.
    bytes_w4a16 = (
        0.5 * K_low * N + 2 * (K_low // gs) * N * 2  # int4 W + (scale, mid)
        + 2 * K_tail * N                              # fp16 tail W
        + 2 * M * K_low + 2 * M * K_tail              # fp16 acts (rotated)
        + 2 * M * N                                   # fp16 output
    )
    # int4 mma not used here (W4A16 uses dequant + fp16 mma typically), so
    # compute is same as fp16. Memory dominates.
    flops_w4a16 = (2 * M * K_low * N) + (2 * M * K_tail * N) + rot_flops

    # ---- W4A4 (triton backend) ----
    # Both W and act 4-bit in low region. Use int8 mma effective throughput
    # as the conservative model (int4 mma is gone on Hopper).
    bytes_w4a4 = (
        0.5 * K_low * N + 2 * (K_low // gs) * N         # int4 W + scale fp16
        + 2 * K_tail * N
        + 0.5 * M * K_low + 2 * M * (K_low // gs)        # int4 act + per-token scale
        + 2 * M * K_tail                                 # fp16 tail act
        + 2 * M * N
    )
    flops_w4a4 = (
        (2 * M * K_low * N) / int8_ratio    # low region on int8 mma
        + (2 * M * K_tail * N)              # tail on fp16
        + rot_flops
    )

    # Effective speedups: limited by the SLOWER of compute and bandwidth ratios.
    def _eff(flops_dirotq, bytes_dirotq):
        comp_sp = flops_fp16 / max(flops_dirotq, 1)
        bw_sp = bytes_fp16 / max(bytes_dirotq, 1)
        return {"compute_speedup": comp_sp, "bw_speedup": bw_sp,
                "effective_speedup": min(comp_sp, bw_sp),
                "flops": flops_dirotq, "bytes": bytes_dirotq}

    return {
        "M": M, "K": K, "N": N, "K_low": K_low, "K_tail": K_tail,
        "fp16": {"flops": flops_fp16, "bytes": bytes_fp16},
        "w4a16": _eff(flops_w4a16, bytes_w4a16),
        "w4a4":  _eff(flops_w4a4, bytes_w4a4),
        "rotation_flops": rot_flops,
    }


# ---------------------------------------------------------------------------
# Walk model and analyze
# ---------------------------------------------------------------------------

def analyze_transformer(transformer, *, M: int,
                        int4_ratio: float, int8_ratio: float) -> dict:
    sys.path.insert(0, str(_DIROTQ_ROOT))
    from utils.quant_utils import ActQuantWrapper

    per_layer: list[dict] = []
    total_fp16_flops = 0
    total_fp16_bytes = 0
    total_w4a16_flops = 0
    total_w4a16_bytes = 0
    total_w4a4_flops = 0
    total_w4a4_bytes = 0

    by_kind: dict[str, list[dict]] = defaultdict(list)

    for name, mod in transformer.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue
        if mod.quantizer.bits >= 16:
            continue

        K = mod.module.weight.shape[1]
        N = mod.module.weight.shape[0]
        hlen = int(mod.quantizer.high_bits_length)
        gs = int(mod.quantizer.groupsize) if mod.quantizer.groupsize > 0 else K

        has_rot = mod.rotation is not None
        has_per_head = mod.rotation_per_head is not None
        has_had = bool(getattr(mod, "use_hadamard", False))
        head_dim = int(getattr(mod, "head_dim", 0) or 0)

        c = _layer_costs(
            M, K, N, hlen, gs,
            has_rotation=has_rot, has_per_head=has_per_head,
            has_hadamard=has_had, head_dim=head_dim,
            int4_ratio=int4_ratio, int8_ratio=int8_ratio,
        )
        c["name"] = name
        per_layer.append(c)
        total_fp16_flops += c["fp16"]["flops"]
        total_fp16_bytes += c["fp16"]["bytes"]
        total_w4a16_flops += c["w4a16"]["flops"]
        total_w4a16_bytes += c["w4a16"]["bytes"]
        total_w4a4_flops += c["w4a4"]["flops"]
        total_w4a4_bytes += c["w4a4"]["bytes"]

        # Bucket by kind for the summary.
        kind = _bucket_name(name)
        by_kind[kind].append(c)

    def _agg(flops_d, bytes_d):
        return {
            "compute_speedup": total_fp16_flops / max(flops_d, 1),
            "bw_speedup": total_fp16_bytes / max(bytes_d, 1),
            "effective_speedup": min(total_fp16_flops / max(flops_d, 1),
                                       total_fp16_bytes / max(bytes_d, 1)),
        }

    return {
        "per_layer": per_layer,
        "by_kind": dict(by_kind),
        "totals": {
            "fp16_flops": total_fp16_flops,
            "fp16_bytes": total_fp16_bytes,
            "w4a16_flops": total_w4a16_flops,
            "w4a16_bytes": total_w4a16_bytes,
            "w4a4_flops": total_w4a4_flops,
            "w4a4_bytes": total_w4a4_bytes,
            "w4a16": _agg(total_w4a16_flops, total_w4a16_bytes),
            "w4a4":  _agg(total_w4a4_flops, total_w4a4_bytes),
        },
    }


def _bucket_name(layer_name: str) -> str:
    if "attn.to_out" in layer_name or "attn.to_add_out" in layer_name:
        return "attn_out"
    if "proj_out.linears.0" in layer_name:
        return "single_proj_out_attn"
    if "proj_out.linears.1" in layer_name:
        return "single_proj_out_mlp"
    if ".to_q" in layer_name or ".to_k" in layer_name or ".to_v" in layer_name \
            or "add_q_proj" in layer_name or "add_k_proj" in layer_name \
            or "add_v_proj" in layer_name:
        return "attn_qkv"
    if "proj_mlp" in layer_name:
        return "single_mlp_up"
    if ".net.0.proj" in layer_name:
        return "ff_up"
    if ".net.2" in layer_name:
        return "ff_down"
    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    print(f"=== DiRotQ Theoretical Speedup Analysis ===")
    print(f"GPU: {gpu_name()}")
    print(f"Model: {args.model}, M={args.batch_tokens}")
    print(f"Compute ratios: int4×{args.int4_compute_ratio} int8×{args.int8_compute_ratio} vs fp16")
    print()

    # Build a CPU-only transformer just for shape introspection.
    # setup_pipeline still loads weights; that's fine because we only need
    # the wrapped transformer's structure.
    bundle = setup_pipeline(
        args.model, "dirotq-fake",  # any DiRotQ precision; we only walk shapes
        nvfp4=args.nvfp4, gptq=args.gptq,
        cache_path=args.cache_path,
        enable_cpu_offload=not args.no_cpu_offload,
        require_cache=False,  # static analysis — actual weight values irrelevant
    )

    report = analyze_transformer(
        bundle.transformer,
        M=args.batch_tokens,
        int4_ratio=args.int4_compute_ratio,
        int8_ratio=args.int8_compute_ratio,
    )

    # ---- Summary by kind ----
    print(f"{'Kind':<22} {'#layers':>8} {'M':>6} {'K':>6} {'N':>6} "
          f"{'high':>6} {'W4A16 sp':>10} {'W4A4 sp':>10}")
    print("-" * 80)
    for kind, layers in report["by_kind"].items():
        n = len(layers)
        # average effective speedup across layers in this bucket
        avg_w16 = sum(l["w4a16"]["effective_speedup"] for l in layers) / n
        avg_w4 = sum(l["w4a4"]["effective_speedup"] for l in layers) / n
        rep = layers[0]
        print(f"{kind:<22} {n:>8} {rep['M']:>6} {rep['K']:>6} {rep['N']:>6} "
              f"{rep['K_tail']:>6} {avg_w16:>9.2f}x {avg_w4:>9.2f}x")

    print()
    t = report["totals"]
    print(f"{'TOTAL':<22}")
    print(f"  FP16 GFLOPs   : {t['fp16_flops']/1e9:.2f}")
    print(f"  W4A16 GFLOPs  : {t['w4a16_flops']/1e9:.2f}  "
          f"(compute {t['w4a16']['compute_speedup']:.2f}x, "
          f"bw {t['w4a16']['bw_speedup']:.2f}x → eff {t['w4a16']['effective_speedup']:.2f}x)")
    print(f"  W4A4  GFLOPs  : {t['w4a4_flops']/1e9:.2f}  "
          f"(compute {t['w4a4']['compute_speedup']:.2f}x, "
          f"bw {t['w4a4']['bw_speedup']:.2f}x → eff {t['w4a4']['effective_speedup']:.2f}x)")

    # ---- Persist ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"theoretical_{args.model}_M{args.batch_tokens}{tag}.json"
    write_json(out_path, {
        "gpu": gpu_name(),
        "model": args.model,
        "M": args.batch_tokens,
        "int4_ratio": args.int4_compute_ratio,
        "int8_ratio": args.int8_compute_ratio,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "report": report,
    })
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
