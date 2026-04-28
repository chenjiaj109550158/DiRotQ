"""
speedup/latency.py

End-to-end and per-step latency benchmark for DiRotQ, modeled on
nunchaku/app/flux.1/t2i/latency.py and the Sana counterpart. Compares any
combination of {fp16, dirotq-fake, dirotq-torch, dirotq-triton} against
each other on the same prompt and same generator.

Defaults target flux-dev (the model the user has the basis + cache for).

Usage:
    python -m speedup.latency \
        --model flux-dev \
        --precisions fp16 dirotq-fake dirotq-torch dirotq-triton \
        --mode end2end \
        --warmup-times 2 --test-times 5

Output:
    A printed comparison table and a JSON file under speedup/results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DIROTQ_ROOT = _HERE.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))

from speedup.utils import (  # noqa: E402
    VALID_PRECISIONS, capture_transformer_inputs, fmt_speedup, gpu_name,
    run_timed, setup_pipeline, write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiRotQ latency benchmark")
    p.add_argument("--model", default="flux-dev",
                   choices=["flux-dev", "flux-schnell", "pixart-sigma", "sana-1.6b"])
    p.add_argument("--precisions", nargs="+", default=list(VALID_PRECISIONS),
                   choices=list(VALID_PRECISIONS),
                   help="Which precisions to benchmark (default: all)")
    p.add_argument("--mode", choices=["end2end", "step"], default="end2end")
    p.add_argument("--warmup-times", type=int, default=2)
    p.add_argument("--test-times", type=int, default=5)
    p.add_argument("--ignore-ratio", type=float, default=0.2)
    p.add_argument("--prompt", default="A cat holding a sign that says hello world")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Override generation_params['num_inference_steps']")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--gptq", action="store_true",
                   help="Use the GPTQ quantized cache (default: RTN cache).")
    p.add_argument("--nvfp4", action="store_true",
                   help="Use the NF4 (nvfp4) cache (default: int4).")
    p.add_argument("--cache-path", default=None,
                   help="Override DiRotQ quantized cache path.")
    p.add_argument("--no-cpu-offload", action="store_true",
                   help="Disable model_cpu_offload (use single-GPU resident weights).")
    p.add_argument("--output-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None,
                   help="Optional tag appended to the output JSON filename.")
    return p.parse_args()


def _make_step_fn(pipe, prompt: str, gen_params: dict):
    """Returns a callable that runs the full pipeline once."""
    def fn():
        with torch.no_grad():
            pipe(prompt, **gen_params)
    return fn


def _make_transformer_step_fn(pipe, captured_args, captured_kwargs):
    def fn():
        with torch.no_grad():
            pipe.transformer(*captured_args, **captured_kwargs)
    return fn


def main() -> None:
    args = _parse_args()
    print(f"=== DiRotQ Latency Benchmark ===")
    print(f"GPU: {gpu_name()}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Precisions: {args.precisions}")
    print()

    results: dict = {
        "gpu": gpu_name(),
        "model": args.model,
        "mode": args.mode,
        "warmup": args.warmup_times,
        "repeats": args.test_times,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "runs": {},
    }

    for precision in args.precisions:
        print(f"--- Loading precision={precision} ---")
        bundle = setup_pipeline(
            args.model, precision,
            nvfp4=args.nvfp4, gptq=args.gptq,
            cache_path=args.cache_path,
            enable_cpu_offload=not args.no_cpu_offload,
        )
        gen_params = dict(bundle.generation_params)
        if args.num_inference_steps is not None:
            gen_params["num_inference_steps"] = args.num_inference_steps
        if args.height is not None:
            gen_params["height"] = args.height
        if args.width is not None:
            gen_params["width"] = args.width

        if args.mode == "end2end":
            fn = _make_step_fn(bundle.pipeline, args.prompt, gen_params)
        else:
            print("  Capturing transformer inputs ...")
            cap_gp = dict(gen_params)
            captured_args, captured_kwargs = capture_transformer_inputs(
                bundle.pipeline, cap_gp, dummy_prompt=args.prompt
            )
            fn = _make_transformer_step_fn(
                bundle.pipeline, captured_args, captured_kwargs)

        print(f"  Warmup ({args.warmup_times}) + timed ({args.test_times}) ...")
        stats = run_timed(
            fn,
            warmup=args.warmup_times,
            repeats=args.test_times,
            ignore_ratio=args.ignore_ratio,
        )
        results["runs"][precision] = {
            "mean_s": stats["mean"],
            "std_s": stats["std"],
            "min_s": stats["min"],
            "max_s": stats["max"],
            "raw": stats["raw"],
        }
        print(f"  → mean={stats['mean']:.4f}s std={stats['std']:.4f}s\n")

        # Drop everything before loading the next precision so we don't OOM.
        del bundle
        torch.cuda.empty_cache()

    # ---- Print summary table ----
    runs = results["runs"]
    ref = "fp16" if "fp16" in runs else next(iter(runs))
    t_ref = runs[ref]["mean_s"]

    print("\n=== Summary ===")
    header = f"{'Precision':<20} {'Mean (s)':>10} {'Std (s)':>10} {'Speedup vs ' + ref:>20}"
    print(header)
    print("-" * len(header))
    for prec, st in runs.items():
        sp = fmt_speedup(t_ref, st["mean_s"])
        print(f"{prec:<20} {st['mean_s']:>10.4f} {st['std_s']:>10.4f} {sp:>20}")
    print()

    # ---- Persist ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = f"latency_{args.model}_{args.mode}{tag}.json"
    out_path = out_dir / fname
    write_json(out_path, results)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
