#!/usr/bin/env python3
"""Sequential native-quantizer + BF16-epilogue FP4 pipeline benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any

import numpy as np
import torch

from .run_correctness import BF16_CUBINS, BUILD_ROOT, PIPELINE, source_bytes, validate_bf16_cubin
from ..optimized_gemm.benchmark import (
    make_benchmark_inputs, run_bf16_context, run_optimized, summarize, telemetry,
)
from ..optimized_gemm.run_correctness import write_runner_input


SHAPES = ((2048,2240,1952),(2048,2240,1960),(256,2240,1952))
CONFIGS = (("00","e2m1","e2xe2"),("01","e0m3","e0xe2"),("11","e0m3","e0xe0"))


def run_pipeline(shape: tuple[int,int,int], variant: str, activation_format: str,
                 activation: torch.Tensor, static_input: Path, root: Path,
                 warmup: int, iterations: int, rounds: int) -> dict[str, Any]:
    validate_bf16_cubin(variant)
    case = root / f"m{shape[0]}_n{shape[1]}_k{shape[2]}" / f"variant_{variant}"
    case.mkdir(parents=True,exist_ok=True)
    activation_path, output_path, report_path = case/"activation.bf16",case/"output.bf16",case/"runner.json"
    activation_path.write_bytes(source_bytes(activation))
    subprocess.run([str(PIPELINE),"--activation",str(activation_path),"--static-input",str(static_input),
        "--dtype","bf16","--format",activation_format,"--cubin",str(BF16_CUBINS[variant]),
        "--output",str(output_path),"--report",str(report_path),"--warmup",str(warmup),
        "--iterations",str(iterations),"--rounds",str(rounds)],check=True)
    raw=json.loads(report_path.read_text())
    absmax=summarize(raw["absmax_alpha_round_ms"]); pack=summarize(raw["quantize_pack_round_ms"])
    quant=summarize(raw["full_quantizer_round_ms"]); gemm=summarize(raw["gemm_bf16_round_ms"])
    pipeline=summarize(raw["full_pipeline_round_ms"])
    source_size=activation.numel()*activation.element_size()
    return {"variant":variant,"activation_format":activation_format,"logical_shape":list(shape),
        "padded_shape":raw["padded_shape"],"global_absmax_alpha":absmax,
        "quantize_pack_native":pack,"full_activation_quantization":quant,"gemm_bf16":gemm,
        "full_low_branch":pipeline,"quantizer_effective_gbps":source_size/(quant["median_ms"]*1e6),
        "output_canary_ok":raw["output_canary_ok"],"runner":raw}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--warmup",type=int,default=100)
    parser.add_argument("--iterations",type=int,default=500); parser.add_argument("--rounds",type=int,default=5)
    parser.add_argument("--output-root",type=Path,default=BUILD_ROOT/"benchmark")
    parser.add_argument("--report",type=Path,default=BUILD_ROOT/"benchmark.json"); args=parser.parse_args()
    if args.warmup<100 or args.iterations<500 or args.rounds<5: raise ValueError("required protocol is 100/500/5 or greater")
    before=telemetry(); results=[]
    for shape_index,shape in enumerate(SHAPES):
        activation=torch.randn((shape[0],shape[2]),generator=torch.Generator().manual_seed(8111+shape_index)).to(torch.bfloat16).contiguous()
        static=make_benchmark_inputs(shape,9201+shape_index)
        static_path=args.output_root/f"m{shape[0]}_n{shape[1]}_k{shape[2]}"/"static_input.bin"
        write_runner_input(static_path,static)
        pipelines={}
        for variant,fmt,name in CONFIGS:
            pipelines[variant]=run_pipeline(shape,variant,fmt,activation,static_path,args.output_root,args.warmup,args.iterations,args.rounds)
            pipelines[variant]["pairing"]=name
        canonical=run_optimized(static,"00",args.output_root/"canonical_scale_context",args.warmup,args.iterations,args.rounds)
        bf16=run_bf16_context(shape,args.warmup,args.iterations,args.rounds)
        transform=canonical["dynamic_a_scale_transform"]
        for variant in pipelines:
            q=pipelines[variant]["full_activation_quantization"]["median_ms"]
            pipelines[variant]["canonical_scale_plus_transform_estimate_ms"]=q+transform["median_ms"]
            pipelines[variant]["native_scale_saved_transform_ms"]=transform["median_ms"]
            pipelines[variant]["full_pipeline_vs_bf16_ratio"]=pipelines[variant]["full_low_branch"]["median_ms"]/bf16["median_ms"]
        e2q=pipelines["00"]["full_activation_quantization"]["median_ms"]
        pipelines["01"]["quantizer_ratio_vs_e2"]=pipelines["01"]["full_activation_quantization"]["median_ms"]/e2q
        pipelines["11"]["quantizer_ratio_vs_e2"]=pipelines["11"]["full_activation_quantization"]["median_ms"]/e2q
        results.append({"shape":list(shape),"pipelines":pipelines,"bf16_pytorch_context":bf16,
                        "canonical_scale_transform_context":transform})
    report={"scope":"quantizer + FP4 GEMM + BF16 output only; not a DiRotQ layer/model speedup",
            "timing":{"method":"CUDA events, normal sequential launches","warmup":args.warmup,
                      "iterations_per_round":args.iterations,"independent_rounds":args.rounds},
            "telemetry_before":before,"telemetry_after":telemetry(),"shapes":results}
    primary=results[:2]
    correctness_path=BUILD_ROOT/"correctness.json"
    correct=correctness_path.exists() and json.loads(correctness_path.read_text()).get("passed",False)
    faster=all(case["pipelines"][variant]["full_low_branch"]["median_ms"]<case["bf16_pytorch_context"]["median_ms"]
               for case in primary for variant in ("01","11"))
    e0_ratio=all(case["pipelines"][variant]["quantizer_ratio_vs_e2"]<=1.10 for case in primary for variant in ("01","11"))
    if not correct:
        report["classification"]="NATIVE QUANTIZER CORRECTNESS FAILED"
    elif faster and e0_ratio:
        report["classification"]="NATIVE E0 LOW-BRANCH PIPELINE FEASIBLE"
    else:
        report["classification"]="E0 GEMM FAST, QUANTIZER BOTTLENECK"
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"classification":report["classification"],"report":str(args.report.resolve())},indent=2))


if __name__=="__main__": main()
