#!/usr/bin/env python3
"""CUDA-event feasibility benchmark for the optimized static FP4 prototype.

The reported FP4 numbers are kernel-core/required-transform measurements, not
DiRotQ layer or model speedups.  Static B preparation is reported separately.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable

import numpy as np
import torch

from kernels.blackwell_e0_probe.e0_probe.run_e0_probe import CudaDriver
from kernels.blackwell_e0_probe.gemm_probe.run_gemm_probe import GemmModule
from kernels.blackwell_e0_probe.optimized_gemm.run_correctness import (
    BUILD_ROOT,
    CUBINS,
    RUNNER,
    validate_allowlisted_cubin,
    write_runner_input,
)
from kernels.blackwell_e0_probe.gemm_probe.packing import CanonicalInputs, GemmShape


MAIN_SHAPES = ((2048, 2240, 1952), (2048, 2240, 1960))
SENSITIVITY_SHAPE = (256, 2240, 1952)
VARIANTS = ("00", "01", "11")
PAIRING_NAMES = {"00": "e2xe2", "01": "e0xe2", "11": "e0xe0"}
ONE_WARP_CUBIN = BUILD_ROOT.parent / "gemm_probe" / "variant_00.cubin"


def make_benchmark_inputs(shape_tuple: tuple[int, int, int], seed: int) -> CanonicalInputs:
    shape = GemmShape(*shape_tuple)
    generator = np.random.default_rng(seed)
    a = generator.integers(0, 256, size=(shape.mp, shape.kp // 2), dtype=np.uint8)
    b = generator.integers(0, 256, size=(shape.np, shape.kp // 2), dtype=np.uint8)
    # Canonical K padding is positive-zero nibble.  Byte/nibble indexing is
    # explicit so K=1960 preserves the low eight valid nibbles of its block.
    for logical in range(shape.k, shape.kp):
        byte = logical // 2
        if logical & 1:
            a[:, byte] &= np.uint8(0x0F)
            b[:, byte] &= np.uint8(0x0F)
        else:
            a[:, byte] &= np.uint8(0xF0)
            b[:, byte] &= np.uint8(0xF0)
    scales_a = np.full((shape.mp, shape.kp // 16), 0x38, dtype=np.uint8)
    scales_b = np.full((shape.np, shape.kp // 16), 0x38, dtype=np.uint8)
    return CanonicalInputs(
        shape=shape,
        packed_a=torch.from_numpy(a.copy()),
        packed_b=torch.from_numpy(b.copy()),
        a_scales=torch.from_numpy(scales_a.copy()),
        b_scales=torch.from_numpy(scales_b.copy()),
        alpha_a=1.0,
        alpha_b=1.0,
    )


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(values)
    return {
        "raw_round_ms": values,
        "median_ms": statistics.median(values),
        "p10_ms": float(np.percentile(ordered, 10)),
        "p90_ms": float(np.percentile(ordered, 90)),
        "min_ms": min(values),
        "standard_deviation_ms": statistics.pstdev(values),
    }


def telemetry() -> dict[str, Any]:
    fields = (
        "timestamp,name,temperature.gpu,power.draw,clocks.gr,clocks.sm,"
        "clocks.mem,utilization.gpu,memory.used"
    )
    query = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    return {"gpu_query": query, "compute_processes": processes.splitlines() if processes else []}


def run_optimized(inputs: CanonicalInputs, variant: str, root: Path,
                  warmup: int, iterations: int, rounds: int) -> dict[str, Any]:
    validate_allowlisted_cubin(CUBINS[variant], variant)
    shape = inputs.shape
    case_root = root / f"m{shape.m}_n{shape.n}_k{shape.k}" / f"variant_{variant}"
    case_root.mkdir(parents=True, exist_ok=True)
    input_path = case_root / "input.bin"
    write_runner_input(input_path, inputs)
    runner_report = case_root / "runner.json"
    subprocess.run([
        str(RUNNER), "--cubin", str(CUBINS[variant]),
        "--input", str(input_path),
        "--output", str(case_root / "output_fp32.bin"),
        "--bf16-output", str(case_root / "output_bf16.bin"),
        "--report", str(runner_report),
        "--warmup", str(warmup), "--iterations", str(iterations),
        "--rounds", str(rounds),
    ], check=True)
    raw = json.loads(runner_report.read_text())
    gemm = summarize(raw["gemm_round_ms"])
    a_scale = summarize(raw["a_scale_transform_round_ms"])
    output = summarize(raw["output_bf16_cast_round_ms"])
    b_scale = summarize(raw["b_scale_prepack_round_ms"])
    m, n, k = shape.m, shape.n, shape.k
    mp, np_, kp = shape.mp, shape.np, shape.kp
    effective_tflops = 2 * m * n * k / (gemm["median_ms"] * 1e9)
    padded_tflops = 2 * mp * np_ * kp / (gemm["median_ms"] * 1e9)
    estimated = a_scale["median_ms"] + gemm["median_ms"] + output["median_ms"]
    canonical_storage = (
        raw["canonical_a_bytes"] + raw["canonical_b_bytes"]
        + raw["canonical_a_scale_bytes"] + raw["canonical_b_scale_bytes"]
    )
    native_storage = (
        raw["canonical_a_bytes"] + raw["canonical_b_bytes"]
        + raw["native_a_scale_bytes"] + raw["native_b_scale_bytes"]
    )
    return {
        "pairing": PAIRING_NAMES[variant],
        "variant": variant,
        "logical_shape": [m, n, k],
        "padded_shape": [mp, np_, kp],
        "gemm_only": gemm,
        "dynamic_a_payload_transform": {"required": False, "median_ms": 0.0},
        "dynamic_a_scale_transform": a_scale,
        "static_b_payload_prepack": {"required": False, "median_ms": 0.0},
        "static_b_scale_prepack": b_scale,
        "output_bf16_cast": output,
        "estimated_low_branch_total_ms": estimated,
        "effective_tflops": effective_tflops,
        "padded_tflops": padded_tflops,
        "effective_to_padded_work_ratio": (m * n * k) / (mp * np_ * kp),
        "canonical_storage_bytes": canonical_storage,
        "native_storage_bytes": native_storage,
        "storage_overhead_bytes": native_storage - canonical_storage,
        "storage_overhead_fraction": (native_storage / canonical_storage) - 1.0,
        "runner": raw,
    }


def _driver_event_rounds(driver: CudaDriver, launch: Callable[[], None],
                         warmup: int, iterations: int, rounds: int) -> list[float]:
    driver.lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    driver.lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    driver.lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
    driver.lib.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    driver.lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
    for _ in range(warmup):
        launch()
    driver._check(driver.lib.cuCtxSynchronize(), "cuCtxSynchronize(warmup)")
    start = ctypes.c_void_p()
    stop = ctypes.c_void_p()
    driver._check(driver.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)")
    driver._check(driver.lib.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate(stop)")
    values = []
    try:
        for _ in range(rounds):
            driver._check(driver.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                launch()
            driver._check(driver.lib.cuEventRecord(stop, None), "cuEventRecord(stop)")
            driver._check(driver.lib.cuEventSynchronize(stop), "cuEventSynchronize")
            elapsed = ctypes.c_float()
            driver._check(driver.lib.cuEventElapsedTime(ctypes.byref(elapsed), start, stop),
                          "cuEventElapsedTime")
            values.append(float(elapsed.value) / iterations)
    finally:
        driver.lib.cuEventDestroy_v2(start)
        driver.lib.cuEventDestroy_v2(stop)
    return values


def run_one_warp_reference(inputs: CanonicalInputs, warmup: int,
                           iterations: int, rounds: int) -> dict[str, Any]:
    arrays = [np.ascontiguousarray(value.numpy(), dtype=np.uint8) for value in (
        inputs.packed_a, inputs.packed_b, inputs.a_scales, inputs.b_scales
    )]
    shape = inputs.shape
    with CudaDriver() as driver, GemmModule(driver, ONE_WARP_CUBIN) as module:
        allocations = []
        try:
            for array in arrays:
                pointer = module._allocate(array.nbytes)
                allocations.append(pointer)
                driver._check(driver.lib.cuMemcpyHtoD_v2(
                    pointer.value, array.ctypes.data_as(ctypes.c_void_p), array.nbytes),
                    "cuMemcpyHtoD_v2(one-warp input)")
            output = module._allocate(shape.m * shape.n * 4)
            allocations.append(output)
            arguments = [
                ctypes.c_uint64(allocations[0].value), ctypes.c_uint64(allocations[1].value),
                ctypes.c_uint64(allocations[2].value), ctypes.c_uint64(allocations[3].value),
                ctypes.c_int(shape.m), ctypes.c_int(shape.n), ctypes.c_int(shape.kp),
                ctypes.c_float(inputs.alpha_a), ctypes.c_float(inputs.alpha_b),
                ctypes.c_uint64(output.value),
            ]
            parameters = (ctypes.c_void_p * len(arguments))(*[
                ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in arguments
            ])
            def launch() -> None:
                driver._check(driver.lib.cuLaunchKernel(
                    module.function, shape.np // 8, shape.mp // 16, 1,
                    32, 1, 1, 0, None, parameters, None), "cuLaunchKernel(one-warp)")
            values = _driver_event_rounds(driver, launch, warmup, iterations, rounds)
        finally:
            for pointer in reversed(allocations):
                driver._check(driver.lib.cuMemFree_v2(pointer.value), "cuMemFree_v2")
    return summarize(values)


def run_bf16_context(shape_tuple: tuple[int, int, int], warmup: int,
                     iterations: int, rounds: int) -> dict[str, Any]:
    m, n, k = shape_tuple
    torch.manual_seed(20260813 + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(rounds):
        start.record()
        for _ in range(iterations):
            torch.matmul(a, b)
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop) / iterations)
    del a, b
    torch.cuda.empty_cache()
    return summarize(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=BUILD_ROOT / "benchmark")
    parser.add_argument("--report", type=Path, default=BUILD_ROOT / "benchmark.json")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--include-small-shape", action="store_true")
    args = parser.parse_args()
    if args.warmup < 100 or args.iterations < 500 or args.rounds < 5:
        raise ValueError("benchmark protocol requires warmup>=100, iterations>=500, rounds>=5")

    shapes = [*MAIN_SHAPES]
    if args.include_small_shape:
        shapes.append(SENSITIVITY_SHAPE)
    before = telemetry()
    results = []
    for index, shape in enumerate(shapes):
        inputs = make_benchmark_inputs(shape, 20260813 + index * 1009)
        optimized = {
            variant: run_optimized(inputs, variant, args.output_root,
                                   args.warmup, args.iterations, args.rounds)
            for variant in VARIANTS
        }
        e2_median = optimized["00"]["gemm_only"]["median_ms"]
        for variant in VARIANTS:
            optimized[variant]["latency_ratio_vs_e2"] = (
                optimized[variant]["gemm_only"]["median_ms"] / e2_median
            )
        one_warp = run_one_warp_reference(inputs, args.warmup,
                                          args.iterations, args.rounds)
        bf16 = run_bf16_context(shape, args.warmup, args.iterations, args.rounds)
        for variant in VARIANTS:
            optimized[variant]["core_speed_ratio_vs_bf16"] = (
                bf16["median_ms"] / optimized[variant]["gemm_only"]["median_ms"]
            )
        results.append({
            "shape": list(shape),
            "optimized": optimized,
            "one_warp_correctness_vehicle_e2": one_warp,
            "bf16_pytorch_context": bf16,
        })
    after = telemetry()
    report = {
        "scope": "kernel-core feasibility only; not a layer or model speedup",
        "benchmark_process_id": os.getpid(),
        "timing": {
            "method": "CUDA events",
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "independent_rounds": args.rounds,
        },
        "telemetry_before": before,
        "telemetry_after": after,
        "shapes": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.report.resolve()), "shapes": len(results)}, indent=2))


if __name__ == "__main__":
    main()
