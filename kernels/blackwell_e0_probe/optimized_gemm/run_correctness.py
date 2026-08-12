#!/usr/bin/env python3
"""Correctness orchestration for the external-CUBIN optimized FP4 prototype."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any

import numpy as np
import torch

from kernels.blackwell_e0_probe.gemm_probe.packing import (
    CanonicalInputs,
    validate_canonical,
)
from kernels.blackwell_e0_probe.gemm_probe.reference import (
    fp32_comparison_bound,
    vectorized_packed_gemm,
)
from kernels.blackwell_e0_probe.gemm_probe.run_gemm_probe import make_inputs
from kernels.blackwell_e0_probe.optimized_gemm.patch_optimized_gemm import (
    BASELINE_SHA256,
    analyze_baseline,
    validate_variant,
)
from kernels.blackwell_e0_probe.real_tile_handoff.verify_package import (
    VerifiedCase,
    verify_package,
)


PROBE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROBE_ROOT / "build" / "optimized_gemm"
RUNNER = BUILD_ROOT / "optimized_fp4_runner"
CUBINS = {
    "00": BUILD_ROOT / "baseline_00.cubin",
    "01": BUILD_ROOT / "variant_01.cubin",
    "11": BUILD_ROOT / "variant_11.cubin",
}
ALLOWLISTED_SHA256 = {
    "00": BASELINE_SHA256,
    "01": "1206da275bc1ce070bda02d5451a229a643d3ca965b361eeb981ab6a77fac33a",
    "11": "80a4c5154f392954b8515531974e74ceab16b32989eb4240c650673107c7ada9",
}
PAIRING_VARIANTS = {
    ("e2m1", "e2m1"): "00",
    ("e0m3", "e2m1"): "01",
    ("e0m3", "e0m3"): "11",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_allowlisted_cubin(path: Path, variant: str) -> dict[str, Any]:
    if variant not in ALLOWLISTED_SHA256:
        raise ValueError(f"unsupported optimized variant: {variant}")
    digest = _sha256(path)
    if digest != ALLOWLISTED_SHA256[variant]:
        raise ValueError(f"non-allowlisted optimized variant {variant}: {digest}")
    baseline = CUBINS["00"].read_bytes()
    analysis = analyze_baseline(baseline)
    candidate = path.read_bytes()
    diff = validate_variant(
        baseline, candidate, variant, analysis["instruction_file_offsets"]
    )
    return {
        "variant": variant,
        "path": str(path.resolve()),
        "sha256": digest,
        "baseline_sha256": BASELINE_SHA256,
        "kernel_symbol": analysis["elf"]["function"]["name"],
        "omma_count": len(analysis["instruction_file_offsets"]),
        "whole_cubin_byte_diff": diff["byte_offsets"],
        "whole_cubin_file_bit_diff": diff["file_bit_offsets"],
    }


def write_runner_input(path: Path, inputs: CanonicalInputs) -> None:
    validate_canonical(inputs)
    shape = inputs.shape
    header = struct.pack(
        "<8sII6i2f16s",
        b"OPTFP4V1", 1, 64,
        shape.m, shape.n, shape.k, shape.kp, shape.mp, shape.np,
        inputs.alpha_a, inputs.alpha_b, b"\0" * 16,
    )
    parts = [
        np.ascontiguousarray(value.numpy(), dtype=np.uint8).tobytes()
        for value in (inputs.packed_a, inputs.packed_b,
                      inputs.a_scales, inputs.b_scales)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(parts))


def _relative(error: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.where(
        expected != 0,
        error / expected.abs(),
        torch.where(error == 0, torch.zeros_like(error),
                    torch.full_like(error, float("inf"))),
    )


def comparison_metrics(actual: torch.Tensor, expected: torch.Tensor,
                       allowed: torch.Tensor) -> dict[str, Any]:
    error = (actual - expected).abs()
    mismatch = error > allowed
    coordinates = torch.nonzero(mismatch, as_tuple=False)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float(_relative(error, expected).max()),
        "tolerance_mismatch_count": int(mismatch.sum()),
        "bitwise_mismatch_count": int((actual != expected).sum()),
        "first_mismatch_coordinates": coordinates[:8].tolist(),
        "max_gamma_k_tolerance": float(allowed.max()),
    }


def runtime_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    error = (actual - expected).abs()
    mismatch = actual != expected
    coordinates = torch.nonzero(mismatch, as_tuple=False)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float(_relative(error, expected).max()),
        "mismatch_count": int(mismatch.sum()),
        "first_mismatch_coordinates": coordinates[:8].tolist(),
        "informational_only": True,
    }


def _read_bf16(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16).copy().reshape(shape)
    return torch.from_numpy(raw).view(torch.bfloat16).float()


def run_case(case_id: str, inputs: CanonicalInputs, a_format: str, b_format: str,
             expected: torch.Tensor, allowed: torch.Tensor,
             output_root: Path, *, runtime_expected: torch.Tensor | None = None,
             runtime_dtype: str = "bfloat16") -> dict[str, Any]:
    variant = PAIRING_VARIANTS[(a_format, b_format)]
    cubin_audit = validate_allowlisted_cubin(CUBINS[variant], variant)
    case_root = output_root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    input_path = case_root / "input.bin"
    output_path = case_root / "output_fp32.bin"
    bf16_path = case_root / "output_bf16.bin"
    runner_report_path = case_root / "runner.json"
    write_runner_input(input_path, inputs)
    command = [
        str(RUNNER), "--cubin", str(CUBINS[variant]),
        "--input", str(input_path), "--output", str(output_path),
        "--bf16-output", str(bf16_path), "--report", str(runner_report_path),
        "--warmup", "1", "--iterations", "1", "--rounds", "1",
    ]
    subprocess.run(command, check=True)
    runner_report = json.loads(runner_report_path.read_text())
    shape = inputs.shape
    padded = torch.from_numpy(
        np.fromfile(output_path, dtype=np.float32).copy().reshape(shape.mp, shape.np)
    )
    hardware = padded[:shape.m, :shape.n].clone()
    packed = comparison_metrics(hardware, expected, allowed)
    bf16_padded = _read_bf16(bf16_path, (shape.mp, shape.np))
    bf16_hardware = bf16_padded[:shape.m, :shape.n].clone()
    runtime = None
    if runtime_expected is not None:
        if runtime_dtype != "bfloat16":
            raise ValueError("optimized v1 runner currently materializes BF16 runtime output")
        runtime = runtime_metrics(bf16_hardware, runtime_expected)
    passed = (
        packed["tolerance_mismatch_count"] == 0
        and runner_report["passed"]
        and runner_report["canary_prefix_ok"]
        and runner_report["canary_suffix_ok"]
    )
    return {
        "case_id": case_id,
        "variant": variant,
        "a_format": a_format,
        "b_format": b_format,
        "shape": [shape.m, shape.n, shape.k],
        "padded_shape": [shape.mp, shape.np, shape.kp],
        "alpha_a": inputs.alpha_a,
        "alpha_b": inputs.alpha_b,
        "cubin": cubin_audit,
        "hardware_fp32_vs_packed_fp32": packed,
        "hardware_bf16_vs_runtime_reference": runtime,
        "runner": runner_report,
        "passed": passed,
    }


def _zero_one_block(inputs: CanonicalInputs) -> CanonicalInputs:
    packed_a = inputs.packed_a.clone()
    packed_b = inputs.packed_b.clone()
    packed_a[:, :8] = 0
    packed_b[:, :8] = 0
    return replace(inputs, packed_a=packed_a, packed_b=packed_b)


SYNTHETIC_CASES = (
    ("e2_aligned_k64", (128, 128, 64), "layout_sensitive", "e2m1", "e2m1"),
    ("e2_tail_global", (17, 9, 65), "global_scales", "e2m1", "e2m1"),
    ("e0xe2_multik_scales", (128, 128, 128), "row_column_scales", "e0m3", "e2m1"),
    ("e0xe2_k_gt_1024", (128, 128, 1088), "deterministic_random", "e0m3", "e2m1"),
    ("e0xe0_scale_extremes", (128, 128, 128), "scale_extremes", "e0m3", "e0m3"),
    ("e0xe0_tail_zero_block", (17, 9, 65), "row_column_scales", "e0m3", "e0m3"),
)


def run_synthetic(output_root: Path) -> dict[str, Any]:
    results = []
    for index, (case_id, shape, pattern, a_format, b_format) in enumerate(SYNTHETIC_CASES):
        inputs = make_inputs(shape, pattern, seed=20260813 + index * 97)
        if case_id.endswith("zero_block"):
            inputs = _zero_one_block(inputs)
        _, expected = vectorized_packed_gemm(inputs, a_format, b_format)
        allowed = fp32_comparison_bound(inputs, a_format, b_format, expected)
        results.append(run_case(
            case_id, inputs, a_format, b_format, expected, allowed,
            output_root / "synthetic",
        ))
    return {"cases": results, "passed": all(case["passed"] for case in results)}


def run_real_package(package_root: Path, output_root: Path) -> dict[str, Any]:
    # The strict, CUDA-free verifier is deliberately the first package action.
    package = verify_package(package_root)
    results = []
    for case in package.cases:
        contract = case.contract
        results.append(run_case(
            contract.case_id,
            case.inputs,
            contract.a_format,
            contract.b_format,
            torch.from_numpy(case.expected_packed_fp32.copy()),
            torch.from_numpy(case.packed_allowed_error.copy()),
            output_root / package.root.name,
            runtime_expected=torch.from_numpy(case.expected_fakequant_runtime.copy()),
            runtime_dtype=contract.runtime_dtype,
        ) | {
            "layer_name": case.metadata["layer_name"],
            "scheduler_timestep": case.metadata["scheduler_timestep"],
        })
    return {
        "package_root": str(package.root),
        "schema": package.manifest["schema"],
        "strict_verifier_passed_before_cuda": True,
        "cases": results,
        "passed": all(case["passed"] for case in results),
    }


def wrong_variant_guard() -> dict[str, Any]:
    rejected = False
    error = ""
    try:
        requested_pairing = ("e0m3", "e0m3")
        selected_variant = "01"
        required_variant = PAIRING_VARIANTS[requested_pairing]
        if selected_variant != required_variant:
            raise ValueError(
                f"pairing guard: {requested_pairing} requires {required_variant}, got {selected_variant}"
            )
    except ValueError as exception:
        rejected = True
        error = str(exception)
    return {"rejected_before_launch": rejected, "error": error, "passed": rejected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-e0xe2", type=Path)
    parser.add_argument("--real-e0xe0", type=Path)
    parser.add_argument("--output-root", type=Path, default=BUILD_ROOT / "correctness")
    parser.add_argument("--report", type=Path, default=BUILD_ROOT / "correctness.json")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "allowlisted_cubins": {
            variant: validate_allowlisted_cubin(path, variant)
            for variant, path in CUBINS.items()
        },
        "synthetic": run_synthetic(args.output_root),
        "wrong_variant_negative_control": wrong_variant_guard(),
    }
    if bool(args.real_e0xe2) != bool(args.real_e0xe0):
        raise ValueError("both real package roots must be supplied together")
    if args.real_e0xe2:
        report["real_e0xe2"] = run_real_package(args.real_e0xe2, args.output_root)
        report["real_e0xe0"] = run_real_package(args.real_e0xe0, args.output_root)
    report["passed"] = all(
        section["passed"] for key, section in report.items()
        if key in {"synthetic", "wrong_variant_negative_control", "real_e0xe2", "real_e0xe0"}
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "synthetic_cases": len(report["synthetic"]["cases"]),
        "real_cases": sum(len(report[key]["cases"]) for key in ("real_e0xe2", "real_e0xe0") if key in report),
        "report": str(args.report.resolve()),
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
