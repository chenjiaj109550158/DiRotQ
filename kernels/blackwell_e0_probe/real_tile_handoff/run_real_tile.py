#!/usr/bin/env python3
"""Verify a v1 package, then run its E0xE2/E0xE0 cases on RTX 5090."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kernels.blackwell_e0_probe.e0_probe.run_e0_probe import CudaDriver
from kernels.blackwell_e0_probe.gemm_probe.patch_static_gemm import (
    analyze_baseline,
    validate_variant,
)
from kernels.blackwell_e0_probe.gemm_probe.run_gemm_probe import (
    BUILD_ROOT as GEMM_BUILD_ROOT,
    GemmModule,
)
from kernels.blackwell_e0_probe.real_tile_handoff.schema import DEFAULT_MAX_PACKAGE_BYTES
from kernels.blackwell_e0_probe.real_tile_handoff.verify_package import (
    VerifiedPackage,
    verify_package,
)


DEFAULT_CUBINS = {
    "01": GEMM_BUILD_ROOT / "variant_01.cubin",
    "11": GEMM_BUILD_ROOT / "variant_11.cubin",
}
ALLOWLISTED_VARIANT_SHA256 = {
    "01": "ec4cc412692855b181cf96e39415ff9f6c089c19951e3acc212eb7439d86f0ae",
    "11": "144c0aa4e7722c417ab624313ccce7066c17cc54a019dca0e48bcbff5d1d0d7a",
}
BASELINE_CUBIN = GEMM_BUILD_ROOT / "baseline_00.cubin"
UNIT_ROUNDOFF_FP32 = 2.0 ** -24


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_allowlisted_cubin(cubin: Path, variant: str,
                               baseline: Path = BASELINE_CUBIN) -> dict[str, Any]:
    if variant not in DEFAULT_CUBINS:
        raise ValueError("real tile receiver permits only variants 01 and 11")
    original = baseline.read_bytes()
    analysis = analyze_baseline(original)
    candidate = cubin.read_bytes()
    digest = _sha256(candidate)
    if digest != ALLOWLISTED_VARIANT_SHA256[variant]:
        raise ValueError(f"non-allowlisted variant {variant} CUBIN SHA-256: {digest}")
    diff = validate_variant(
        original,
        candidate,
        variant,
        analysis["instruction_file_offsets"],
    )
    return {
        "variant": variant,
        "path": str(cubin.resolve()),
        "sha256": digest,
        "baseline_sha256": _sha256(original),
        "omma_count": len(analysis["instruction_file_offsets"]),
        "whole_cubin_byte_diff": diff["byte_offsets"],
        "whole_cubin_file_bit_diff": diff["file_bit_offsets"],
    }


def _relative_error(error: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.where(
        expected != 0,
        error / expected.abs(),
        torch.where(error == 0, torch.zeros_like(error), torch.full_like(error, float("inf"))),
    )


def _packed_metrics(actual: torch.Tensor, expected: torch.Tensor,
                    allowed: torch.Tensor) -> dict[str, Any]:
    error = (actual - expected).abs()
    mismatch = error > allowed
    coordinates = torch.nonzero(mismatch, as_tuple=False)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float(_relative_error(error, expected).max()),
        "mismatch_count": int(mismatch.sum()),
        "bitwise_mismatch_count": int((actual != expected).sum()),
        "first_mismatch_coordinates": coordinates[:8].tolist(),
        "max_gamma_k_tolerance": float(allowed.max()),
    }


def _runtime_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    error = (actual - expected).abs()
    mismatch = actual != expected
    coordinates = torch.nonzero(mismatch, as_tuple=False)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float(_relative_error(error, expected).max()),
        "mismatch_count": int(mismatch.sum()),
        "first_mismatch_coordinates": coordinates[:8].tolist(),
        "comparison_rule": "exact comparison after hardware FP32 is cast to runtime_dtype",
        "informational_only": True,
    }


def _runtime_cast(values: torch.Tensor, runtime_dtype: str) -> torch.Tensor:
    dtype = torch.bfloat16 if runtime_dtype == "bfloat16" else torch.float16
    return values.to(dtype).float()


def run_verified_package(package: VerifiedPackage,
                         cubin_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    selected = dict(DEFAULT_CUBINS if cubin_paths is None else cubin_paths)
    needed_variants = sorted({case.contract.variant for case in package.cases})
    allowlist = {}
    for variant in needed_variants:
        if variant not in selected:
            raise ValueError(f"missing CUBIN path for required variant {variant}")
        allowlist[variant] = validate_allowlisted_cubin(selected[variant], variant)

    results = []
    with CudaDriver() as driver:
        modules = {
            variant: GemmModule(driver, selected[variant]) for variant in needed_variants
        }
        try:
            for case in package.cases:
                output, guard = modules[case.contract.variant].run(case.inputs)
                hardware = torch.from_numpy(output.copy())
                packed_expected = torch.from_numpy(case.expected_packed_fp32.copy())
                allowed = torch.from_numpy(case.packed_allowed_error.copy())
                packed = _packed_metrics(hardware, packed_expected, allowed)

                runtime_hardware = _runtime_cast(hardware, case.contract.runtime_dtype)
                runtime_expected = torch.from_numpy(
                    case.expected_fakequant_runtime.copy()
                )
                runtime = _runtime_metrics(runtime_hardware, runtime_expected)
                gamma_k = (
                    case.contract.shape.k * UNIT_ROUNDOFF_FP32
                    / (1 - case.contract.shape.k * UNIT_ROUNDOFF_FP32)
                )
                passed = (
                    packed["mismatch_count"] == 0
                    and guard["prefix_canary_ok"]
                    and guard["suffix_canary_ok"]
                )
                results.append({
                    "case_id": case.contract.case_id,
                    "pairing": case.contract.pairing,
                    "variant": case.contract.variant,
                    "shape": [
                        case.contract.shape.m,
                        case.contract.shape.n,
                        case.contract.shape.k,
                    ],
                    "padded_shape": [
                        case.contract.shape.mp,
                        case.contract.shape.np,
                        case.contract.shape.kp,
                    ],
                    "alpha_A": case.contract.alpha_a,
                    "alpha_B": case.contract.alpha_b,
                    "gamma_K": gamma_k,
                    "hardware_fp32_vs_expected_packed_fp32": packed,
                    "hardware_runtime_cast_vs_expected_fakequant_runtime": {
                        "runtime_dtype": case.contract.runtime_dtype,
                        "stored_expected_dtype": "float32 materialization",
                        **runtime,
                    },
                    "output_guard": guard,
                    "passed": passed,
                })
        finally:
            for module in modules.values():
                module.close()
    return {
        "package_root": str(package.root),
        "schema": package.manifest["schema"],
        "strict_verifier_passed_before_cuda": True,
        "allowlisted_cubins": allowlist,
        "cases": results,
        "passed": all(case["passed"] for case in results),
        "runtime_comparison_is_acceptance_gate": False,
    }


def run_package(package_root: Path | str, *,
                max_package_bytes: int = DEFAULT_MAX_PACKAGE_BYTES,
                cubin_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    # This call performs every untrusted package check before CUBIN inspection,
    # CUDA initialization, module loading, allocation, or launch.
    package = verify_package(package_root, max_package_bytes=max_package_bytes)
    return run_verified_package(package, cubin_paths=cubin_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--max-package-bytes", type=int, default=DEFAULT_MAX_PACKAGE_BYTES)
    parser.add_argument("--cubin-01", type=Path, default=DEFAULT_CUBINS["01"])
    parser.add_argument("--cubin-11", type=Path, default=DEFAULT_CUBINS["11"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_package(
        args.package,
        max_package_bytes=args.max_package_bytes,
        cubin_paths={"01": args.cubin_01, "11": args.cubin_11},
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "ALLOWLISTED_VARIANT_SHA256", "DEFAULT_CUBINS", "run_package",
    "run_verified_package", "validate_allowlisted_cubin",
]
