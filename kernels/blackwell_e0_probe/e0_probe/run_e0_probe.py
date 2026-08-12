#!/usr/bin/env python3
"""Numerically identify and validate undocumented SM120 E0M3 format bits."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from kernels.blackwell_e0_probe.e0_probe.patch_operand_format import parse_elf
from kernels.blackwell_e0_probe.packing import (
    A_SCALE_SHAPE,
    A_SHAPE,
    B_SCALE_SHAPE,
    B_SHAPE,
    K_BLOCK,
    pack_a,
    pack_a_scales,
    pack_b,
    pack_b_scales,
    unpack_a_scales,
    unpack_b_scales,
)
from kernels.blackwell_e0_probe.reference import (
    decoded_operands,
    scalar_mma,
    vectorized_mma,
)


PROBE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROBE_ROOT / "build" / "e0_probe"
DEFAULT_REPORT = BUILD_ROOT / "e0_results.json"
VARIANT_PATHS = {variant: BUILD_ROOT / f"variant_{variant}.cubin"
                 for variant in ("00", "01", "10", "11")}
PAIRINGS = (("e0m3", "e2m1"), ("e2m1", "e0m3"), ("e0m3", "e0m3"))
ALL_FORMATS = (("e2m1", "e2m1"), *PAIRINGS)


class CudaError(RuntimeError):
    pass


class CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._bind()
        self._check(self.lib.cuInit(0), "cuInit")
        device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        self.context = ctypes.c_void_p()
        self._check(
            self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, device),
            "cuCtxCreate_v2",
        )

    def _bind(self) -> None:
        void_pp = ctypes.POINTER(ctypes.c_void_p)
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuCtxCreate_v2.argtypes = [void_pp, ctypes.c_uint, ctypes.c_int]
        self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cuModuleLoad.argtypes = [void_pp, ctypes.c_char_p]
        self.lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.lib.cuModuleGetFunction.argtypes = [void_pp, ctypes.c_void_p, ctypes.c_char_p]
        self.lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def _check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        name = ctypes.c_char_p()
        description = ctypes.c_char_p()
        self.lib.cuGetErrorName(status, ctypes.byref(name))
        self.lib.cuGetErrorString(status, ctypes.byref(description))
        error_name = name.value.decode() if name.value else "CUDA_ERROR_UNKNOWN"
        error_description = description.value.decode() if description.value else "unknown"
        raise CudaError(f"{operation}: {error_name} ({status}): {error_description}")

    def load(self, cubin: Path) -> "CudaModule":
        return CudaModule(self, cubin)

    def close(self) -> None:
        if self.context:
            self._check(self.lib.cuCtxDestroy_v2(self.context), "cuCtxDestroy_v2")
            self.context = ctypes.c_void_p()

    def __enter__(self) -> "CudaDriver":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class CudaModule:
    SIZES = (16 * 64 // 2, 64 * 8 // 2, 16 * 4, 8 * 4, 16 * 8 * 4)

    def __init__(self, driver: CudaDriver, cubin: Path) -> None:
        self.driver = driver
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.allocations: list[ctypes.c_uint64] = []
        elf = parse_elf(cubin.read_bytes())
        self.kernel_symbol = elf["function"]["name"]
        self.driver._check(
            self.driver.lib.cuModuleLoad(ctypes.byref(self.module), os.fsencode(cubin)),
            "cuModuleLoad",
        )
        try:
            self.driver._check(
                self.driver.lib.cuModuleGetFunction(
                    ctypes.byref(self.function), self.module, self.kernel_symbol.encode()
                ),
                "cuModuleGetFunction",
            )
            for size in self.SIZES:
                pointer = ctypes.c_uint64()
                self.driver._check(
                    self.driver.lib.cuMemAlloc_v2(ctypes.byref(pointer), size),
                    "cuMemAlloc_v2",
                )
                self.allocations.append(pointer)
        except Exception:
            self.close()
            raise

    def run(self, packed_a, packed_b, a_scales, b_scales) -> np.ndarray:
        inputs = [
            np.ascontiguousarray(value.cpu().numpy(), dtype=np.uint8)
            for value in (packed_a, packed_b, a_scales, b_scales)
        ]
        for allocation, array in zip(self.allocations[:4], inputs):
            self.driver._check(
                self.driver.lib.cuMemcpyHtoD_v2(
                    allocation.value, array.ctypes.data_as(ctypes.c_void_p), array.nbytes
                ),
                "cuMemcpyHtoD_v2",
            )
        arguments = [ctypes.c_uint64(pointer.value) for pointer in self.allocations]
        parameters = (ctypes.c_void_p * len(arguments))(*[
            ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
            for argument in arguments
        ])
        self.driver._check(
            self.driver.lib.cuLaunchKernel(
                self.function,
                1, 1, 1,
                32, 1, 1,
                0, None,
                parameters, None,
            ),
            "cuLaunchKernel",
        )
        self.driver._check(self.driver.lib.cuCtxSynchronize(), "cuCtxSynchronize")
        output = np.empty((16, 8), dtype=np.float32)
        self.driver._check(
            self.driver.lib.cuMemcpyDtoH_v2(
                output.ctypes.data_as(ctypes.c_void_p),
                self.allocations[4].value,
                output.nbytes,
            ),
            "cuMemcpyDtoH_v2",
        )
        return output

    def close(self) -> None:
        for pointer in reversed(self.allocations):
            self.driver._check(self.driver.lib.cuMemFree_v2(pointer.value), "cuMemFree_v2")
        self.allocations.clear()
        if self.module:
            self.driver._check(self.driver.lib.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = ctypes.c_void_p()

    def __enter__(self) -> "CudaModule":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _u8(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.uint8)


def _case_tensors(case: dict):
    return (
        _u8(case["packed_a_bytes"]),
        _u8(case["packed_b_bytes"]),
        _u8(case["a_block_scale_bytes"]),
        _u8(case["b_block_scale_bytes"]),
    )


def _make_case(name: str, a_codes: torch.Tensor, b_codes: torch.Tensor) -> dict:
    return {
        "name": name,
        "packed_a_bytes": pack_a(a_codes).tolist(),
        "packed_b_bytes": pack_b(b_codes).tolist(),
        "a_block_scale_bytes": pack_a_scales(torch.ones(A_SCALE_SHAPE)).tolist(),
        "b_block_scale_bytes": pack_b_scales(torch.ones(B_SCALE_SHAPE)).tolist(),
    }


def _single_nonzero_case(a_format: str, b_format: str) -> dict:
    a = torch.zeros(A_SHAPE, dtype=torch.uint8)
    b = torch.zeros(B_SHAPE, dtype=torch.uint8)
    a[3, 17] = 1 if a_format == "e0m3" else 2
    b[17, 5] = 1 if b_format == "e0m3" else 2
    return _make_case("single_nonzero", a, b)


def _diagnostic_case(name: str) -> dict:
    a = torch.zeros(A_SHAPE, dtype=torch.uint8)
    b = torch.zeros(B_SHAPE, dtype=torch.uint8)
    if name == "a_sensitive":
        a[3, 17] = 7
        b[17, 5] = 2
    elif name == "b_sensitive":
        a[3, 17] = 2
        b[17, 5] = 7
    elif name == "cross_check":
        m = torch.arange(16).reshape(16, 1)
        k_a = torch.arange(64).reshape(1, 64)
        k_b = torch.arange(64).reshape(64, 1)
        n = torch.arange(8).reshape(1, 8)
        a = (1 + (m + 3 * k_a) % 7).to(torch.uint8)
        b = (1 + (5 * k_b + n) % 7).to(torch.uint8)
    else:
        raise ValueError(name)
    return _make_case(name, a, b)


def _rounding_bound(pa, pb, sa, sb, a_format: str, b_format: str) -> torch.Tensor:
    a, b = decoded_operands(pa, pb, a_format, b_format)
    a_scale = unpack_a_scales(sa).repeat_interleave(K_BLOCK, dim=1)
    b_scale = unpack_b_scales(sb).repeat_interleave(K_BLOCK, dim=1).T
    sum_absolute_products = torch.matmul((a * a_scale).abs(), (b * b_scale).abs())
    unit_roundoff = 2.0**-24
    gamma_64 = (64 * unit_roundoff) / (1 - 64 * unit_roundoff)
    return 2 * gamma_64 * sum_absolute_products


def _metrics(actual: torch.Tensor, expected: torch.Tensor,
             allowed: torch.Tensor) -> dict:
    error = (actual - expected).abs()
    relative = torch.where(
        expected != 0,
        error / expected.abs(),
        torch.where(error == 0, torch.zeros_like(error),
                    torch.full_like(error, float("inf"))),
    )
    mismatches = error > allowed
    coordinates = torch.nonzero(mismatches, as_tuple=False)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float(relative.max()),
        "tolerance_mismatch_count": int(mismatches.sum()),
        "bitwise_mismatch_count": int((actual != expected).sum()),
        "first_mismatch_coordinates": coordinates[:8].tolist(),
        "max_allowed_rounding_error": float(allowed.max()),
    }


def _reference(case: dict, a_format: str, b_format: str):
    pa, pb, sa, sb = _case_tensors(case)
    scalar, _ = scalar_mma(pa, pb, sa, sb, a_format, b_format, 1.0, 1.0)
    vector, _ = vectorized_mma(pa, pb, sa, sb, a_format, b_format, 1.0, 1.0)
    allowed = _rounding_bound(pa, pb, sa, sb, a_format, b_format)
    return (pa, pb, sa, sb), scalar, vector, allowed


def _compare(module: CudaModule, case: dict, a_format: str, b_format: str) -> dict:
    inputs, scalar, vector, allowed = _reference(case, a_format, b_format)
    hardware = torch.from_numpy(module.run(*inputs).copy())
    scalar_vector = _metrics(vector, scalar, allowed)
    scalar_hardware = _metrics(hardware, scalar, allowed)
    vector_hardware = _metrics(hardware, vector, allowed)
    return {
        "name": case["name"],
        "global_fp32_scales": [1.0, 1.0],
        "scalar_vs_vectorized": scalar_vector,
        "scalar_vs_hardware": scalar_hardware,
        "vectorized_vs_hardware": vector_hardware,
        "passed": all(
            metrics["tolerance_mismatch_count"] == 0
            for metrics in (scalar_vector, scalar_hardware, vector_hardware)
        ),
    }


def identify_bit_mapping(driver: CudaDriver) -> dict:
    cases = [_diagnostic_case(name) for name in
             ("a_sensitive", "b_sensitive", "cross_check")]
    variants = {}
    variant_pairings = {}
    for variant, path in VARIANT_PATHS.items():
        with driver.load(path) as module:
            case_reports = []
            for case in cases:
                inputs = _case_tensors(case)
                hardware = torch.from_numpy(module.run(*inputs).copy())
                candidates = {}
                for a_format, b_format in ALL_FORMATS:
                    _, scalar, _, allowed = _reference(case, a_format, b_format)
                    metrics = _metrics(hardware, scalar, allowed)
                    candidates[f"{a_format}_x_{b_format}"] = metrics
                exact = [name for name, metrics in candidates.items()
                         if metrics["tolerance_mismatch_count"] == 0]
                case_reports.append({
                    "name": case["name"],
                    "candidate_references": candidates,
                    "matching_pairings": exact,
                })
            common = set(case_reports[0]["matching_pairings"])
            for report in case_reports[1:]:
                common &= set(report["matching_pairings"])
            variants[variant] = {
                "driver_load": "passed",
                "cases": case_reports,
                "unique_pairing": sorted(common),
            }
            if len(common) != 1:
                raise RuntimeError(
                    f"SASS BIT MAPPING INCONCLUSIVE: variant {variant} matches {sorted(common)}"
                )
            variant_pairings[variant] = next(iter(common))

    expected_set = {
        "e2m1_x_e2m1", "e0m3_x_e2m1", "e2m1_x_e0m3", "e0m3_x_e0m3"
    }
    if set(variant_pairings.values()) != expected_set:
        raise RuntimeError(f"SASS BIT MAPPING INCONCLUSIVE: {variant_pairings}")
    if variant_pairings["00"] != "e2m1_x_e2m1" or variant_pairings["11"] != "e0m3_x_e0m3":
        raise RuntimeError(f"SASS BIT MAPPING INCONCLUSIVE: {variant_pairings}")

    bit78_pairing = variant_pairings["01"]
    bit79_pairing = variant_pairings["10"]
    bit78_operand = "A" if bit78_pairing == "e0m3_x_e2m1" else "B"
    bit79_operand = "A" if bit79_pairing == "e0m3_x_e2m1" else "B"
    if {bit78_operand, bit79_operand} != {"A", "B"}:
        raise RuntimeError(f"SASS BIT MAPPING INCONCLUSIVE: {variant_pairings}")
    return {
        "variant_spelling": "bit79_bit78",
        "variant_pairings": variant_pairings,
        "bit78_operand": bit78_operand,
        "bit79_operand": bit79_operand,
        "diagnostics": variants,
        "passed": True,
    }


def _load_golden_cases(a_format: str, b_format: str) -> list[dict]:
    path = PROBE_ROOT / "golden" / f"{a_format}_x_{b_format}.json"
    document = json.loads(path.read_text())
    return [_single_nonzero_case(a_format, b_format), *document["cases"]]


def run_pairing_suites(driver: CudaDriver, mapping: dict) -> dict:
    inverse = {pairing: variant for variant, pairing in mapping["variant_pairings"].items()}
    reports = {}
    for a_format, b_format in PAIRINGS:
        pairing = f"{a_format}_x_{b_format}"
        variant = inverse[pairing]
        with driver.load(VARIANT_PATHS[variant]) as module:
            cases = [_compare(module, case, a_format, b_format)
                     for case in _load_golden_cases(a_format, b_format)]
        reports[pairing] = {
            "variant": variant,
            "cases": cases,
            "passed": all(case["passed"] for case in cases),
        }
    return reports


def _negative_metrics(module: CudaModule, case: dict, a_format: str,
                      b_format: str, packed_b_override=None) -> dict:
    inputs, scalar, _, allowed = _reference(case, a_format, b_format)
    if packed_b_override is not None:
        inputs = (inputs[0], packed_b_override, inputs[2], inputs[3])
    hardware = torch.from_numpy(module.run(*inputs).copy())
    metrics = _metrics(hardware, scalar, allowed)
    return {**metrics, "caught": metrics["tolerance_mismatch_count"] > 0}


def run_negative_controls(driver: CudaDriver, mapping: dict) -> dict:
    inverse = {pairing: variant for variant, pairing in mapping["variant_pairings"].items()}
    controls = {}

    cross = _diagnostic_case("cross_check")
    with driver.load(VARIANT_PATHS["00"]) as module:
        controls["e0_payload_on_unpatched_e2"] = _negative_metrics(
            module, cross, "e0m3", "e0m3"
        )

    a_sensitive = _diagnostic_case("a_sensitive")
    b_sensitive = _diagnostic_case("b_sensitive")
    a_correct = inverse["e0m3_x_e2m1"]
    b_correct = inverse["e2m1_x_e0m3"]
    with driver.load(VARIANT_PATHS[b_correct]) as wrong_a_module:
        controls["a_sensitive_wrong_operand_bit"] = _negative_metrics(
            wrong_a_module, a_sensitive, "e0m3", "e2m1"
        )
    with driver.load(VARIANT_PATHS[a_correct]) as wrong_b_module:
        controls["b_sensitive_wrong_operand_bit"] = _negative_metrics(
            wrong_b_module, b_sensitive, "e2m1", "e0m3"
        )

    layout_case = next(
        case for case in _load_golden_cases("e0m3", "e0m3")
        if case["name"] == "layout_sensitive"
    )
    pa, pb, sa, sb = _case_tensors(layout_case)
    wrong_b = ((pb & 0xF) << 4) | (pb >> 4)
    with driver.load(VARIANT_PATHS[inverse["e0m3_x_e0m3"]]) as module:
        controls["b_high_low_nibbles_swapped"] = _negative_metrics(
            module, layout_case, "e0m3", "e0m3", wrong_b
        )

    with driver.load(VARIANT_PATHS[a_correct]) as module:
        controls["confirmed_e0_a_interpreted_as_e2"] = _negative_metrics(
            module, a_sensitive, "e2m1", "e2m1"
        )
    with driver.load(VARIANT_PATHS[b_correct]) as module:
        controls["confirmed_e0_b_interpreted_as_e2"] = _negative_metrics(
            module, b_sensitive, "e2m1", "e2m1"
        )
    controls["passed"] = all(value["caught"] for key, value in controls.items()
                             if key != "passed")
    return controls


def _code_case(operand: str, code: int) -> dict:
    a = torch.zeros(A_SHAPE, dtype=torch.uint8)
    b = torch.zeros(B_SHAPE, dtype=torch.uint8)
    if operand == "A":
        a[0, 0] = code
        b[0, 0] = 2
    else:
        a[0, 0] = 2
        b[0, 0] = code
    return _make_case(f"{operand.lower()}_code_{code}", a, b)


def run_encoding_audit(driver: CudaDriver, mapping: dict) -> dict:
    inverse = {pairing: variant for variant, pairing in mapping["variant_pairings"].items()}
    results = {}
    for operand, pairing in (("A", "e0m3_x_e2m1"), ("B", "e2m1_x_e0m3")):
        a_format, b_format = pairing.split("_x_")
        observed = []
        output_bits = []
        passed = True
        with driver.load(VARIANT_PATHS[inverse[pairing]]) as module:
            for code in range(16):
                case = _code_case(operand, code)
                inputs, scalar, _, allowed = _reference(case, a_format, b_format)
                hardware = torch.from_numpy(module.run(*inputs).copy())
                metrics = _metrics(hardware, scalar, allowed)
                passed &= metrics["tolerance_mismatch_count"] == 0
                value = float(hardware[0, 0])
                observed.append(value)
                output_bits.append(int(hardware[0, 0].numpy().view(np.uint32)))
        results[operand] = {
            "observed_code_values": observed,
            "output_fp32_bits": output_bits,
            "positive_zero": observed[0],
            "negative_zero": observed[8],
            "positive_five": observed[5],
            "negative_five": observed[13],
            "positive_seven": observed[7],
            "negative_seven": observed[15],
            "e2_positive_half_code_in_e0_mode": observed[1],
            "e2_negative_half_code_in_e0_mode": observed[9],
            "all_finite": bool(np.isfinite(np.asarray(observed)).all()),
            "sign_magnitude_linear": observed[:8] == [float(i) for i in range(8)]
                                     and observed[9:] == [-float(i) for i in range(1, 8)],
            "signed_zero_changes_accumulated_output_bits": output_bits[0] != output_bits[8],
            "passed": passed,
        }
    results["passed"] = (
        all(results[operand]["passed"] for operand in ("A", "B"))
        and all(results[operand]["all_finite"] for operand in ("A", "B"))
        and all(results[operand]["sign_magnitude_linear"] for operand in ("A", "B"))
    )
    return results


def _output_digest(module: CudaModule, cases: list[dict]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(module.run(*_case_tensors(case)).tobytes())
    return digest.hexdigest()


def _replay_child(variant: str, a_format: str, b_format: str) -> str:
    with CudaDriver() as driver:
        with driver.load(VARIANT_PATHS[variant]) as module:
            return _output_digest(module, _load_golden_cases(a_format, b_format))


def run_stability(driver: CudaDriver, mapping: dict) -> dict:
    inverse = {pairing: variant for variant, pairing in mapping["variant_pairings"].items()}
    pairings = {}
    for a_format, b_format in PAIRINGS:
        pairing = f"{a_format}_x_{b_format}"
        variant = inverse[pairing]
        cases = _load_golden_cases(a_format, b_format)
        with driver.load(VARIANT_PATHS[variant]) as module:
            reference_outputs = [module.run(*_case_tensors(case)).copy() for case in cases]
            deterministic = True
            for _ in range(99):
                for case, expected in zip(cases, reference_outputs):
                    deterministic &= np.array_equal(
                        module.run(*_case_tensors(case)), expected
                    )
            expected_digest = _output_digest(module, cases)

        reload_hashes = []
        for _ in range(10):
            with driver.load(VARIANT_PATHS[variant]) as module:
                reload_hashes.append(_output_digest(module, cases))

        process_hashes = []
        for _ in range(10):
            completed = subprocess.run(
                [
                    sys.executable, "-m",
                    "kernels.blackwell_e0_probe.e0_probe.run_e0_probe",
                    "--single-replay", variant, a_format, b_format,
                ],
                check=True, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[3],
            )
            process_hashes.append(completed.stdout.strip())
        pairings[pairing] = {
            "variant": variant,
            "same_loaded_module_repetitions_per_case": 100,
            "same_loaded_module_deterministic": deterministic,
            "module_reload_count": len(reload_hashes),
            "module_reload_hashes": reload_hashes,
            "new_process_count": len(process_hashes),
            "new_process_hashes": process_hashes,
            "expected_digest": expected_digest,
            "passed": deterministic
                      and all(value == expected_digest for value in reload_hashes)
                      and all(value == expected_digest for value in process_hashes),
        }
    pairings["passed"] = all(value["passed"] for key, value in pairings.items()
                             if key != "passed")
    return pairings


def run_experiment(include_stability: bool = True) -> dict:
    with CudaDriver() as driver:
        mapping = identify_bit_mapping(driver)
        suites = run_pairing_suites(driver, mapping)
        negatives = run_negative_controls(driver, mapping)
        encoding = run_encoding_audit(driver, mapping)
        stability = run_stability(driver, mapping) if include_stability else {"skipped": True}
    passed = (
        mapping["passed"]
        and all(report["passed"] for report in suites.values())
        and negatives["passed"]
        and encoding["passed"]
        and (not include_stability or stability["passed"])
    )
    return {
        "unsupported_warning": (
            "Undocumented SASS operand-format patch; unsupported by NVIDIA and "
            "not a portable or production API."
        ),
        "rounding_rule": (
            "elementwise abs(error) <= 2*gamma_64*sum(abs(products)), "
            "gamma_64=(64*2^-24)/(1-64*2^-24)"
        ),
        "bit_mapping": mapping,
        "pairing_suites": suites,
        "negative_controls": negatives,
        "encoding_audit": encoding,
        "stability": stability,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--single-replay", nargs=3,
                        metavar=("VARIANT", "A_FORMAT", "B_FORMAT"))
    args = parser.parse_args()
    if args.single_replay:
        print(_replay_child(*args.single_replay))
        return
    report = run_experiment(include_stability=not args.skip_stability)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
