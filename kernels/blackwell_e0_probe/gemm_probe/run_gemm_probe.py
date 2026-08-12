#!/usr/bin/env python3
"""Run canonical static-format GEMM inputs through patched SM120 CUBINs."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from kernels.blackwell_e0_probe.e0_probe.patch_operand_format import parse_elf
from kernels.blackwell_e0_probe.e0_probe.run_e0_probe import CudaDriver
from kernels.blackwell_e0_probe.gemm_probe.packing import (
    CanonicalInputs,
    GemmShape,
    pack_canonical_a,
    pack_canonical_a_scales,
    pack_canonical_b,
    pack_canonical_b_scales,
    pack_nibbles,
    unpack_canonical_a,
    unpack_canonical_b,
    validate_canonical,
)
from kernels.blackwell_e0_probe.gemm_probe.reference import (
    decoded_fp64_gemm,
    fp32_comparison_bound,
    sequential_fp32_gemm,
    vectorized_packed_gemm,
)


PROBE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROBE_ROOT / "build" / "gemm_probe"
VARIANT_PATHS = {variant: BUILD_ROOT / f"variant_{variant}.cubin"
                 for variant in ("00", "01", "10", "11")}
PAIRINGS = {
    "00": ("e2m1", "e2m1"),
    "01": ("e0m3", "e2m1"),
    "10": ("e2m1", "e0m3"),
    "11": ("e0m3", "e0m3"),
}
CANARY_BYTES = 64
CANARY_VALUE = 0xA5


class GemmModule:
    def __init__(self, driver: CudaDriver, cubin: Path) -> None:
        self.driver = driver
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        elf = parse_elf(cubin.read_bytes())
        symbol = elf["function"]["name"]
        if symbol != "static_fp4_gemm":
            raise RuntimeError(f"unexpected kernel symbol {symbol}")
        self.driver._check(
            self.driver.lib.cuModuleLoad(ctypes.byref(self.module), os.fsencode(cubin)),
            "cuModuleLoad",
        )
        try:
            self.driver._check(
                self.driver.lib.cuModuleGetFunction(
                    ctypes.byref(self.function), self.module, symbol.encode()
                ),
                "cuModuleGetFunction",
            )
        except Exception:
            self.close()
            raise

    def _allocate(self, size: int) -> ctypes.c_uint64:
        pointer = ctypes.c_uint64()
        self.driver._check(
            self.driver.lib.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc_v2"
        )
        return pointer

    def run(self, inputs: CanonicalInputs, *, validate: bool = True):
        if validate:
            validate_canonical(inputs)
        arrays = [
            np.ascontiguousarray(value.cpu().numpy(), dtype=np.uint8)
            for value in (inputs.packed_a, inputs.packed_b,
                          inputs.a_scales, inputs.b_scales)
        ]
        output_bytes = inputs.shape.m * inputs.shape.n * 4
        allocations: list[ctypes.c_uint64] = []
        try:
            for array in arrays:
                pointer = self._allocate(array.nbytes)
                allocations.append(pointer)
                self.driver._check(
                    self.driver.lib.cuMemcpyHtoD_v2(
                        pointer.value, array.ctypes.data_as(ctypes.c_void_p), array.nbytes
                    ),
                    "cuMemcpyHtoD_v2",
                )
            guarded = np.full(output_bytes + 2 * CANARY_BYTES,
                              CANARY_VALUE, dtype=np.uint8)
            output_base = self._allocate(guarded.nbytes)
            allocations.append(output_base)
            self.driver._check(
                self.driver.lib.cuMemcpyHtoD_v2(
                    output_base.value, guarded.ctypes.data_as(ctypes.c_void_p), guarded.nbytes
                ),
                "cuMemcpyHtoD_v2(output canary)",
            )
            output_pointer = output_base.value + CANARY_BYTES
            arguments = [
                ctypes.c_uint64(allocations[0].value),
                ctypes.c_uint64(allocations[1].value),
                ctypes.c_uint64(allocations[2].value),
                ctypes.c_uint64(allocations[3].value),
                ctypes.c_int(inputs.shape.m),
                ctypes.c_int(inputs.shape.n),
                ctypes.c_int(inputs.shape.kp),
                ctypes.c_float(inputs.alpha_a),
                ctypes.c_float(inputs.alpha_b),
                ctypes.c_uint64(output_pointer),
            ]
            parameters = (ctypes.c_void_p * len(arguments))(*[
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            ])
            self.driver._check(
                self.driver.lib.cuLaunchKernel(
                    self.function,
                    inputs.shape.np // 8, inputs.shape.mp // 16, 1,
                    32, 1, 1,
                    0, None,
                    parameters, None,
                ),
                "cuLaunchKernel(static_fp4_gemm)",
            )
            self.driver._check(self.driver.lib.cuCtxSynchronize(), "cuCtxSynchronize")
            self.driver._check(
                self.driver.lib.cuMemcpyDtoH_v2(
                    guarded.ctypes.data_as(ctypes.c_void_p), output_base.value, guarded.nbytes
                ),
                "cuMemcpyDtoH_v2(output canary)",
            )
            prefix_ok = bool(np.all(guarded[:CANARY_BYTES] == CANARY_VALUE))
            suffix_ok = bool(np.all(guarded[-CANARY_BYTES:] == CANARY_VALUE))
            output = guarded[CANARY_BYTES:CANARY_BYTES + output_bytes].copy().view(np.float32)
            output = output.reshape(inputs.shape.m, inputs.shape.n)
            return output, {
                "prefix_canary_ok": prefix_ok,
                "suffix_canary_ok": suffix_ok,
                "guard_bytes_each_side": CANARY_BYTES,
            }
        finally:
            for pointer in reversed(allocations):
                self.driver._check(self.driver.lib.cuMemFree_v2(pointer.value), "cuMemFree_v2")

    def close(self) -> None:
        if self.module:
            self.driver._check(self.driver.lib.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = ctypes.c_void_p()

    def __enter__(self) -> "GemmModule":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


SCALE_LEVELS = torch.tensor([
    0.0, 2.0 ** -9, 2.0 ** -6, 0.125, 0.5, 1.0, 2.0, 4.0,
    8.0, 16.0, 64.0, 256.0, 384.0, 448.0,
], dtype=torch.float32)


def make_inputs(shape_tuple: tuple[int, int, int], pattern: str,
                *, seed: int = 20260813) -> CanonicalInputs:
    shape = GemmShape(*shape_tuple)
    m = torch.arange(shape.m).reshape(shape.m, 1)
    k_a = torch.arange(shape.k).reshape(1, shape.k)
    k_b = torch.arange(shape.k).reshape(shape.k, 1)
    n = torch.arange(shape.n).reshape(1, shape.n)
    if pattern == "all_zero":
        a = torch.zeros((shape.m, shape.k), dtype=torch.uint8)
        b = torch.zeros((shape.k, shape.n), dtype=torch.uint8)
    elif pattern == "deterministic_random":
        generator = torch.Generator().manual_seed(seed + shape.m * 3 + shape.n * 5 + shape.k * 7)
        a = torch.randint(0, 16, (shape.m, shape.k), generator=generator, dtype=torch.uint8)
        b = torch.randint(0, 16, (shape.k, shape.n), generator=generator, dtype=torch.uint8)
    elif pattern == "alternating_signs":
        a = (((m + k_a) % 8) | (((m + k_a) % 2) << 3)).to(torch.uint8)
        b = (((k_b + 3 * n) % 8) | (((k_b + n) % 2) << 3)).to(torch.uint8)
    else:
        a = (((3 * m + 5 * k_a) % 8)
             | ((((m + 2 * k_a) % 3) == 0).to(torch.int64) << 3)).to(torch.uint8)
        b = (((7 * k_b + 3 * n) % 8)
             | ((((2 * k_b + n) % 5) == 0).to(torch.int64) << 3)).to(torch.uint8)

    blocks = shape.k_blocks
    a_scales = torch.ones((shape.m, blocks), dtype=torch.float32)
    b_scales = torch.ones((shape.n, blocks), dtype=torch.float32)
    alpha_a, alpha_b = 1.0, 1.0
    block = torch.arange(blocks).reshape(1, blocks)
    if pattern in {"per_k_block_scales", "row_column_scales"}:
        a_scales = SCALE_LEVELS[3 + ((m + block) % 7)]
        b_scales = SCALE_LEVELS[3 + ((n.T + 2 * block) % 7)]
    elif pattern == "scale_extremes":
        indices = torch.tensor([1, 2, 5, 13])
        a_scales = SCALE_LEVELS[indices[block % 4]].expand(shape.m, blocks).clone()
        b_scales = SCALE_LEVELS[indices[(block + 2) % 4]].expand(shape.n, blocks).clone()
    elif pattern == "zero_scales":
        a_scales[:, ::2] = 0.0
        b_scales[:, 1::2] = 0.0
    elif pattern == "global_scales":
        alpha_a, alpha_b = 0.125, 3.5

    return CanonicalInputs(
        shape=shape,
        packed_a=pack_canonical_a(a, shape),
        packed_b=pack_canonical_b(b, shape),
        a_scales=pack_canonical_a_scales(a_scales, shape),
        b_scales=pack_canonical_b_scales(b_scales, shape),
        alpha_a=alpha_a,
        alpha_b=alpha_b,
    )


CASE_MATRIX = [
    ((16, 8, 64), "all_zero"),
    ((16, 8, 128), "per_k_block_scales"),
    ((16, 8, 192), "alternating_signs"),
    ((32, 16, 64), "layout_sensitive"),
    ((64, 32, 128), "deterministic_random"),
    ((128, 64, 256), "row_column_scales"),
    ((1, 1, 1), "deterministic_random"),
    ((15, 7, 63), "alternating_signs"),
    ((17, 9, 65), "layout_sensitive"),
    ((31, 15, 127), "per_k_block_scales"),
    ((33, 17, 129), "global_scales"),
    ((256, 256, 512), "deterministic_random"),
    ((16, 8, 192), "scale_extremes"),
    ((17, 9, 65), "zero_scales"),
]


def _metrics(actual: torch.Tensor, expected: torch.Tensor,
             allowed: torch.Tensor) -> dict:
    error = (actual - expected).abs()
    relative = torch.where(
        expected != 0,
        error / expected.abs(),
        torch.where(error == 0, torch.zeros_like(error),
                    torch.full_like(error, float("inf"))),
    )
    mismatch = error > allowed
    coordinates = torch.nonzero(mismatch, as_tuple=False)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float(relative.max()),
        "tolerance_mismatch_count": int(mismatch.sum()),
        "bitwise_mismatch_count": int((actual != expected).sum()),
        "first_mismatch_coordinates": coordinates[:8].tolist(),
        "max_allowed_error": float(allowed.max()),
    }


def run_case(module: GemmModule, inputs: CanonicalInputs,
             a_format: str, b_format: str) -> dict:
    scalar_raw, scalar = sequential_fp32_gemm(inputs, a_format, b_format)
    vector_raw, vector = vectorized_packed_gemm(inputs, a_format, b_format)
    fp64 = decoded_fp64_gemm(inputs, a_format, b_format)
    hardware_np, guard = module.run(inputs)
    hardware = torch.from_numpy(hardware_np.copy())
    allowed = fp32_comparison_bound(inputs, a_format, b_format, scalar)
    fp64_error = (hardware.double() - fp64).abs()
    comparisons = {
        "scalar_vs_vectorized": _metrics(vector, scalar, allowed),
        "scalar_vs_hardware": _metrics(hardware, scalar, allowed),
        "vectorized_vs_hardware": _metrics(hardware, vector, allowed),
    }
    passed = (all(value["tolerance_mismatch_count"] == 0
                  for value in comparisons.values())
              and guard["prefix_canary_ok"] and guard["suffix_canary_ok"])
    return {
        "shape": [inputs.shape.m, inputs.shape.n, inputs.shape.k],
        "padded_shape": [inputs.shape.mp, inputs.shape.np, inputs.shape.kp],
        "alpha_a": inputs.alpha_a,
        "alpha_b": inputs.alpha_b,
        **comparisons,
        "fp64_sanity": {
            "max_absolute_error": float(fp64_error.max()),
            "mean_absolute_error": float(fp64_error.mean()),
        },
        "output_guard": guard,
        "passed": passed,
    }


def run_positive_matrix() -> dict:
    report = {}
    with CudaDriver() as driver:
        for variant, (a_format, b_format) in PAIRINGS.items():
            cases = []
            with GemmModule(driver, VARIANT_PATHS[variant]) as module:
                for index, (shape, pattern) in enumerate(CASE_MATRIX):
                    inputs = make_inputs(shape, pattern, seed=20260813 + index * 101)
                    result = run_case(module, inputs, a_format, b_format)
                    cases.append({"pattern": pattern, **result})
            report[f"{a_format}_x_{b_format}"] = {
                "variant": variant,
                "cases": cases,
                "passed": all(case["passed"] for case in cases),
            }
    report["passed"] = all(value["passed"] for key, value in report.items()
                           if key != "passed")
    return report


def _numerical_negative(module: GemmModule, actual_inputs: CanonicalInputs,
                        expected_inputs: CanonicalInputs, a_format: str,
                        b_format: str, *, validate: bool = True) -> dict:
    _, expected = sequential_fp32_gemm(expected_inputs, a_format, b_format)
    allowed = fp32_comparison_bound(
        expected_inputs, a_format, b_format, expected
    )
    hardware_np, guard = module.run(actual_inputs, validate=validate)
    metrics = _metrics(torch.from_numpy(hardware_np.copy()), expected, allowed)
    return {
        **metrics,
        "output_guard": guard,
        "caught": metrics["tolerance_mismatch_count"] > 0,
    }


def run_negative_controls() -> dict:
    controls = {}
    with CudaDriver() as driver:
        layout = make_inputs((32, 16, 64), "layout_sensitive")
        b_codes = unpack_canonical_b(layout.packed_b, layout.shape)[
            :layout.shape.k, :layout.shape.n
        ]
        wrong_b_row_major = pack_nibbles(
            b_codes.contiguous().reshape(-1)
        ).reshape(layout.shape.np, layout.shape.kp // 2)
        with GemmModule(driver, VARIANT_PATHS["11"]) as module:
            controls["b_row_major_instead_of_column_major"] = _numerical_negative(
                module, replace(layout, packed_b=wrong_b_row_major),
                layout, "e0m3", "e0m3"
            )

            swapped_a = ((layout.packed_a & 0xF) << 4) | (layout.packed_a >> 4)
            controls["a_high_low_nibbles_swapped"] = _numerical_negative(
                module, replace(layout, packed_a=swapped_a),
                layout, "e0m3", "e0m3"
            )
            swapped_b = ((layout.packed_b & 0xF) << 4) | (layout.packed_b >> 4)
            controls["b_high_low_nibbles_swapped"] = _numerical_negative(
                module, replace(layout, packed_b=swapped_b),
                layout, "e0m3", "e0m3"
            )

        scale_case = make_inputs((32, 16, 128), "row_column_scales")
        wrong_a_scales = torch.roll(scale_case.a_scales, shifts=1, dims=1)
        wrong_b_scales = torch.roll(scale_case.b_scales, shifts=1, dims=1)
        with GemmModule(driver, VARIANT_PATHS["01"]) as module:
            controls["a_scale_block_offset_plus_one"] = _numerical_negative(
                module, replace(scale_case, a_scales=wrong_a_scales),
                scale_case, "e0m3", "e2m1"
            )
            controls["b_scale_block_offset_plus_one"] = _numerical_negative(
                module, replace(scale_case, b_scales=wrong_b_scales),
                scale_case, "e0m3", "e2m1"
            )

        global_case = make_inputs((33, 17, 129), "global_scales")
        unscaled_inputs = replace(global_case, alpha_a=1.0, alpha_b=1.0)
        _, unscaled = sequential_fp32_gemm(unscaled_inputs, "e2m1", "e2m1")
        with GemmModule(driver, VARIANT_PATHS["00"]) as module:
            correct_output, guard = module.run(global_case)
        unscaled_bound = fp32_comparison_bound(
            unscaled_inputs, "e2m1", "e2m1", unscaled,
        )
        global_metrics = _metrics(
            torch.from_numpy(correct_output.copy()), unscaled, unscaled_bound
        )
        controls["forgot_global_scale"] = {
            **global_metrics,
            "output_guard": guard,
            "caught": global_metrics["tolerance_mismatch_count"] > 0,
        }

        padding = make_inputs((17, 9, 65), "layout_sensitive")
        bad_a = padding.packed_a.clone()
        bad_b = padding.packed_b.clone()
        # K=65: padded k=65 is the high nibble of byte 32.  Make both
        # operands nonzero so the invalid padding has a numerical effect.
        bad_a[0, 32] |= 0x20
        bad_b[0, 32] |= 0x20
        invalid_padding = replace(padding, packed_a=bad_a, packed_b=bad_b)
        rejected = False
        rejection = ""
        try:
            validate_canonical(invalid_padding)
        except ValueError as error:
            rejected = True
            rejection = str(error)
        with GemmModule(driver, VARIANT_PATHS["11"]) as module:
            numerical = _numerical_negative(
                module, invalid_padding, padding, "e0m3", "e0m3", validate=False
            )
        controls["nonzero_k_padding"] = {
            **numerical,
            "host_contract_rejected": rejected,
            "host_rejection": rejection,
            "caught": rejected and numerical["caught"],
        }

        e0_payload = make_inputs((32, 16, 128), "layout_sensitive")
        with GemmModule(driver, VARIANT_PATHS["00"]) as module:
            controls["e0_payload_on_e2_binary"] = _numerical_negative(
                module, e0_payload, e0_payload, "e0m3", "e0m3"
            )
        with GemmModule(driver, VARIANT_PATHS["10"]) as module:
            controls["wrong_pairing_binary"] = _numerical_negative(
                module, e0_payload, e0_payload, "e0m3", "e2m1"
            )

    from kernels.blackwell_e0_probe.gemm_probe.patch_static_gemm import (
        analyze_baseline,
        validate_variant,
        variant_instruction,
    )
    baseline = (BUILD_ROOT / "baseline_00.cubin").read_bytes()
    analysis = analyze_baseline(baseline)
    offset = analysis["instruction_file_offsets"][0]
    partially_patched = bytearray(baseline)
    partially_patched[offset:offset + 16] = variant_instruction("01")
    partial_rejected = False
    partial_error = ""
    try:
        validate_variant(
            baseline, bytes(partially_patched), "11",
            analysis["instruction_file_offsets"],
        )
    except ValueError as error:
        partial_rejected = True
        partial_error = str(error)
    controls["partial_omma_patch"] = {
        "expected_target_count": 1,
        "patched_candidate_bits": [78],
        "required_candidate_bits": [78, 79],
        "rejected": partial_rejected,
        "error": partial_error,
        "caught": partial_rejected,
    }
    controls["passed"] = all(value["caught"] for key, value in controls.items()
                             if key != "passed")
    return controls


def output_digest(module: GemmModule, inputs: CanonicalInputs) -> str:
    output, guard = module.run(inputs)
    if not guard["prefix_canary_ok"] or not guard["suffix_canary_ok"]:
        raise RuntimeError("output canary corruption")
    return hashlib.sha256(output.tobytes()).hexdigest()


STABILITY_CASES = [
    ((16, 8, 64), "deterministic_random"),
    ((17, 9, 65), "layout_sensitive"),
    ((128, 64, 256), "row_column_scales"),
]


def stability_digest(module: GemmModule) -> str:
    digest = hashlib.sha256()
    for shape, pattern in STABILITY_CASES:
        inputs = make_inputs(shape, pattern)
        output, guard = module.run(inputs)
        if not guard["prefix_canary_ok"] or not guard["suffix_canary_ok"]:
            raise RuntimeError("output canary corruption during stability replay")
        digest.update(output.tobytes())
    return digest.hexdigest()


def run_stability() -> dict:
    reports = {}
    with CudaDriver() as driver:
        for variant, (a_format, b_format) in PAIRINGS.items():
            reference_outputs = []
            with GemmModule(driver, VARIANT_PATHS[variant]) as module:
                for shape, pattern in STABILITY_CASES:
                    output, guard = module.run(make_inputs(shape, pattern))
                    if not guard["prefix_canary_ok"] or not guard["suffix_canary_ok"]:
                        raise RuntimeError("output canary corruption")
                    reference_outputs.append(output.copy())
                deterministic = True
                for _ in range(99):
                    for (shape, pattern), expected in zip(
                        STABILITY_CASES, reference_outputs
                    ):
                        output, guard = module.run(make_inputs(shape, pattern))
                        deterministic &= np.array_equal(output, expected)
                        deterministic &= guard["prefix_canary_ok"]
                        deterministic &= guard["suffix_canary_ok"]
                expected_digest = stability_digest(module)

            reload_hashes = []
            for _ in range(10):
                with GemmModule(driver, VARIANT_PATHS[variant]) as module:
                    reload_hashes.append(stability_digest(module))

            process_hashes = []
            for _ in range(10):
                completed = subprocess.run(
                    [
                        sys.executable, "-m",
                        "kernels.blackwell_e0_probe.gemm_probe.run_gemm_probe",
                        "--stability-digest", variant,
                    ],
                    check=True, capture_output=True, text=True,
                    cwd=Path(__file__).resolve().parents[3],
                )
                process_hashes.append(completed.stdout.strip())
            reports[f"{a_format}_x_{b_format}"] = {
                "variant": variant,
                "shapes": [list(shape) for shape, _ in STABILITY_CASES],
                "same_process_repetitions_per_shape": 100,
                "same_process_deterministic": deterministic,
                "module_reload_count": len(reload_hashes),
                "module_reload_hashes": reload_hashes,
                "new_process_count": len(process_hashes),
                "new_process_hashes": process_hashes,
                "expected_digest": expected_digest,
                "passed": (
                    deterministic
                    and all(value == expected_digest for value in reload_hashes)
                    and all(value == expected_digest for value in process_hashes)
                ),
            }
    reports["passed"] = all(value["passed"] for key, value in reports.items()
                            if key != "passed")
    return reports


def single_case(variant: str, shape_text: str, pattern: str) -> dict:
    shape = tuple(int(value) for value in shape_text.split("x"))
    if len(shape) != 3:
        raise ValueError("shape must be MxNxK")
    inputs = make_inputs(shape, pattern)
    with CudaDriver() as driver:
        with GemmModule(driver, VARIANT_PATHS[variant]) as module:
            return run_case(module, inputs, *PAIRINGS[variant])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=BUILD_ROOT / "gemm_results.json")
    parser.add_argument("--single-case", nargs=3, metavar=("VARIANT", "MxNxK", "PATTERN"))
    parser.add_argument("--digest", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--stability-digest", choices=("00", "01", "10", "11"))
    args = parser.parse_args()
    if args.stability_digest:
        with CudaDriver() as driver:
            with GemmModule(driver, VARIANT_PATHS[args.stability_digest]) as module:
                print(stability_digest(module))
        return
    if args.stability:
        report = run_stability()
    elif args.negative_controls:
        report = run_negative_controls()
    elif args.single_case:
        variant, shape, pattern = args.single_case
        if args.digest:
            inputs = make_inputs(tuple(int(value) for value in shape.split("x")), pattern)
            with CudaDriver() as driver:
                with GemmModule(driver, VARIANT_PATHS[variant]) as module:
                    print(output_digest(module, inputs))
            return
        report = single_case(variant, shape, pattern)
    else:
        report = run_positive_matrix()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
