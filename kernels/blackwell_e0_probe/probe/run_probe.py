#!/usr/bin/env python3
"""Compare one public SM120 NVFP4 MMA against the frozen CPU references."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch

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
DEFAULT_LIBRARY = PROBE_ROOT / "build" / "libe2m1_mma_probe.so"
DEFAULT_GOLDEN = PROBE_ROOT / "golden" / "e2m1_x_e2m1.json"
DEFAULT_REPORT = PROBE_ROOT / "build" / "probe_results.json"


class HardwareProbe:
    def __init__(self, library: Path):
        self._library = ctypes.CDLL(str(library))
        self._run = self._library.blackwell_e2m1_m16n8k64
        u8_ptr = ctypes.POINTER(ctypes.c_uint8)
        self._run.argtypes = [u8_ptr, u8_ptr, u8_ptr, u8_ptr,
                              ctypes.POINTER(ctypes.c_float)]
        self._run.restype = ctypes.c_int

    def __call__(self, packed_a, packed_b, a_scales, b_scales) -> torch.Tensor:
        arrays = [
            np.ascontiguousarray(value.cpu().numpy(), dtype=np.uint8)
            for value in (packed_a, packed_b, a_scales, b_scales)
        ]
        output = np.empty((16, 8), dtype=np.float32)
        status = self._run(
            *(value.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
              for value in arrays),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status:
            raise RuntimeError(f"CUDA probe failed with cudaError_t={status}")
        return torch.from_numpy(output.copy())


def _u8(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.uint8)


def _case_tensors(case: dict):
    return (
        _u8(case["packed_a_bytes"]),
        _u8(case["packed_b_bytes"]),
        _u8(case["a_block_scale_bytes"]),
        _u8(case["b_block_scale_bytes"]),
    )


def _single_nonzero_case() -> dict:
    a = torch.zeros(A_SHAPE, dtype=torch.uint8)
    b = torch.zeros(B_SHAPE, dtype=torch.uint8)
    a[3, 17] = 2  # E2M1 +1
    b[17, 5] = 2  # E2M1 +1
    a_scales = torch.ones(A_SCALE_SHAPE)
    b_scales = torch.ones(B_SCALE_SHAPE)
    return {
        "name": "single_nonzero",
        "packed_a_bytes": pack_a(a).tolist(),
        "packed_b_bytes": pack_b(b).tolist(),
        "a_block_scale_bytes": pack_a_scales(a_scales).tolist(),
        "b_block_scale_bytes": pack_b_scales(b_scales).tolist(),
    }


def _rounding_bound(pa, pb, sa, sb) -> torch.Tensor:
    a, b = decoded_operands(pa, pb, "e2m1", "e2m1")
    a_scale = unpack_a_scales(sa).repeat_interleave(K_BLOCK, dim=1)
    b_scale = unpack_b_scales(sb).repeat_interleave(K_BLOCK, dim=1).T
    sum_absolute_products = torch.matmul(
        (a * a_scale).abs(), (b * b_scale).abs()
    )
    unit_roundoff = 2.0**-24
    gamma_64 = (64 * unit_roundoff) / (1 - 64 * unit_roundoff)
    # Both the sequential scalar reference and the hardware reduction are
    # bounded relative to the exact dot product; compare them with 2*gamma.
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
        "mismatch_element_count": int(mismatches.sum()),
        "bitwise_mismatch_element_count": int((actual != expected).sum()),
        "first_mismatching_coordinates": coordinates[:8].tolist(),
        "max_allowed_rounding_error": float(allowed.max()),
    }


def _run_case(probe: HardwareProbe, case: dict) -> dict:
    pa, pb, sa, sb = _case_tensors(case)
    scalar, _ = scalar_mma(pa, pb, sa, sb, "e2m1", "e2m1", 1.0, 1.0)
    vector, _ = vectorized_mma(pa, pb, sa, sb, "e2m1", "e2m1", 1.0, 1.0)
    hardware = probe(pa, pb, sa, sb)
    allowed = _rounding_bound(pa, pb, sa, sb)
    scalar_vector = _metrics(vector, scalar, allowed)
    scalar_hardware = _metrics(hardware, scalar, allowed)
    vector_hardware = _metrics(hardware, vector, allowed)
    passed = (scalar_vector["mismatch_element_count"] == 0 and
              scalar_hardware["mismatch_element_count"] == 0 and
              vector_hardware["mismatch_element_count"] == 0)
    return {
        "name": case["name"],
        "global_fp32_scales": [1.0, 1.0],
        "scalar_vs_vectorized": scalar_vector,
        "scalar_vs_hardware": scalar_hardware,
        "vectorized_vs_hardware": vector_hardware,
        "passed": passed,
    }


def run_all_cases(library: Path = DEFAULT_LIBRARY,
                  golden: Path = DEFAULT_GOLDEN) -> dict:
    document = json.loads(golden.read_text())
    if document["pairing"] != "e2m1_x_e2m1":
        raise RuntimeError("probe must only consume the E2M1 x E2M1 golden")
    probe = HardwareProbe(library)
    cases = [_single_nonzero_case(), *document["cases"]]
    results = [_run_case(probe, case) for case in cases]

    layout_case = next(case for case in cases if case["name"] == "layout_sensitive")
    pa, pb, sa, sb = _case_tensors(layout_case)
    scalar, _ = scalar_mma(pa, pb, sa, sb, "e2m1", "e2m1", 1.0, 1.0)
    wrong_b = ((pb & 0xf) << 4) | (pb >> 4)
    wrong_output = probe(pa, wrong_b, sa, sb)
    negative_metrics = _metrics(
        wrong_output, scalar, _rounding_bound(pa, pb, sa, sb)
    )
    negative_control = {
        "name": "layout_sensitive_b_nibbles_swapped",
        "expected_to_fail": True,
        **negative_metrics,
        "caught": negative_metrics["mismatch_element_count"] > 0,
    }

    return {
        "contract": "single m16n8k64 E2M1xE2M1 UE4M3 MMA, FP32 accumulator",
        "rounding_rule": (
            "elementwise abs(error) <= 2*gamma_64*sum(abs(products)), "
            "gamma_64=(64*2^-24)/(1-64*2^-24)"
        ),
        "positive_cases": results,
        "negative_control": negative_control,
        "passed": all(case["passed"] for case in results)
                  and negative_control["caught"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_all_cases(args.library, args.golden)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
