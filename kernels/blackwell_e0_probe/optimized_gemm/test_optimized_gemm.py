from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kernels.blackwell_e0_probe.optimized_gemm import patch_optimized_gemm as patcher
from kernels.blackwell_e0_probe.optimized_gemm.run_correctness import (
    CUBINS,
    RUNNER,
    validate_allowlisted_cubin,
    wrong_variant_guard,
)


def test_pinned_contract_has_all_optimized_omma_slots() -> None:
    assert patcher.EXPECTED_OMMA_COUNT == 64
    assert len(patcher.OMMA_OFFSETS_IN_TEXT) == 64
    assert len(set(patcher.OMMA_OFFSETS_IN_TEXT)) == 64
    assert patcher.CANDIDATE_BITS == (78, 79)


def test_unknown_baseline_is_rejected() -> None:
    if not CUBINS["00"].is_file():
        pytest.skip("ignored optimized baseline has not been built")
    corrupted = bytearray(CUBINS["00"].read_bytes())
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="unknown optimized baseline"):
        patcher.analyze_baseline(bytes(corrupted))


def test_partial_slot_list_is_rejected() -> None:
    if not CUBINS["00"].is_file():
        pytest.skip("ignored optimized baseline has not been built")
    baseline = CUBINS["00"].read_bytes()
    analysis = patcher.analyze_baseline(baseline)
    with pytest.raises(ValueError, match="partial or duplicate"):
        patcher.validate_variant(
            baseline, baseline, "00", analysis["instruction_file_offsets"][:-1]
        )


@pytest.mark.parametrize("variant,bits", [("00", []), ("01", [78]), ("11", [78, 79])])
def test_allowlisted_variants_have_exact_all_slot_diff(variant: str,
                                                        bits: list[int]) -> None:
    if not CUBINS[variant].is_file():
        pytest.skip(f"ignored optimized variant {variant} has not been built")
    report = validate_allowlisted_cubin(CUBINS[variant], variant)
    assert report["omma_count"] == 64
    assert len(report["whole_cubin_file_bit_diff"]) == 64 * len(bits)


def test_wrong_pairing_is_rejected_before_launch() -> None:
    report = wrong_variant_guard()
    assert report["passed"]
    assert report["rejected_before_launch"]


def test_smoke_runner_canary_and_native_scale_mapping(tmp_path: Path) -> None:
    if not RUNNER.is_file() or not all(path.is_file() for path in CUBINS.values()):
        pytest.skip("ignored optimized runner/CUBINs have not been built")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        pytest.skip("requires an SM120 CUDA device")
    from kernels.blackwell_e0_probe.gemm_probe.reference import (
        fp32_comparison_bound,
        vectorized_packed_gemm,
    )
    from kernels.blackwell_e0_probe.gemm_probe.run_gemm_probe import make_inputs
    from kernels.blackwell_e0_probe.optimized_gemm.run_correctness import run_case

    inputs = make_inputs((17, 9, 65), "row_column_scales", seed=20260813)
    _, expected = vectorized_packed_gemm(inputs, "e0m3", "e0m3")
    allowed = fp32_comparison_bound(inputs, "e0m3", "e0m3", expected)
    report = run_case(
        "pytest_tail", inputs, "e0m3", "e0m3", expected, allowed, tmp_path
    )
    assert report["passed"]
    assert report["hardware_fp32_vs_packed_fp32"]["tolerance_mismatch_count"] == 0
    assert report["runner"]["canary_prefix_ok"]
    assert report["runner"]["canary_suffix_ok"]
