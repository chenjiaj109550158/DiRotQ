import stat

import pytest

from kernels.blackwell_e0_probe.e0_probe.patch_operand_format import (
    CANDIDATE_BITS,
    analyze_baseline,
    expected_variant_instruction,
    patch_cubin,
    validate_patched_binary,
)
from kernels.blackwell_e0_probe.e0_probe.run_e0_probe import (
    BUILD_ROOT,
    run_experiment,
)


BASELINE = BUILD_ROOT / "baseline_e2e2.cubin"


def test_patcher_generates_only_candidate_bit_diffs(tmp_path):
    original = BASELINE.read_bytes()
    analysis = analyze_baseline(original)
    instruction_offset = analysis["instruction_file_offset"]
    expected_bits = {
        "00": set(),
        "01": {78},
        "10": {79},
        "11": {78, 79},
    }
    for variant in expected_bits:
        output = tmp_path / f"variant_{variant}.cubin"
        manifest = patch_cubin(BASELINE, output, variant)
        assert set(manifest["instruction_bits_set"]) == expected_bits[variant]
        assert manifest["size_unchanged"]
        assert stat.S_IMODE(output.stat().st_mode) == 0o444
        patched = output.read_bytes()
        assert patched[instruction_offset:instruction_offset + 16] == (
            expected_variant_instruction(variant)
        )
        allowed_file_bits = {instruction_offset * 8 + bit for bit in CANDIDATE_BITS}
        assert set(manifest["whole_cubin_file_bit_diff"]).issubset(allowed_file_bits)


def test_patcher_rejects_unknown_sha_without_output(tmp_path):
    unknown = bytearray(BASELINE.read_bytes())
    unknown[0x100] ^= 1
    source = tmp_path / "unknown.cubin"
    source.write_bytes(unknown)
    output = tmp_path / "must_not_exist.cubin"
    with pytest.raises(ValueError, match="unknown baseline SHA-256"):
        patch_cubin(source, output, "01")
    assert not output.exists()


def test_invariant_rejects_any_non_candidate_instruction_bit():
    original = BASELINE.read_bytes()
    analysis = analyze_baseline(original)
    instruction_offset = analysis["instruction_file_offset"]
    patched = bytearray(original)
    patched[instruction_offset:instruction_offset + 16] = expected_variant_instruction("01")
    patched[instruction_offset + 2] ^= 1
    with pytest.raises(ValueError, match="outside the exact candidate-bit patch"):
        validate_patched_binary(original, bytes(patched), instruction_offset, "01")


def test_e0_diagnostics_pairings_encoding_and_negative_controls():
    report = run_experiment(include_stability=False)
    assert report["passed"]
    assert report["bit_mapping"]["bit78_operand"] == "A"
    assert report["bit_mapping"]["bit79_operand"] == "B"
    assert all(value["passed"] for value in report["pairing_suites"].values())
    assert report["negative_controls"]["passed"]
    assert report["encoding_audit"]["passed"]
