from dataclasses import replace

import pytest
import torch

from kernels.blackwell_e0_probe.gemm_probe.packing import (
    GemmShape,
    unpack_canonical_a,
    unpack_canonical_a_scales,
    unpack_canonical_b,
    unpack_canonical_b_scales,
    validate_canonical,
)
from kernels.blackwell_e0_probe.gemm_probe.patch_static_gemm import (
    analyze_baseline,
    patch_cubin,
    validate_variant,
    variant_instruction,
)
from kernels.blackwell_e0_probe.gemm_probe.run_gemm_probe import (
    BUILD_ROOT,
    make_inputs,
    run_negative_controls,
    run_positive_matrix,
)
from kernels.blackwell_e0_probe.gemm_probe.reference import (
    decoded_fp64_gemm,
    sequential_fp32_gemm,
    vectorized_packed_gemm,
)


BASELINE = BUILD_ROOT / "baseline_00.cubin"


@pytest.mark.parametrize("dimensions", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_zero_dimensions_are_rejected(dimensions):
    with pytest.raises(ValueError, match="positive"):
        GemmShape(*dimensions)


def test_canonical_padding_round_trip_and_safe_scales():
    inputs = make_inputs((17, 9, 65), "layout_sensitive")
    validate_canonical(inputs)
    shape = inputs.shape
    a = unpack_canonical_a(inputs.packed_a, shape)
    b = unpack_canonical_b(inputs.packed_b, shape)
    sa = unpack_canonical_a_scales(inputs.a_scales, shape)
    sb = unpack_canonical_b_scales(inputs.b_scales, shape)
    assert not torch.any(a[shape.m:, :])
    assert not torch.any(a[:shape.m, shape.k:])
    assert not torch.any(b[:, shape.n:])
    assert not torch.any(b[shape.k:, :shape.n])
    assert torch.all(sa[shape.m:, :] == 1)
    assert torch.all(sa[:shape.m, shape.k_blocks:] == 1)
    assert torch.all(sb[shape.n:, :] == 1)
    assert torch.all(sb[:shape.n, shape.k_blocks:] == 1)


def test_nonzero_padding_is_rejected():
    inputs = make_inputs((17, 9, 65), "layout_sensitive")
    bad_a = inputs.packed_a.clone()
    bad_a[0, 32] |= 0x20
    with pytest.raises(ValueError, match="positive-zero"):
        validate_canonical(replace(inputs, packed_a=bad_a))


def test_three_references_agree_on_padded_scaled_case():
    inputs = make_inputs((31, 15, 127), "per_k_block_scales")
    scalar_raw, scalar = sequential_fp32_gemm(inputs, "e0m3", "e2m1")
    vector_raw, vector = vectorized_packed_gemm(inputs, "e0m3", "e2m1")
    fp64 = decoded_fp64_gemm(inputs, "e0m3", "e2m1")
    assert torch.equal(scalar_raw, vector_raw)
    assert torch.equal(scalar, vector)
    assert torch.equal(scalar.double(), fp64)


def test_gemm_patcher_rejects_unknown_sha_and_partial_patch(tmp_path):
    original = BASELINE.read_bytes()
    analysis = analyze_baseline(original)
    assert len(analysis["instruction_file_offsets"]) == 1
    unknown = bytearray(original)
    unknown[100] ^= 1
    source = tmp_path / "unknown.cubin"
    source.write_bytes(unknown)
    with pytest.raises(ValueError, match="unknown GEMM baseline SHA-256"):
        patch_cubin(source, tmp_path / "output.cubin", "01")

    offset = analysis["instruction_file_offsets"][0]
    partial = bytearray(original)
    partial[offset:offset + 16] = variant_instruction("01")
    with pytest.raises(ValueError, match="outside complete OMMA patch"):
        validate_variant(original, bytes(partial), "11", [offset])


def test_all_static_pairings_and_case_matrix():
    report = run_positive_matrix()
    assert report["passed"]


def test_all_required_negative_controls():
    report = run_negative_controls()
    assert report["passed"]
