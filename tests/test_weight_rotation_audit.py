import os

import pytest
import torch

from utils.hardware_weight_fp4 import (
    frozen_block_scales,
    hardware_global_scale,
    quantize_with_frozen_scales,
)
from utils.weight_mixfp4 import hessian_trace_loss
from utils.weight_rotation_audit import (
    analyze_weight_basis,
    assert_files_unchanged,
    file_snapshot,
    logical_tile_to_stored_slice,
    tile_scores,
    transform_weight_hessian_pair,
    weight_candidate,
)


def orthogonal(dim):
    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return q


def block_rotation(low, high):
    return torch.block_diag(orthogonal(low), torch.eye(high))


def test_hidden_identity_random_transform_and_output_equivalence():
    torch.manual_seed(901)
    dim, high, out = 8, 2, 5
    u = orthogonal(dim)
    r = block_rotation(dim-high, high)
    weight = torch.randn(out, dim)
    x = torch.randn(23, dim)
    hpre = 2/x.shape[0] * x.T @ x
    pairs, validation = transform_weight_hessian_pair(
        weight, hpre, u, r, high, per_head=False
    )
    wid, hid = pairs["identity"]
    wrand, hrand = pairs["random"]
    expected = weight @ u[:, :dim-high]
    assert torch.allclose(wid, expected)
    assert torch.allclose(wrand, expected @ r[:dim-high, :dim-high])
    assert torch.allclose(
        hid, u[:, :dim-high].T @ hpre @ u[:, :dim-high], atol=1e-5
    )
    assert validation["unquantized_output_relative_max_error"] < 1e-5
    assert validation["weight_frobenius_relative_error"] < 1e-5
    assert validation["hessian_trace_relative_error"] < 1e-5


def test_per_head_transform_orientation_and_output_equivalence():
    torch.manual_seed(902)
    heads, dim, high, out = 2, 4, 1, 7
    u = torch.stack([orthogonal(dim) for _ in range(heads)])
    r = block_rotation(dim-high, high)
    weight = torch.randn(out, heads*dim)
    x = torch.randn(31, heads*dim)
    hpre = 2/x.shape[0] * x.T @ x
    pairs, validation = transform_weight_hessian_pair(
        weight, hpre, u, r, high, per_head=True
    )
    expected = torch.einsum(
        "ohd,hdk->ohk", weight.reshape(out, heads, dim), u[..., :dim-high]
    ).reshape(out, heads*(dim-high))
    assert torch.allclose(pairs["identity"][0], expected, atol=1e-5)
    assert validation["unquantized_output_relative_max_error"] < 1e-5
    assert validation["weight_frobenius_relative_error"] < 1e-5
    assert validation["hessian_trace_relative_error"] < 1e-5


def test_basis_specific_global_scale_and_hardware_candidate_regression():
    source_id = torch.tensor([[10.0, 0.0] + [0.0]*14])
    source_rand = source_id @ block_rotation(16, 0)
    assert torch.allclose(source_id.norm(), source_rand.norm(), atol=1e-5)
    assert hardware_global_scale(source_id) != hardware_global_scale(source_rand)
    for source in (source_id, source_rand):
        for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
            actual, scales, raw, _, _ = weight_candidate(source, fmt, rounded=True)
            alpha = hardware_global_scale(source)
            expected_scales, expected_raw = frozen_block_scales(source, fmt, alpha)
            expected = quantize_with_frozen_scales(source, fmt, alpha, expected_scales)
            assert torch.equal(actual, expected)
            assert torch.equal(scales, expected_scales)
            assert torch.equal(raw, expected_raw)


def test_exact_and_e4m3_rounded_scale_paths_are_separate():
    source = torch.tensor(
        [
            [0.13, 0.27, 0.51, 0.99] + [0.0] * 12,
            [1.37] + [0.0] * 15,
        ]
    )
    exact, exact_scale, _, _, _ = weight_candidate(
        source, "hardware-fixed-e2", rounded=False
    )
    rounded, rounded_scale, raw, _, _ = weight_candidate(
        source, "hardware-fixed-e2", rounded=True
    )
    assert torch.equal(exact_scale, torch.where(raw == 0, torch.ones_like(raw), raw))
    assert not torch.equal(exact_scale[0], rounded_scale[0])
    assert not torch.equal(exact[0], rounded[0])


def test_hessian_trace_matches_direct_sampled_activation_error():
    torch.manual_seed(903)
    rows, k, n = 29, 11, 6
    x = torch.randn(rows, k)
    hessian = 2/rows * x.T @ x
    source = torch.randn(n, k)
    quantized = source + .05*torch.randn_like(source)
    trace = hessian_trace_loss(source, quantized, hessian)
    direct = 2/rows * (x @ (source-quantized).T).double().square().sum()
    torch.testing.assert_close(trace, direct, rtol=1e-5, atol=1e-6)


def test_logical_64x8_to_stored_8x64_mapping():
    n_slice, k_slice = logical_tile_to_stored_slice(1, 2)
    assert (n_slice.start, n_slice.stop) == (16, 24)
    assert (k_slice.start, k_slice.stop) == (64, 128)
    stored = torch.arange(25*130).reshape(25, 130)
    assert stored[n_slice, k_slice].shape == (8, 64)


def test_tile_tie_selects_e2_and_partial_nk_is_safe():
    source = torch.zeros(9, 70, dtype=torch.bfloat16)
    scores = tile_scores(source, source.clone(), source.clone())
    assert scores["tile_count"] == 4
    assert scores["e0_tile_count"] == 0
    assert scores["e2_tile_count"] == 4
    assert scores["selected_sse"] == 0


def test_read_only_snapshot_detects_change(tmp_path):
    cache = tmp_path/"cache.pt"
    cache.write_bytes(b"immutable")
    snapshot = file_snapshot([cache])
    assert_files_unchanged(snapshot)
    os.utime(cache, ns=(cache.stat().st_atime_ns, cache.stat().st_mtime_ns+1))
    with pytest.raises(RuntimeError, match="changed input"):
        assert_files_unchanged(snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpu_audit_zero_partial_k_dtype_and_no_cpu_fallback(dtype):
    source = torch.zeros(9, 70, device="cuda", dtype=dtype)
    hessian = torch.eye(70, device="cuda")
    row, detail = analyze_weight_basis(source, hessian)
    assert row["block_count"] == 9*5
    assert row["zero_rate"] == 1
    assert row["rounded_raw_e2_loss"] == 0
    assert row["rounded_raw_e0_loss"] == 0
    assert row["e0_tile_count"] == 0
    assert sum(detail["normalized_magnitude_hist"]) == source.numel()


def test_production_audit_rejects_silent_cpu_fallback():
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        analyze_weight_basis(torch.randn(2, 16), torch.eye(16))
