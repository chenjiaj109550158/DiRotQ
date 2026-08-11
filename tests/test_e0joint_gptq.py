from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils.e0joint_gptq import (
    collect_joint_moments,
    direct_joint_loss,
    e0joint_metadata_path,
    extract_fused_low_weight,
    project_low_activation,
    quadratic_joint_loss,
    quantize_e0_per_chunk,
    solve_compensated_weight,
    validate_e0joint_cache_path,
    validate_e0joint_metadata,
    validate_nvfp4_e2m1_weight,
    write_e0joint_metadata,
)
from utils.quant_utils import ActQuantWrapper, _quant_group_nvfp4, _rotate_and_split_W
from utils.tilemixfp4_utils import fake_quantize_e0m3


def _moments(a, z):
    return a.T @ a, z.T @ z, z.T @ a


def test_continuous_least_squares_reduces_joint_loss():
    torch.manual_seed(801)
    a = torch.randn(96, 12, dtype=torch.float64)
    z = a @ torch.diag(torch.linspace(0.7, 1.3, 12, dtype=torch.float64))
    w = torch.randn(12, 9, dtype=torch.float64)
    s, h, c = _moments(a, z)
    wc, damping, solver, _ = solve_compensated_weight(h, c, w, damp_pct=0.0)
    assert damping == 0.0 and solver == "cholesky"
    assert direct_joint_loss(a, z, w, wc) < direct_joint_loss(a, z, w, w)


def test_z_equal_a_degenerates_to_standard_problem():
    torch.manual_seed(802)
    a = torch.randn(64, 8, dtype=torch.float64)
    w = torch.randn(8, 5, dtype=torch.float64)
    s, h, c = _moments(a, a)
    wc, _, _, _ = solve_compensated_weight(h, c, w, damp_pct=0.0)
    torch.testing.assert_close(wc, w, rtol=1e-10, atol=1e-10)
    q = torch.randn_like(w)
    torch.testing.assert_close(
        quadratic_joint_loss(s, h, c, w, q),
        ((a @ (w - q)).square().sum()),
        rtol=1e-10,
        atol=1e-10,
    )


def test_quadratic_form_matches_direct_matmul_loss():
    torch.manual_seed(803)
    a = torch.randn(41, 10, dtype=torch.float64)
    z = torch.randn(41, 10, dtype=torch.float64)
    w = torch.randn(10, 7, dtype=torch.float64)
    q = torch.randn(10, 7, dtype=torch.float64)
    s, h, c = _moments(a, z)
    torch.testing.assert_close(
        quadratic_joint_loss(s, h, c, w, q),
        direct_joint_loss(a, z, w, q),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_e0_quantization_uses_independent_chunk_global_scales(dtype):
    torch.manual_seed(804)
    a = torch.randn(2, 3, 32, dtype=dtype)
    a[1].mul_(1000)
    actual = quantize_e0_per_chunk(a)
    expected = torch.stack([fake_quantize_e0m3(x) for x in a.unbind(0)])
    assert torch.equal(actual, expected)
    assert actual.shape == a.shape and actual.dtype == dtype
    assert torch.isfinite(actual).all()


def test_joint_and_standard_cache_paths_are_isolated(tmp_path):
    standard = tmp_path / "nvfp4_g16_gptq_model.pt"
    joint = tmp_path / "nvfp4_g16_e0joint_gptq_model.pt"
    validate_e0joint_cache_path(standard, False)
    validate_e0joint_cache_path(joint, True)
    with pytest.raises(ValueError, match="e0joint-tagged"):
        validate_e0joint_cache_path(standard, True)
    with pytest.raises(ValueError, match="only be loaded"):
        validate_e0joint_cache_path(joint, False)
    metadata = {"objective_version": "e0joint-v1", "activation_format": "e0m3"}
    write_e0joint_metadata(joint, metadata)
    assert e0joint_metadata_path(joint).exists()
    assert validate_e0joint_metadata(joint, metadata) == metadata


def test_runtime_weight_remains_legal_e2m1_and_high_branch_unchanged():
    torch.manual_seed(805)
    layer = ActQuantWrapper(nn.Linear(32, 7, bias=False).half())
    layer.rotation = torch.linalg.qr(torch.randn(32, 32))[0]
    layer.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=8, quant_dtype="e0m3"
    )
    low, high, stitch, _ = _rotate_and_split_W(layer, layer.module.weight)
    q = _quant_group_nvfp4(low, 16)
    assert validate_nvfp4_e2m1_weight(q, low, 16)
    fused = stitch(q).half()
    extracted_low, extracted_high = extract_fused_low_weight(layer, fused)
    torch.testing.assert_close(extracted_low, q, rtol=0, atol=2e-3)
    torch.testing.assert_close(extracted_high, high, rtol=0, atol=2e-3)
    x = torch.randn(2, 4, 24, dtype=torch.float16)
    expected_activation = torch.cat(
        (fake_quantize_e0m3(x[..., :16]), x[..., 16:]), dim=-1
    )
    assert torch.equal(layer.quantizer(x), expected_activation)


def test_cached_high_branch_comparison_uses_native_dtype_round_trip():
    torch.manual_seed(8051)
    layer = ActQuantWrapper(
        nn.Linear(32, 7, bias=False).to(dtype=torch.bfloat16)
    )
    layer.module.weight.data.mul_(64)
    layer.rotation = torch.linalg.qr(torch.randn(32, 32))[0]
    layer.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=8, quant_dtype="e0m3"
    )
    low, high, stitch, _ = _rotate_and_split_W(layer, layer.module.weight)
    cached = stitch(_quant_group_nvfp4(low, 16)).to(torch.bfloat16)
    _, cached_high = extract_fused_low_weight(layer, cached)
    # The pre-cast FP32 rotation can differ by much more than an arbitrary
    # 2e-3 tolerance while the cache is nevertheless the exact native-dtype
    # representation of the unchanged high branch.
    assert (high - cached_high).abs().max() > 2e-3
    assert torch.equal(high.to(torch.bfloat16), cached_high.to(torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_projection_dtype_device_and_noncontiguous(dtype):
    torch.manual_seed(806)
    layer = ActQuantWrapper(nn.Linear(32, 5, bias=False).to(dtype=dtype))
    layer.rotation = torch.linalg.qr(torch.randn(32, 32))[0]
    layer.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=8, quant_dtype="e0m3"
    )
    x = torch.randn(2, 3, 32, dtype=dtype).transpose(0, 1)
    assert not x.is_contiguous()
    projected = project_low_activation(layer, x)
    assert projected.shape == (3, 2, 24)
    assert projected.dtype == dtype and projected.device == x.device
    assert torch.isfinite(projected).all()


def test_nonfinite_inputs_and_reported_solve_fallback():
    bad = torch.zeros(1, 2, 16)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        quantize_e0_per_chunk(bad)
    h = torch.zeros(4, 4)
    c = torch.eye(4)
    w = torch.ones(4, 2)
    wc, damping, solver, attempts = solve_compensated_weight(
        h, c, w, damp_pct=0.0, max_cholesky_tries=1
    )
    assert torch.isfinite(wc).all()
    assert damping > 0 and solver == "solve-fallback" and attempts == 1


def test_calibration_refuses_cpu_fallback():
    with pytest.raises(RuntimeError, match="CPU fallback"):
        collect_joint_moments(
            nn.Linear(2, 2), Path("unused"), 1, 1, device="cpu"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_projection_and_e0_stay_on_device(dtype):
    layer = ActQuantWrapper(nn.Linear(32, 4, bias=False).to("cuda", dtype=dtype))
    layer.rotation = torch.linalg.qr(torch.randn(32, 32, device="cuda"))[0]
    layer.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=8, quant_dtype="e0m3"
    )
    x = torch.randn(2, 3, 32, device="cuda", dtype=dtype)
    a = project_low_activation(layer, x)
    z = quantize_e0_per_chunk(a)
    assert a.device.type == "cuda" and z.device.type == "cuda"
    assert z.dtype == dtype and torch.isfinite(z).all()
