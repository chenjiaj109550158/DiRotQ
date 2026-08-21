import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from utils.flux_nvfp4_w4a16 import (
    NVFP4W4A16Linear,
    install_states,
    is_target,
    quantize_weight,
)
from utils.hardware_weight_fp4 import (
    decode_packing_record,
    frozen_block_scales,
    hardware_global_scale,
    quantize_with_frozen_scales,
)


def test_nvfp4_w4a16_matches_existing_hardware_e2_primitive():
    torch.manual_seed(9)
    weight = torch.randn(7, 35, dtype=torch.bfloat16)
    record, report = quantize_weight(weight)
    alpha = hardware_global_scale(weight.float())
    scales, _ = frozen_block_scales(weight.float(), "hardware-fixed-e2", alpha)
    expected = quantize_with_frozen_scales(
        weight.float(), "hardware-fixed-e2", alpha, scales
    )
    actual = decode_packing_record(record, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert float(record["global_scale"]) == float(weight.float().abs().max() / 2688)
    assert record["group_size"] == 16
    assert report["padded_k"] == 48


def test_nvfp4_w4a16_zero_weight_is_safe_and_positive_zero_padded():
    weight = torch.zeros(3, 17, dtype=torch.bfloat16)
    record, report = quantize_weight(weight)
    assert float(record["global_scale"]) == 1.0
    assert torch.equal(record["block_scales"].float(), torch.ones(3, 2))
    # E2M1 uses the explicit zero codebook index 7 in each nibble (0x77).
    assert torch.equal(
        record["packed_payload"],
        torch.full_like(record["packed_payload"], 0x77),
    )
    assert torch.count_nonzero(decode_packing_record(record)) == 0
    assert report["saturation_count"] == 0


def test_nvfp4_w4a16_keeps_activation_native_and_matches_decoded_linear():
    torch.manual_seed(17)
    weight = torch.randn(9, 33, dtype=torch.bfloat16)
    bias = torch.randn(9, dtype=torch.bfloat16)
    record, _ = quantize_weight(weight)
    module = NVFP4W4A16Linear(
        record, bias=bias, runtime_dtype=torch.bfloat16, require_cuda=False
    )
    x = torch.randn(2, 5, 33, dtype=torch.bfloat16).transpose(0, 1)
    assert not x.is_contiguous()
    expected = F.linear(x, decode_packing_record(record, dtype=torch.bfloat16), bias)
    actual = module(x)
    assert actual.dtype == x.dtype and actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_target_families_only():
    linear = nn.Linear(4, 4)
    assert is_target("transformer_blocks.0.norm1.linear", linear)
    assert is_target("transformer_blocks.0.norm1_context.linear", linear)
    assert is_target("single_transformer_blocks.0.norm.linear", linear)
    assert not is_target("transformer_blocks.0.ff.net.2", linear)


def test_install_fails_closed_on_incomplete_coverage():
    root = nn.Module()
    root.norm1 = nn.Module()
    root.norm1.linear = nn.Linear(4, 4)
    try:
        install_states(root, {}, require_cuda=False)
    except RuntimeError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("incomplete adaptive-norm coverage was accepted")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_nvfp4_w4a16_cuda_reference_has_no_cpu_fallback():
    torch.manual_seed(23)
    weight = torch.randn(11, 37, dtype=torch.bfloat16)
    bias = torch.randn(11, dtype=torch.bfloat16, device="cuda")
    record, _ = quantize_weight(weight)
    module = NVFP4W4A16Linear(
        record, bias=bias, runtime_dtype=torch.bfloat16, require_cuda=True
    ).cuda()
    x = torch.randn(2, 37, dtype=torch.bfloat16, device="cuda")
    expected = F.linear(x, module.weight, module.bias)
    actual = module(x)
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
