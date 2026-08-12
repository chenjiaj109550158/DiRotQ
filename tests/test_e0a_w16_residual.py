import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dirotq_fused_unrotation_fast import _fused_forward_fast
from utils.quant_utils import (
    ActQuantWrapper,
    capture_transformed_w16_weights,
    install_transformed_w16_weights,
)
from utils.tilemixfp4_utils import fake_quantize_activation


def _orthogonal(dim):
    q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float32))
    return q


class _TinyModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer


def _configured(dtype=torch.bfloat16, dim=32, high=16, out_features=7):
    layer = ActQuantWrapper(
        nn.Linear(dim, out_features, bias=True).to(dtype=dtype)
    )
    layer.rotation = _orthogonal(dim).to(dtype)
    layer.quantizer.configure(
        bits=4,
        groupsize=16,
        sym=True,
        high_bits_length=high,
        quant_dtype="e0a-w16-residual",
    )
    return _TinyModel(layer), layer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_e0a_w16_matches_e0_low_times_original_transformed_weight(dtype):
    torch.manual_seed(711)
    model, layer = _configured(dtype=dtype)
    original_bias = layer.bias.detach().clone()
    captured = capture_transformed_w16_weights(model)
    transformed = captured["layer"].clone()

    # Simulate loading a standard fake-quant GPTQ state before installing W16.
    layer.module.weight.data.zero_()
    cache_state = {"layer.module.weight": torch.zeros_like(transformed)}
    assert install_transformed_w16_weights(model, captured, cache_state) == 1
    assert layer.e0a_w16_weight_ready and layer._unrot_fused
    assert torch.equal(layer.module.weight.cpu(), transformed)

    x = torch.randn(2, 3, 32, dtype=dtype)
    actual = _fused_forward_fast(layer, x)
    x_rot = (x.reshape(-1, 32) @ layer.rotation).reshape_as(x)
    low = 16
    q_low = fake_quantize_activation(x_rot[..., :low], "e0m3")
    expected_x = torch.cat((q_low, x_rot[..., low:]), dim=-1).to(dtype)
    expected = F.linear(expected_x, transformed, original_bias)
    atol = 2e-2 if dtype == torch.float16 else 1.5e-1
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=atol)


def test_e0a_w16_guard_and_standard_cache_coverage():
    model, layer = _configured()
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="original transformed"):
        layer.quantize_activation_for_linear(x)

    captured = capture_transformed_w16_weights(model)
    with pytest.raises(RuntimeError, match="provenance is missing"):
        install_transformed_w16_weights(model, captured, {})


def test_e0a_w16_rejects_untransformed_original_linear_weight():
    layer = ActQuantWrapper(nn.Linear(32, 4, bias=False).bfloat16())
    layer.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=16,
        quant_dtype="e0a-w16-residual",
    )
    with pytest.raises(RuntimeError, match="untransformed active weight"):
        capture_transformed_w16_weights(_TinyModel(layer))


def test_e0a_w16_fixed_e0_low_and_native_high_tail():
    model, layer = _configured(dtype=torch.bfloat16)
    captured = capture_transformed_w16_weights(model)
    install_transformed_w16_weights(
        model, captured, {"layer.module.weight": captured["layer"]}
    )
    x = torch.randn(32, 3, dtype=torch.bfloat16).T
    assert not x.is_contiguous()
    out = layer.quantize_activation_for_linear(x)
    expected = torch.cat(
        (fake_quantize_activation(x[..., :16], "e0m3"), x[..., 16:]), dim=-1
    ).to(x.dtype)
    assert torch.equal(out, expected)
    assert torch.equal(out[..., 16:], x[..., 16:])
    assert out.dtype == x.dtype and out.device == x.device


@pytest.mark.parametrize(
    ("model_dir", "dtype"),
    [("pixart-sigma", torch.float16), ("sana-1.6b", torch.bfloat16)],
)
def test_model_routes_e0a_w16_in_native_dtype(model_dir, dtype):
    path = Path(__file__).parents[1] / "models" / model_dir / "model_utils.py"
    spec = importlib.util.spec_from_file_location(f"{model_dir}_e0aw16", path)
    model_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_utils)

    active = ActQuantWrapper(nn.Linear(32, 8, bias=False).to(dtype=dtype))
    transformer = nn.Module()
    # Give the wrapper a SANA/PixArt-recognized name without a duplicate alias.
    transformer.transformer_blocks = nn.ModuleList([
        nn.ModuleDict({"attn1": nn.ModuleDict({"to_q": active})})
    ])
    cfg = {
        "quantization": {"a_bits": 4},
        "dims": {"head": 32, "hidden": 32, "intermediate": 64},
        "nvfp4": {"a_groupsize": 16, "a_groupsize_attn_out": 32},
    }
    model_utils.configure_quantizers_by_name(
        transformer, 16, 4, cfg, nvfp4=True,
        activation_format="e0a-w16-residual",
    )
    routed = transformer.transformer_blocks[0]["attn1"]["to_q"]
    assert routed.quantizer.quant_dtype == "e0a-w16-residual"
    assert routed.quantizer.bits == 4
    assert next(routed.parameters()).dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_e0a_w16_cuda_preserves_device_and_forbids_cpu_fallback(dtype):
    model, layer = _configured(dtype=dtype)
    captured = capture_transformed_w16_weights(model)
    install_transformed_w16_weights(
        model, captured, {"layer.module.weight": captured["layer"]}
    )
    x = torch.randn(2, 32, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="different devices"):
        layer.quantize_activation_for_linear(x)
    model = model.cuda()
    out = model.layer.quantize_activation_for_linear(x)
    assert out.device.type == "cuda" and out.dtype == dtype
    assert torch.isfinite(out).all()
