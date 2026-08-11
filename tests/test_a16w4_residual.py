import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dirotq_fused_unrotation_fast import _fused_forward_fast
from utils.quant_utils import ActQuantWrapper


def _load_model_utils(model_dir):
    path = Path(__file__).parents[1] / "models" / model_dir / "model_utils.py"
    spec = importlib.util.spec_from_file_location(f"{model_dir}_a16_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NamedTransformer:
    def __init__(self, entries):
        self.entries = entries

    def named_modules(self):
        yield "", self
        yield from self.entries


def _orthogonal(dim):
    q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float32))
    return q


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a16w4_fused_wrapper_matches_high_plus_cached_low_formula(dtype):
    torch.manual_seed(501)
    dim, high, out_features = 32, 16, 11
    low = dim - high
    module = nn.Linear(dim, out_features, bias=True).to(dtype=dtype)
    wrapper = ActQuantWrapper(module)
    wrapper.quantizer.configure(
        bits=4,
        groupsize=16,
        sym=True,
        high_bits_length=high,
        quant_dtype="a16w4-residual",
    )
    # Represents a successfully loaded fused cache: low columns are the
    # dequantized E2M1 GPTQ values and high columns remain native precision.
    cached_low = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0,
         -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0, 0.0],
        dtype=dtype,
    ).repeat(out_features, 1)
    cached_high = torch.randn(out_features, high, dtype=dtype)
    wrapper.module.weight.data.copy_(torch.cat((cached_low, cached_high), dim=1))
    wrapper.rotation = _orthogonal(dim).to(dtype)
    wrapper._unrot_fused = True
    wrapper.a16w4_weight_ready = True

    x = torch.randn(2, 3, dim, dtype=dtype)
    out = _fused_forward_fast(wrapper, x)
    x_rot = x.reshape(-1, dim) @ wrapper.rotation
    expected_low = F.linear(x_rot[:, :low], cached_low)
    expected_high = F.linear(x_rot[:, low:], cached_high, wrapper.bias)
    expected = (expected_low + expected_high).reshape_as(out)
    tolerance = 2e-2 if dtype == torch.float16 else 1.5e-1
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=tolerance)
    assert wrapper.quantizer.bits == 4
    assert wrapper._unrot_fused


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a16w4_low_residual_is_exact_native_dtype_passthrough(dtype):
    wrapper = ActQuantWrapper(nn.Linear(32, 5, bias=False).to(dtype=dtype))
    wrapper.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=7,
        quant_dtype="a16w4-residual",
    )
    wrapper.a16w4_weight_ready = True
    x = torch.randn(3, 2, 32, dtype=dtype).transpose(0, 1)
    assert not x.is_contiguous()
    out = wrapper.quantize_activation_for_linear(x)
    assert torch.equal(out, x)
    assert out.dtype == dtype and out.device == x.device and out.shape == x.shape


def test_a16w4_requires_loaded_gptq_weight_and_is_not_fp_reference_path():
    wrapper = ActQuantWrapper(nn.Linear(32, 4, bias=False))
    wrapper.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=16,
        quant_dtype="a16w4-residual",
    )
    assert wrapper.quantizer.bits == 4
    assert isinstance(wrapper, ActQuantWrapper)
    with pytest.raises(RuntimeError, match="loaded NVFP4 E2M1 GPTQ cache"):
        wrapper.quantize_activation_for_linear(torch.randn(2, 32))


@pytest.mark.parametrize(
    ("model_dir", "dtype"),
    [("pixart-sigma", torch.float16), ("sana-1.6b", torch.bfloat16)],
)
def test_model_routes_a16w4_with_native_dtype_and_preserves_skip_semantics(
    model_dir, dtype,
):
    model_utils = _load_model_utils(model_dir)
    active = ActQuantWrapper(nn.Linear(32, 8, bias=False).to(dtype=dtype))
    skipped = ActQuantWrapper(nn.Linear(32, 8, bias=False).to(dtype=dtype))
    transformer = _NamedTransformer([
        ("transformer_blocks.0.attn1.to_q", active),
        ("transformer_blocks.0.ff.net.2", skipped),
    ])
    cfg = {
        "quantization": {"a_bits": 4},
        "dims": {"head": 32, "hidden": 32, "intermediate": 64},
        "nvfp4": {"a_groupsize": 16, "a_groupsize_attn_out": 32},
    }
    model_utils.configure_quantizers_by_name(
        transformer,
        high_len_hidden=16,
        high_len_head=4,
        high_len_down=16,
        cfg=cfg,
        nvfp4=True,
        activation_format="a16w4-residual",
        skip_quant_layers=["ff.net.2"],
    )
    assert active.quantizer.quant_dtype == "a16w4-residual"
    assert active.quantizer.bits == 4
    assert active.quantizer.high_bits_length == 16
    assert next(active.parameters()).dtype == dtype
    assert skipped.quantizer.bits == 16


def test_random_pca_residual_basis_is_executed_before_a16_passthrough():
    torch.manual_seed(502)
    dim = 32
    module = nn.Linear(dim, 3, bias=False).half()
    wrapper = ActQuantWrapper(module)
    wrapper.quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=16,
        quant_dtype="a16w4-residual",
    )
    pca, random_r = _orthogonal(dim), _orthogonal(dim)
    wrapper.rotation = (pca @ random_r).half()
    wrapper._unrot_fused = True
    wrapper.a16w4_weight_ready = True
    x = torch.randn(2, dim, dtype=torch.float16)
    expected = F.linear(x @ wrapper.rotation, wrapper.module.weight)
    actual = _fused_forward_fast(wrapper, x)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a16w4_cuda_device_is_preserved_and_mismatch_fails(dtype):
    wrapper = ActQuantWrapper(nn.Linear(32, 4, bias=False).to(dtype=dtype))
    wrapper.quantizer.configure(
        bits=4, groupsize=16, sym=True, quant_dtype="a16w4-residual"
    )
    wrapper.a16w4_weight_ready = True
    x = torch.randn(2, 32, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="different devices"):
        wrapper.quantize_activation_for_linear(x)
    wrapper = wrapper.cuda()
    out = wrapper.quantize_activation_for_linear(x)
    assert torch.equal(out, x)
    assert out.device.type == "cuda" and out.dtype == dtype
