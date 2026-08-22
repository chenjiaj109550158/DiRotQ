import pytest
import torch
import torch.nn as nn

from apply_dirotq import _flux_transformer_only_inputs
from utils.flux_real_quant import configure_real_int4_activation_reuse
from utils.packed_int4_runtime import (
    PackedSplitInt4Linear,
    pack_signed_int4,
    quantize_activation_int4,
)
from utils.quant_utils import ActQuantWrapper


def test_schnell_synthetic_forward_omits_dev_only_guidance():
    schnell = _flux_transformer_only_inputs("cpu", guidance_embeds=False)
    dev = _flux_transformer_only_inputs("cpu", guidance_embeds=True)
    assert "guidance" not in schnell
    assert dev["guidance"].shape == (1,)


def _packed_linear(k_low=128, high=64, n=32, *, device="cpu"):
    codes = torch.randint(-8, 8, (n, k_low), dtype=torch.int8, device=device)
    return PackedSplitInt4Linear(
        pack_signed_int4(codes),
        torch.rand(n, k_low // 64, dtype=torch.bfloat16, device=device) + 0.01,
        logical_low_k=k_low,
        group_size=64,
        high_weight=torch.randn(n, high, dtype=torch.bfloat16, device=device),
        bias=torch.randn(n, dtype=torch.bfloat16, device=device),
        require_cuda=device != "cpu",
    )


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = ActQuantWrapper(nn.Linear(192, 32, bias=False))
        self.to_k = ActQuantWrapper(nn.Linear(192, 32, bias=False))
        self.to_v = ActQuantWrapper(nn.Linear(192, 32, bias=False))


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()


class _TinySingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_mlp = ActQuantWrapper(nn.Linear(192, 32, bias=False))
        self.attn = _Attention()


def test_qkv_reuse_configuration_requires_one_shared_frame():
    model = _Tiny()
    rotation = torch.eye(192)
    for module in (model.attn.to_q, model.attn.to_k, model.attn.to_v):
        module.rotation = rotation
        module.quantizer.high_bits_length = 64
        module.module = _packed_linear()
    report = configure_real_int4_activation_reuse(model)
    assert report == {
        "reuse_groups": 1, "reuse_layers": 3,
        "qkv_groups": 1, "mlp_qkv_groups": 0,
    }
    assert [model.attn.to_q._real_int4_reuse_role,
            model.attn.to_k._real_int4_reuse_role,
            model.attn.to_v._real_int4_reuse_role] == [0, 1, 2]


def test_qkv_reuse_rejects_distinct_frames():
    model = _Tiny()
    for module in (model.attn.to_q, model.attn.to_k, model.attn.to_v):
        module.rotation = torch.eye(192).clone()
        module.quantizer.high_bits_length = 64
        module.module = _packed_linear()
    assert configure_real_int4_activation_reuse(model) == {
        "reuse_groups": 0, "reuse_layers": 0,
        "qkv_groups": 0, "mlp_qkv_groups": 0,
    }


def test_shared_width_single_block_reuses_mlp_and_qkv_but_operator_does_not():
    width = _TinySingle()
    shared = torch.eye(192)
    width_modules = (
        width.proj_mlp, width.attn.to_q, width.attn.to_k, width.attn.to_v
    )
    for module in width_modules:
        module.rotation = shared
        module.quantizer.high_bits_length = 64
        module.module = _packed_linear()
    report = configure_real_int4_activation_reuse(width)
    assert report["mlp_qkv_groups"] == 1
    assert [module._real_int4_reuse_role for module in width_modules] == [0, 1, 2, 3]

    operator = _TinySingle()
    operator.proj_mlp.rotation = torch.eye(192)
    attention_frame = torch.eye(192)
    for module in (operator.attn.to_q, operator.attn.to_k, operator.attn.to_v):
        module.rotation = attention_frame
    operator_modules = (
        operator.proj_mlp, operator.attn.to_q, operator.attn.to_k, operator.attn.to_v
    )
    for module in operator_modules:
        module.quantizer.high_bits_length = 64
        module.module = _packed_linear()
    report = configure_real_int4_activation_reuse(operator)
    assert report["mlp_qkv_groups"] == 0
    assert report["qkv_groups"] == 1


def test_reuse_configuration_is_idempotent():
    model = _TinySingle()
    shared = torch.eye(192)
    modules = (model.proj_mlp, model.attn.to_q, model.attn.to_k, model.attn.to_v)
    for module in modules:
        module.rotation = shared
        module.quantizer.high_bits_length = 64
        module.module = _packed_linear()
    first = configure_real_int4_activation_reuse(model)
    second = configure_real_int4_activation_reuse(model)
    assert first == second == {
        "reuse_groups": 1, "reuse_layers": 4,
        "qkv_groups": 0, "mlp_qkv_groups": 1,
    }


def test_fused_packed_module_forbids_cpu():
    module = _packed_linear()
    activation = quantize_activation_int4(
        torch.randn(2, 128, dtype=torch.bfloat16), 64
    )
    with pytest.raises(RuntimeError, match="forbids.*CPU fallback"):
        module.forward_quantized(
            activation,
            torch.randn(2, 64, dtype=torch.bfloat16),
            kernel_mode="fused",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("logical_k", [64, 127, 3008])
def test_fused_activation_pack_matches_reference(logical_k):
    from utils.packed_int4_fused_triton import quantize_pack_u4

    torch.manual_seed(29 + logical_k)
    # Slice from a wider allocation to exercise the real non-contiguous
    # rotated-low layout without allowing an implicit contiguous copy.
    base = torch.randn(2, 7, logical_k + 64, device="cuda", dtype=torch.bfloat16)
    x = base[..., :logical_k]
    assert not x.is_contiguous() if logical_k != base.shape[-1] else True
    reference = quantize_activation_int4(x, 64)
    payload, scales, zeros, actual_k, padded_k = quantize_pack_u4(x, 64)
    torch.cuda.synchronize()
    assert actual_k == reference.logical_k
    assert padded_k == reference.padded_k
    assert torch.equal(payload, reference.payload)
    assert torch.equal(scales, reference.scales.reshape_as(scales))
    assert torch.equal(zeros, reference.zeros.reshape_as(zeros))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_activation_pack_zero_tensor():
    from utils.packed_int4_fused_triton import quantize_pack_u4

    x = torch.zeros(3, 129, device="cuda", dtype=torch.bfloat16)
    reference = quantize_activation_int4(x, 64)
    payload, scales, zeros, _, _ = quantize_pack_u4(x, 64)
    assert torch.equal(payload, reference.payload)
    assert torch.equal(scales, reference.scales.reshape_as(scales))
    assert torch.equal(zeros, reference.zeros.reshape_as(zeros))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_low_high_epilogue_matches_reference():
    torch.manual_seed(91)
    module = _packed_linear(k_low=128, high=64, n=35, device="cuda")
    x_low = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    x_high = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
    activation = quantize_activation_int4(x_low, 64)
    reference = module.forward_quantized(
        activation, x_high, kernel_mode="reference"
    )
    actual = module.forward_quantized(activation, x_high, kernel_mode="fused")
    torch.cuda.synchronize()
    diff = (actual.float() - reference.float()).abs()
    assert float(diff.max()) <= 0.25
    assert float(diff.mean()) <= 0.02
    assert actual.dtype == torch.bfloat16
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qkv_projected_activation_is_reused_once_and_released():
    from utils.flux_real_quant import (
        _project_and_pack_reused,
        configure_real_int4_activation_reuse,
        fused_reuse_stats,
    )

    model = _Tiny()
    rotation = torch.eye(192, device="cuda", dtype=torch.bfloat16)
    modules = (model.attn.to_q, model.attn.to_k, model.attn.to_v)
    for module in modules:
        module.rotation = rotation
        module.quantizer.high_bits_length = 64
        module.module = _packed_linear(device="cuda")
    assert configure_real_int4_activation_reuse(model)["qkv_groups"] == 1
    x = torch.randn(2, 5, 192, device="cuda", dtype=torch.bfloat16)
    projected = [_project_and_pack_reused(module, x) for module in modules]
    assert projected[0][0].payload.data_ptr() == projected[1][0].payload.data_ptr()
    assert projected[0][0].payload.data_ptr() == projected[2][0].payload.data_ptr()
    assert projected[0][1].data_ptr() == projected[1][1].data_ptr()
    assert fused_reuse_stats() == {
        "created": 1, "hits": 2, "misses": 0, "cleared": 1, "live_entries": 0
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qkv_reuse_accepts_inference_tensors_without_version_counter():
    from utils.flux_real_quant import (
        _project_and_pack_reused,
        configure_real_int4_activation_reuse,
        fused_reuse_stats,
    )

    model = _Tiny()
    rotation = torch.eye(192, device="cuda", dtype=torch.bfloat16)
    modules = (model.attn.to_q, model.attn.to_k, model.attn.to_v)
    for module in modules:
        module.rotation = rotation
        module.quantizer.high_bits_length = 64
        module.module = _packed_linear(device="cuda")
    configure_real_int4_activation_reuse(model)
    with torch.inference_mode():
        x = torch.randn(2, 5, 192, device="cuda", dtype=torch.bfloat16)
        projected = [_project_and_pack_reused(module, x) for module in modules]
    assert projected[0][0].payload.data_ptr() == projected[-1][0].payload.data_ptr()
    assert fused_reuse_stats() == {
        "created": 1, "hits": 2, "misses": 0, "cleared": 1, "live_entries": 0
    }
