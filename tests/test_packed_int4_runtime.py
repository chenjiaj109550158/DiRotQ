from __future__ import annotations

import torch
import pytest
from torch import nn

from utils.flux_real_quant import (
    exact_gptq_cache_from_states,
    real_int4_storage_report,
    resolve_readonly_provenance_sha256,
    validate_relocated_model_id,
    validate_states_against_fake_quant_cache,
)
from utils.gptq_utils import gptq_quantize_weights
from utils.packed_int4_runtime import (
    PackedSplitInt4Linear,
    decode_weight_int4,
    pack_signed_int4,
    pack_unsigned_int4,
    quantize_activation_int4,
    unpack_signed_int4,
    unpack_unsigned_int4,
)
from utils.quant_utils import ActQuantWrapper


def test_signed_and_unsigned_nibble_order_and_tail():
    signed = torch.tensor([[-8, -1, 0, 1, 7]], dtype=torch.int8)
    packed = pack_signed_int4(signed)
    assert packed.tolist()[0][0] == 0xF8  # earlier K element is low nibble
    assert torch.equal(unpack_signed_int4(packed, 5), signed)

    unsigned = torch.tensor([[0, 15, 2, 13, 7]], dtype=torch.uint8)
    packed_u = pack_unsigned_int4(unsigned)
    assert packed_u.tolist()[0][0] == 0xF0
    assert torch.equal(unpack_unsigned_int4(packed_u, 5), unsigned)


def test_activation_codes_decode_match_existing_asymmetric_fake_quant():
    gen = torch.Generator().manual_seed(7)
    base = torch.randn(2, 5, 130, generator=gen, dtype=torch.bfloat16)
    x = base.transpose(0, 1)  # deliberately non-contiguous
    wrapper = ActQuantWrapper(nn.Linear(130, 3, bias=False, dtype=torch.bfloat16))
    wrapper.quantizer.configure(bits=4, groupsize=64, sym=False)
    wrapper.quantizer.find_params(x)
    expected = wrapper.quantizer(x)

    encoded = quantize_activation_int4(x, group_size=64)
    actual = encoded.decode(x.dtype)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.equal(actual, expected)
    # K padding is positive-zero payload and never appears in the output.
    assert encoded.padded_k == 192


def test_zero_activation_matches_repository_zero_group_rule():
    x = torch.zeros(1, 3, 65, dtype=torch.bfloat16)
    encoded = quantize_activation_int4(x, group_size=64)
    assert torch.count_nonzero(encoded.decode(x.dtype)) == 0
    assert torch.isfinite(encoded.scales).all()
    padded_codes = encoded.codes()[:, 65:]
    padded_zeros = encoded.zeros.reshape(3, 2)[:, -1:].expand_as(padded_codes)
    assert torch.equal(padded_codes, padded_zeros.to(torch.uint8))


def test_packed_split_linear_matches_integer_reference_and_high_bias_once():
    gen = torch.Generator().manual_seed(11)
    m, low_k, high_k, out = 5, 64, 3, 7
    qweight = torch.randint(-8, 8, (out, low_k), generator=gen, dtype=torch.int8)
    scales = torch.rand(out, 1, generator=gen, dtype=torch.float32) * 0.05 + 0.01
    high = torch.randn(out, high_k, generator=gen, dtype=torch.bfloat16)
    bias = torch.randn(out, generator=gen, dtype=torch.bfloat16)
    module = PackedSplitInt4Linear(
        pack_signed_int4(qweight), scales,
        logical_low_k=low_k, group_size=64,
        high_weight=high, bias=bias, require_cuda=False,
    )
    x_low = torch.randn(m, low_k, generator=gen, dtype=torch.bfloat16)
    x_high = torch.randn(m, high_k, generator=gen, dtype=torch.bfloat16)
    result = module(x_low, x_high)

    activation = quantize_activation_int4(x_low, 64)
    centered = activation.codes().to(torch.int32) - activation.zeros.reshape(m, 1).to(torch.int32)
    integer = centered @ qweight.to(torch.int32).T
    expected = integer.float() * activation.scales.reshape(m, 1).float() * scales.T
    expected += torch.nn.functional.linear(x_high, high, None).float()
    expected += bias.float()
    assert torch.equal(result, expected.to(torch.bfloat16))


def test_production_module_refuses_silent_cpu_fallback():
    module = PackedSplitInt4Linear(
        torch.zeros(1, 32, dtype=torch.uint8),
        torch.ones(1, 1),
        logical_low_k=64,
        group_size=64,
        high_weight=None,
        bias=None,
        require_cuda=True,
    )
    try:
        module(torch.zeros(1, 64, dtype=torch.bfloat16))
    except RuntimeError as error:
        assert "CPU fallback" in str(error)
    else:
        raise AssertionError("packed production path silently ran on CPU")


def test_relocated_inference_provenance_is_fail_closed(tmp_path):
    source = tmp_path / "producer.pt"
    source.write_bytes(b"immutable producer")
    observed = resolve_readonly_provenance_sha256(source, None, label="producer")
    assert resolve_readonly_provenance_sha256(
        tmp_path / "not-transferred.pt", observed, label="producer"
    ) == observed
    with pytest.raises(RuntimeError, match="mismatch"):
        resolve_readonly_provenance_sha256(source, "0" * 64, label="producer")
    with pytest.raises(FileNotFoundError, match="provide its immutable SHA-256"):
        resolve_readonly_provenance_sha256(
            tmp_path / "not-transferred.pt", None, label="producer"
        )


def test_relocated_model_id_requires_same_exact_snapshot_revision():
    revision = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    validate_relocated_model_id(f"/producer/snapshots/{revision}", f"/receiver/{revision}")
    with pytest.raises(ValueError, match="model provenance mismatch"):
        validate_relocated_model_id(
            f"/producer/snapshots/{revision}", "/receiver/" + "0" * 40
        )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = ActQuantWrapper(nn.Linear(8, 4, bias=True))
        self.proj.quantizer.configure(bits=4, groupsize=4, sym=False)


def test_gptq_exports_actual_pre_bf16_codes_and_scales(tmp_path):
    torch.manual_seed(23)
    model = _TinyModel()
    original = model.proj.module.weight.detach().clone()
    states = {}
    summary = gptq_quantize_weights(
        model,
        {"proj": torch.eye(8)},
        bits=4,
        groupsize=4,
        sym=True,
        damp_pct=0.01,
        block_size=4,
        num_inv_tries=5,
        device="cpu",
        packed_state_out=states,
    )
    assert summary == {
        "gptq_layers": 1,
        "configured_rtn_layers": 0,
        "rtn_fallback_layers": 0,
        "packed_layers": 1,
        "packed_scale_dtype": "torch.float32",
    }
    state = states["proj"]
    decoded = decode_weight_int4(
        state["qweight"], state["weight_scales"], 8, 4, torch.float32
    )
    assert torch.allclose(decoded, model.proj.module.weight, atol=2e-6, rtol=0)
    expected_scale = original.reshape(4, 2, 4).abs().amax(-1) / 7
    assert torch.equal(state["weight_scales"], expected_scale)
    assert state["packing_max_abs_error_fp32"] <= 2e-6

    cache, report = exact_gptq_cache_from_states(
        states, expected_layers=1, gptq_summary=summary
    )
    assert cache["layers"] is states
    assert report["aggregate"]["payload_bytes"] == 16
    assert report["aggregate"]["scale_bytes"] == 32
    assert report["aggregate"]["source"] == "exact-pre-bf16-gptq-codes-and-frozen-scales"
    fake_path = tmp_path / "fake.pt"
    torch.save(model.state_dict(), fake_path)
    parity = validate_states_against_fake_quant_cache(states, fake_path)
    assert parity["bitwise_bf16_equal"]
    assert parity["unequal_elements"] == 0


def test_storage_report_separates_payload_scales_high_and_shared_frames():
    transformer = nn.Module()
    transformer.a = ActQuantWrapper(nn.Linear(8, 2, bias=False))
    transformer.b = ActQuantWrapper(nn.Linear(8, 2, bias=False))
    shared_rotation = torch.eye(8, dtype=torch.bfloat16)
    transformer.a.rotation = shared_rotation
    transformer.b.rotation = shared_rotation
    for wrapper in (transformer.a, transformer.b):
        wrapper.module = PackedSplitInt4Linear(
            torch.zeros(2, 2, dtype=torch.uint8),
            torch.ones(2, 1),
            logical_low_k=4,
            group_size=4,
            high_weight=torch.ones(2, 4, dtype=torch.bfloat16),
            bias=None,
            require_cuda=False,
        )
        delattr(wrapper, "weight")
        wrapper.register_parameter("weight", None)
        delattr(wrapper, "bias")
        wrapper.register_parameter("bias", None)
    report = real_int4_storage_report(transformer)
    assert report["packed_low_payload"] == 8
    assert report["low_group_scales_fp32"] == 16
    assert report["protected_high_bf16"] == 32
    assert report["online_pca_residual_frames"] == 8 * 8 * 2


def test_bf16_scales_are_frozen_before_gptq_and_accounted_as_bf16():
    torch.manual_seed(29)
    model = _TinyModel()
    original = model.proj.module.weight.detach().clone()
    states = {}
    summary = gptq_quantize_weights(
        model,
        {"proj": torch.eye(8)},
        bits=4,
        groupsize=4,
        sym=True,
        damp_pct=0.01,
        block_size=4,
        num_inv_tries=5,
        device="cpu",
        packed_state_out=states,
        packed_scale_dtype=torch.bfloat16,
    )
    state = states["proj"]
    expected_scale = (
        original.reshape(4, 2, 4).abs().amax(-1) / 7
    ).to(torch.bfloat16)
    assert summary["packed_scale_dtype"] == "torch.bfloat16"
    assert state["weight_scales"].dtype == torch.bfloat16
    assert torch.equal(state["weight_scales"], expected_scale)
    decoded = decode_weight_int4(
        state["qweight"], state["weight_scales"], 8, 4, torch.bfloat16
    )
    assert torch.equal(decoded, model.proj.module.weight.to(torch.bfloat16))
    _, report = exact_gptq_cache_from_states(
        states, expected_layers=1, gptq_summary=summary
    )
    assert report["aggregate"]["scale_dtype"] == "torch.bfloat16"
    assert report["aggregate"]["scale_bytes"] == 16

    transformer = nn.Module()
    transformer.proj = model.proj
    transformer.proj.module = PackedSplitInt4Linear(
        state["qweight"], state["weight_scales"],
        logical_low_k=8, group_size=4, high_weight=None, bias=None,
        require_cuda=False,
    )
    delattr(transformer.proj, "weight")
    transformer.proj.register_parameter("weight", None)
    delattr(transformer.proj, "bias")
    transformer.proj.register_parameter("bias", None)
    storage = real_int4_storage_report(transformer)
    assert storage["low_group_scales_fp32"] == 0
    assert storage["low_group_scales_bf16"] == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_packed_kernel_matches_groupwise_integer_reference():
    from utils.packed_int4_triton import packed_w4a4_gemm

    gen = torch.Generator().manual_seed(41)
    m, n, k, group = 17, 37, 128, 64
    a = torch.randint(0, 16, (m, k), generator=gen, dtype=torch.uint8)
    b = torch.randint(-8, 8, (n, k), generator=gen, dtype=torch.int8)
    a_scale = torch.rand(m, k // group, generator=gen) * 0.2 + 0.01
    a_zero = torch.randint(5, 11, (m, k // group), generator=gen).float()
    b_scale = torch.rand(n, k // group, generator=gen) * 0.1 + 0.01
    expected = torch.zeros(m, n)
    for g in range(k // group):
        sl = slice(g * group, (g + 1) * group)
        partial = (a[:, sl].to(torch.int32) - a_zero[:, g, None].to(torch.int32)) @ b[:, sl].to(torch.int32).T
        expected += partial.float() * a_scale[:, g, None] * b_scale[None, :, g]
    actual = packed_w4a4_gemm(
        pack_unsigned_int4(a).cuda(),
        pack_signed_int4(b).cuda(),
        a_scale.cuda(),
        a_zero.cuda(),
        b_scale.cuda(),
    ).cpu()
    assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-5)
