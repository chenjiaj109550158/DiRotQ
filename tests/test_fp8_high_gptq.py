import copy

import pytest
import torch
import torch.nn as nn

from utils.fp8_high_e0_low import (
    quantize_mxfp8_e4m3,
    quantize_mxfp8_e4m3_activation,
    quantize_plain_e4m3_per_token,
)
from utils.fp8_high_gptq import (
    E4_SCALE_MULTIPLIERS,
    choose_e4_per_channel_scales,
    choose_mx_scale_bytes,
    decode_high_weight_record,
    hessian_weighted_error,
    make_high_weight_record,
    quantize_e4_per_channel_gptq,
    quantize_e4_per_channel_rtn,
    quantize_mx_weight_gptq,
    quantize_mx_weight_rtn,
    serialized_high_record_bytes,
)
from utils.fp8_high_gptq_experiment import (
    build_high_weight_sidecars,
    load_high_hessian_cache,
    materialize_high_sidecar_into_state,
    project_high_activation,
    summarize_dev_weight_gate,
    write_high_hessian_cache,
)
from utils.quant_utils import ActQuantWrapper, ActQuantizer


def _positive_hessian(k: int) -> torch.Tensor:
    torch.manual_seed(77)
    x = torch.randn(96, k)
    return 2.0 * x.T @ x / x.shape[0]


def test_e4_per_channel_scale_contract_and_legal_payload():
    source = torch.tensor([
        [0.0, 1.0, -448.0, 224.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    result, _ = quantize_e4_per_channel_rtn(source)
    assert result.scales.dtype == torch.float32
    torch.testing.assert_close(result.scales, torch.tensor([1.0, 1.0]))
    assert result.payload.dtype == torch.uint8
    assert result.payload[1].eq(0).all()
    assert torch.isfinite(result.reconstructed).all()


def test_e4_activation_uses_full_per_token_vector_not_chunks():
    source = torch.tensor([[1.0, 2.0, 1000.0], [4.0, 0.0, -8.0]])
    result = quantize_plain_e4m3_per_token(source)
    assert result.scale.shape == (2, 1)
    assert result.scale[0].item() == pytest.approx(1000.0 / 448.0)
    assert result.scale[1].item() == pytest.approx(8.0 / 448.0)
    selected_only = quantize_plain_e4m3_per_token(source[:, :2])
    assert not torch.equal(result.payload[:, :2], selected_only.payload)


def test_mx_activation_nosat_and_neighbor_have_k32_scale_mapping():
    source = torch.tensor([[448.1] + [1.0] * 32], dtype=torch.bfloat16)
    nosat = quantize_mxfp8_e4m3_activation(source, "nosat")
    neighbor = quantize_mxfp8_e4m3_activation(source, "neighbor")
    assert nosat.scale_bytes.shape == (1, 2)
    assert neighbor.scale_bytes.shape == (1, 2)
    assert nosat.saturation_count == 0
    assert nosat.payload[:, source.shape[1]:].eq(0).all()
    assert neighbor.payload[:, source.shape[1]:].eq(0).all()


@pytest.mark.parametrize(
    "high_format", ("e4m3-token", "mxfp8-nosat", "mxfp8-neighbor")
)
def test_conditional_a8_routing_cannot_change_low_e0_operand(high_format):
    source = torch.tensor(
        [[[float(index - 8) for index in range(16)] + [1.0, 1000.0]]],
        dtype=torch.bfloat16,
    )
    baseline = ActQuantizer()
    baseline.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=2,
        quant_dtype="e0m3", high_quant_format="bf16",
    )
    candidate = ActQuantizer()
    candidate.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=2,
        quant_dtype="e0m3", high_quant_format=high_format,
    )
    expected = baseline(source)
    observed = candidate(source)
    assert torch.equal(observed[..., :16], expected[..., :16])
    assert observed.dtype == source.dtype
    assert torch.isfinite(observed).all()


def test_e4_fit_scale_selection_is_rowwise_and_deterministic():
    torch.manual_seed(4)
    source = torch.randn(7, 19)
    hessian = _positive_hessian(19)
    first = choose_e4_per_channel_scales(
        source, hessian, E4_SCALE_MULTIPLIERS
    )
    second = choose_e4_per_channel_scales(
        source, hessian, E4_SCALE_MULTIPLIERS
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2] == second[2]
    assert sum(first[2]["selection_counts"].values()) == source.shape[0]


def test_high_hessian_objective_matches_direct_fit_outputs():
    torch.manual_seed(31)
    activations = torch.randn(23, 9)
    source = torch.randn(5, 9)
    quantized = source + .1 * torch.randn_like(source)
    hessian = 2.0 * activations.T @ activations / activations.shape[0]
    direct = 2.0 / activations.shape[0] * (
        activations @ (source - quantized).T
    ).double().square().sum()
    traced = hessian_weighted_error(source, quantized, hessian)
    torch.testing.assert_close(traced, direct, rtol=1e-5, atol=1e-6)


def test_e4_gptq_changes_payload_and_improves_matched_objective():
    torch.manual_seed(1)
    source = torch.randn(5, 12) * torch.linspace(.1, 10, 12)
    hessian = _positive_hessian(12)
    scales, _, _ = choose_e4_per_channel_scales(source, hessian)
    rtn, _ = quantize_e4_per_channel_rtn(source, scales=scales)
    gptq = quantize_e4_per_channel_gptq(
        source, hessian, scales, require_cuda=False
    )
    assert gptq.gptq_status == "gptq"
    assert gptq.payload_mismatch_vs_rtn > 0
    assert hessian_weighted_error(source, gptq.reconstructed, hessian) <= (
        hessian_weighted_error(source, rtn.reconstructed, hessian) + 1e-6
    )
    assert not gptq.reconstructed.requires_grad


def test_mx_current_rtn_bitwise_reproduces_existing_reference():
    torch.manual_seed(8)
    source = torch.randn(4, 67, dtype=torch.bfloat16)
    old = quantize_mxfp8_e4m3(source)
    new, metadata = quantize_mx_weight_rtn(source, recipe="current")
    assert metadata["recipe"] == "current"
    assert torch.equal(new.payload, old.payload)
    assert torch.equal(new.scale_bytes, old.scale_bytes)
    torch.testing.assert_close(new.reconstructed, old.reconstructed, rtol=0, atol=0)


def test_mx_nosat_and_neighbor_are_legal_and_padding_is_positive_zero():
    source = torch.tensor([[448.1] + [(-1.0) ** index * index for index in range(34)]])
    for recipe in ("nosat", "neighbor"):
        result, metadata = quantize_mx_weight_rtn(source, recipe=recipe)
        assert result.scale_bytes.dtype == torch.uint8
        assert result.scale_bytes.ne(255).all()
        assert result.payload[:, source.shape[1]:].eq(0).all()
        assert torch.isfinite(result.reconstructed).all()
        assert metadata["recipe"] == recipe
    nosat_scales, _ = choose_mx_scale_bytes(source, "nosat")
    nosat, _ = quantize_mx_weight_rtn(source, scale_bytes=nosat_scales)
    assert nosat.saturation_count == 0


def test_mx_neighbor_gptq_improves_matched_neighbor_rtn():
    torch.manual_seed(13)
    source = torch.randn(6, 37) * torch.linspace(.2, 5.0, 37)
    hessian = _positive_hessian(37)
    scales, _ = choose_mx_scale_bytes(source, "neighbor")
    rtn, _ = quantize_mx_weight_rtn(source, scale_bytes=scales)
    gptq = quantize_mx_weight_gptq(
        source, hessian, scales, require_cuda=False
    )
    assert gptq.gptq_status == "gptq"
    assert gptq.payload_mismatch_vs_rtn > 0
    assert hessian_weighted_error(source, gptq.reconstructed, hessian) <= (
        hessian_weighted_error(source, rtn.reconstructed, hessian) + 1e-5
    )


@pytest.mark.parametrize("fmt", ("e4", "mx"))
def test_high_record_round_trip_keeps_real_payload_and_scales(fmt):
    torch.manual_seed(3)
    source = torch.randn(3, 35)
    if fmt == "e4":
        result, _ = quantize_e4_per_channel_rtn(source)
        format_name = "e4m3-per-channel"
    else:
        result, _ = quantize_mx_weight_rtn(source, recipe="neighbor")
        format_name = "mxfp8-e4m3-k32"
    record = make_high_weight_record(
        result, fmt=format_name, recipe="test", metadata={"fit": "only"}
    )
    decoded = decode_high_weight_record(record)
    torch.testing.assert_close(decoded, result.reconstructed.float(), rtol=0, atol=0)
    sizes = serialized_high_record_bytes(record)
    assert sizes["total"] == sizes["payload"] + sizes["scales"]
    broken = copy.deepcopy(record)
    broken["payload"][0, 0] ^= 1
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        decode_high_weight_record(broken)


def test_project_high_hidden_uses_tail_and_full_input_rows():
    wrapper = ActQuantWrapper(nn.Linear(6, 2, bias=False).to(torch.bfloat16))
    wrapper.rotation = torch.eye(6, dtype=torch.bfloat16)
    wrapper.quantizer.high_bits_length = 2
    x = torch.arange(24, dtype=torch.bfloat16).reshape(2, 2, 6)
    high = project_high_activation(wrapper, x)
    assert high.shape == (2, 2, 2)
    assert torch.equal(high, x[..., -2:])


def test_project_high_per_head_never_crosses_heads():
    wrapper = ActQuantWrapper(nn.Linear(8, 2, bias=False).to(torch.bfloat16))
    wrapper.rotation_per_head = torch.eye(4, dtype=torch.bfloat16).repeat(2, 1, 1)
    wrapper.num_heads = 2
    wrapper.head_dim = 4
    wrapper.quantizer.high_bits_length = 1
    x = torch.tensor([[[1, 2, 3, 4, 10, 20, 30, 40]]], dtype=torch.bfloat16)
    high = project_high_activation(wrapper, x)
    assert torch.equal(high, torch.tensor([[[4, 40]]], dtype=torch.bfloat16))


def test_materialized_sidecar_changes_only_high_region():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                ActQuantWrapper(nn.Linear(6, 3, bias=False).to(torch.bfloat16))
                for _ in range(120)
            ])

    model = Tiny()
    base_weight = torch.arange(18, dtype=torch.bfloat16).reshape(3, 6)
    result, _ = quantize_e4_per_channel_rtn(torch.full((3, 2), 2.0))
    records = {}
    for index, layer in enumerate(model.layers):
        layer.rotation = torch.eye(6)
        layer.quantizer.bits = 4
        layer.quantizer.high_bits_length = 2
        layer.module.weight.data.copy_(base_weight)
        records[f"layers.{index}"] = make_high_weight_record(
            result, fmt="e4m3-per-channel", recipe="test", metadata={}
        )
    base_state = {key: value.clone() for key, value in model.state_dict().items()}
    materialized = materialize_high_sidecar_into_state(
        model, base_state, {"layers": records}
    )
    decoded = decode_high_weight_record(records["layers.0"], dtype=torch.bfloat16)
    for index in range(120):
        key = f"layers.{index}.module.weight"
        assert torch.equal(materialized[key][:, :4], base_state[key][:, :4])
        assert torch.equal(materialized[key][:, 4:], decoded)
        # The immutable input mapping itself must not be mutated.
        assert torch.equal(base_state[key], base_weight)


def test_formal_gptq_refuses_silent_cpu_fallback():
    source = torch.randn(2, 8)
    hessian = _positive_hessian(8)
    scales, _, _ = choose_e4_per_channel_scales(source, hessian)
    with pytest.raises(RuntimeError, match="forbids silent CPU fallback"):
        quantize_e4_per_channel_gptq(source, hessian, scales)
    mx_scales, _ = choose_mx_scale_bytes(source, "neighbor")
    with pytest.raises(RuntimeError, match="forbids silent CPU fallback"):
        quantize_mx_weight_gptq(source, hessian, mx_scales)


def test_tiny_120_layer_build_is_complete_atomic_and_fallback_free(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                ActQuantWrapper(nn.Linear(6, 3, bias=False).to(torch.bfloat16))
                for _ in range(120)
            ])

    torch.manual_seed(21)
    model = Tiny()
    hessians = {}
    for index, layer in enumerate(model.layers):
        layer.rotation = torch.eye(6)
        layer.quantizer.bits = 4
        layer.quantizer.high_bits_length = 2
        layer.module.weight.data.normal_()
        hessians[f"layers.{index}"] = _positive_hessian(2)
    output = tmp_path / "complete"
    report = build_high_weight_sidecars(
        model, hessians, output, common_metadata={"fit": "frozen"},
        require_cuda=False,
    )
    assert output.is_dir()
    assert not output.with_name("complete.incomplete").exists()
    assert report["layer_count"] == report["gptq_coverage"] == 120
    assert report["rtn_fallbacks"] == report["cpu_fallbacks"] == 0
    for record in report["sidecars"].values():
        assert torch.load(record["path"], weights_only=False)["metadata"]["fit"] == "frozen"


def test_high_hessian_cache_is_provenance_isolated_and_no_overwrite(tmp_path):
    path = tmp_path / "high.pt"
    metadata = {"fit_manifest_sha256": "a" * 64, "basis_sha256": "b" * 64}
    write_high_hessian_cache(path, {"layer": torch.eye(3)}, metadata)
    loaded = load_high_hessian_cache(path, metadata)
    assert torch.equal(loaded["layer"], torch.eye(3))
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_high_hessian_cache(path, {**metadata, "basis_sha256": "c" * 64})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_high_hessian_cache(path, {"layer": torch.eye(3)}, metadata)


def test_weight_gate_uses_recovery_wins_groups_and_bytes():
    prompts = {str(index): 100.0 for index in range(32)}
    steps = {str(index): 160.0 for index in range(20)}
    summaries = {
        "B0": {"raw_sse": 3200.0, "per_prompt": prompts, "per_timestep": steps},
        "B1": {"raw_sse": 2400.0,
               "per_prompt": {key: 75.0 for key in prompts},
               "per_timestep": {key: 120.0 for key in steps}},
        "candidate": {"raw_sse": 2700.0,
                      "per_prompt": {key: 80.0 if int(key) < 24 else 110.0 for key in prompts},
                      "per_timestep": {key: 135.0 for key in steps}},
    }
    report = summarize_dev_weight_gate(
        summaries, b0_persistent_bytes=1000,
        arm_persistent_bytes={"candidate": 1005},
    )
    assert report["arms"]["candidate"]["rank_headroom_recovery"] == pytest.approx(.625)
    assert report["arms"]["candidate"]["prompt_wins"] == 24
    assert report["arms"]["candidate"]["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_gptq_is_deterministic_and_device_resident():
    torch.manual_seed(5)
    source = torch.randn(8, 24, device="cuda")
    x = torch.randn(64, 24, device="cuda")
    hessian = 2 * x.T @ x / x.shape[0]
    scales, _, _ = choose_e4_per_channel_scales(source, hessian)
    first = quantize_e4_per_channel_gptq(source, hessian, scales)
    second = quantize_e4_per_channel_gptq(source, hessian, scales)
    assert first.reconstructed.device.type == "cuda"
    assert torch.equal(first.payload, second.payload)
    assert torch.equal(first.reconstructed, second.reconstructed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ada_per_token_per_channel_e4_native_raw_mma_epilogue_parity():
    torch.manual_seed(17)
    activation = torch.randn(16, 64, device="cuda")
    # Public Ada cuBLASLt path requires N to be 16-aligned.
    weight = torch.randn(16, 64, device="cuda")
    qa = quantize_plain_e4m3_per_token(activation)
    qw, _ = quantize_e4_per_channel_rtn(weight)
    raw = torch._scaled_mm(
        qa.payload.view(torch.float8_e4m3fn),
        qw.payload.view(torch.float8_e4m3fn).t(),
        scale_a=torch.ones((), device="cuda"),
        scale_b=torch.ones((), device="cuda"),
        out_dtype=torch.float32,
    )
    native_with_epilogue = raw * qa.scale * qw.scales[None, :]
    reference = qa.decoded_fp32 @ (
        qw.payload.view(torch.float8_e4m3fn).float().t()
    )
    reference = reference * qa.scale * qw.scales[None, :]
    torch.testing.assert_close(native_with_epilogue, reference, rtol=2e-4, atol=2e-3)
