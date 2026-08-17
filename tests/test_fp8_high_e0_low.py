import math

import pytest
import torch
import torch.nn as nn

from utils.fp8_high_e0_low import (
    HighFormatStats,
    decode_e4m3_bytes,
    decode_ue8m0,
    derive_rank_contract,
    generate_matched_residual_rotations,
    quantize_mxfp8_e4m3,
    quantize_plain_e4m3,
    scalar_e4m3_encode,
    scalar_mxfp8_reference,
    serialized_weight_bytes,
    validate_residual_rotation,
)
from utils.quant_utils import ActQuantizer
from utils.fp8_high_e0_low_experiment import (
    collect_teacher_cache, dev_gate, stitch_low_high,
)
from utils.quant_utils import ActQuantWrapper


def test_rank_contract_matches_sana_baseline_and_triple():
    baseline = derive_rank_contract(
        hidden_dim=2240, head_dim=32, num_heads=70,
        high_fraction=.125, multiplier=1,
    )
    triple = derive_rank_contract(
        hidden_dim=2240, head_dim=32, num_heads=70,
        high_fraction=.125, multiplier=3,
    )
    assert (baseline.high_hidden, baseline.low_hidden) == (288, 1952)
    assert (baseline.high_per_head, baseline.low_per_head) == (4, 28)
    assert (triple.high_hidden, triple.low_hidden) == (864, 1376)
    assert (triple.high_per_head, triple.low_per_head) == (12, 20)


@pytest.mark.parametrize("fmt", ("e4m3", "mxfp8"))
def test_high_format_stats_include_device_side_saturation_and_scale_distribution(fmt):
    source = torch.tensor([[0.0, 1.0, 2.0, 448.0] * 8], dtype=torch.float32)
    result = quantize_plain_e4m3(source) if fmt == "e4m3" else quantize_mxfp8_e4m3(source)
    stats = HighFormatStats()
    stats.observe(source, result.reconstructed, fmt)
    snapshot = stats.snapshot()
    assert snapshot["calls"] == 1
    assert snapshot["elements"] == source.numel()
    assert snapshot["scale_count"] == (1 if fmt == "e4m3" else 1)
    assert snapshot["scale_min"] > 0
    assert snapshot["scale_max"] >= snapshot["scale_min"]
    assert sum(snapshot["scale_log2_histogram"].values()) == snapshot["scale_count"]
    assert 0 <= snapshot["saturation_rate"] <= 1


def test_matched_residual_rotation_is_deterministic_and_tail_isolated():
    contract = derive_rank_contract(
        hidden_dim=32, head_dim=8, num_heads=4,
        high_fraction=.25, multiplier=1,
    )
    first = generate_matched_residual_rotations(contract, seed=42)
    second = generate_matched_residual_rotations(contract, seed=42)
    assert torch.equal(first["R1"], second["R1"])
    assert torch.equal(first["R2"], second["R2"])
    report = validate_residual_rotation(first, contract)
    assert report["R1_orthogonality_max_abs"] < 1e-10
    low = contract.low_per_head
    assert first["R2"][:low, low:].eq(0).all()


def test_e4m3_bytes_boundaries_zero_and_ties():
    values = torch.tensor([0.0, -0.0, 2.0 ** -9, 1.0, 448.0, -448.0])
    result = quantize_plain_e4m3(values)
    assert result.payload[0].item() == 0
    assert result.payload[1].item() == 0
    assert torch.isfinite(result.reconstructed).all()
    # With max=448 the global scale is exactly one.
    assert result.scale.item() == 1.0
    assert torch.equal(result.decoded_fp32, values)
    for value in [0.0, -0.0, 2.0 ** -9, 1.0, 1.0625, 447.9, 1000.0]:
        code = scalar_e4m3_encode(value)
        expected = torch.tensor(value).clamp(-448, 448).to(torch.float8_e4m3fn)
        assert code == expected.view(torch.uint8).item() or (value == -0.0 and code == 0)


def test_plain_e4m3_uses_full_call_scale_not_selected_rows():
    x = torch.tensor([[1.0, -2.0], [1000.0, 0.0]])
    whole = quantize_plain_e4m3(x)
    selected = quantize_plain_e4m3(x[:1])
    assert whole.scale.item() == pytest.approx(1000.0 / 448.0)
    assert selected.scale.item() == pytest.approx(2.0 / 448.0)
    assert not torch.equal(whole.payload[:1], selected.payload)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mxfp8_vectorized_matches_independent_scalar(dtype):
    source = torch.tensor(
        [[0.0, -0.0, 1.0, -2.0, 255.0, 448.0, 449.0] +
         [(-1.0) ** i * (i + 1) / 17 for i in range(30)]], dtype=dtype,
    )
    vector = quantize_mxfp8_e4m3(source)
    payload, scales, decoded = scalar_mxfp8_reference(source.float())
    assert vector.padded_k == 64
    assert torch.equal(vector.payload, payload)
    assert torch.equal(vector.scale_bytes, scales)
    torch.testing.assert_close(vector.decoded_fp32.float(), decoded, rtol=0, atol=0)
    assert vector.payload[0, source.shape[-1]:].eq(0).all()


def test_ue8m0_decode_extremes_and_nan():
    decoded = decode_ue8m0(torch.tensor([0, 127, 254, 255], dtype=torch.uint8))
    assert decoded[0].item() == pytest.approx(2.0 ** -127)
    assert decoded[1].item() == 1.0
    assert math.isfinite(decoded[2].item())
    assert torch.isnan(decoded[3])


def test_mxfp8_noncontiguous_zero_and_extreme():
    base = torch.arange(66, dtype=torch.float32).reshape(3, 22)
    x = base.t()  # non-contiguous [22,3]
    result = quantize_mxfp8_e4m3(x)
    assert result.reconstructed.shape == x.shape
    assert result.reconstructed.dtype == x.dtype
    assert torch.isfinite(result.reconstructed).all()
    zero = quantize_mxfp8_e4m3(torch.zeros(2, 33, dtype=torch.bfloat16))
    assert zero.payload.eq(0).all()
    assert zero.scale_bytes.eq(127).all()
    assert zero.reconstructed.eq(0).all()


def test_actquantizer_high_path_and_low_regression():
    x = torch.tensor([[[0.0] * 16 + [1.0, 1000.0]]], dtype=torch.bfloat16)
    baseline = ActQuantizer()
    baseline.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=2,
        quant_dtype="e0m3", high_quant_format="bf16",
    )
    candidate = ActQuantizer()
    candidate.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=2,
        quant_dtype="e0m3", high_quant_format="e4m3",
    )
    out_baseline = baseline(x)
    out_candidate = candidate(x)
    # High-format routing cannot change the hardware-E0 low operand.
    assert torch.equal(out_baseline[..., :16], out_candidate[..., :16])
    assert not torch.equal(out_baseline[..., 16:], out_candidate[..., 16:])
    assert out_candidate.dtype == x.dtype
    with pytest.raises(ValueError):
        candidate.configure(bits=4, quant_dtype="e0m3", high_quant_format="ue4m3")


def test_serialized_bytes_exposes_non_equal_budget():
    b0 = serialized_weight_bytes(
        out_features=2240, low_rank=1952, high_rank=288, high_format="bf16",
    )
    e4 = serialized_weight_bytes(
        out_features=2240, low_rank=1376, high_rank=864, high_format="e4m3",
    )
    mx = serialized_weight_bytes(
        out_features=2240, low_rank=1376, high_rank=864, high_format="mxfp8",
    )
    assert e4["total"] < b0["total"]
    assert mx["total"] > e4["total"]
    assert mx["high_scales"] == 2240 * (864 // 32)


def test_stitch_low_high_hidden_and_per_head_layout():
    hidden = ActQuantWrapper(nn.Linear(8, 3, bias=False))
    hidden.rotation = torch.eye(8)
    hidden.quantizer.high_bits_length = 2
    low = torch.arange(18).reshape(3, 6)
    high = torch.arange(6).reshape(3, 2) + 100
    assert torch.equal(stitch_low_high(hidden, low, high), torch.cat((low, high), 1))

    per_head = ActQuantWrapper(nn.Linear(8, 3, bias=False))
    per_head.rotation_per_head = torch.eye(4).repeat(2, 1, 1)
    per_head.num_heads = 2
    per_head.head_dim = 4
    per_head.quantizer.high_bits_length = 1
    low = torch.tensor([[0, 1, 2, 10, 11, 12]])
    high = torch.tensor([[9, 19]])
    expected = torch.tensor([[0, 1, 2, 9, 10, 11, 12, 19]])
    assert torch.equal(stitch_low_high(per_head, low, high), expected)


def test_dev_gate_uses_raw_mass_prompt_and_timestep_groups():
    baseline = {
        "raw_sse": 100.0,
        "per_prompt": {str(i): 1.0 for i in range(32)},
        "per_timestep": {str(i): 5.0 for i in range(20)},
    }
    good = {
        "raw_sse": 99.5,
        "per_prompt": {str(i): .9 if i < 16 else 1.1 for i in range(32)},
        "per_timestep": {str(i): 5.0 for i in range(20)},
    }
    bad = {
        "raw_sse": 103.0,
        "per_prompt": {str(i): 1.1 for i in range(32)},
        "per_timestep": {str(i): 5.2 for i in range(20)},
    }
    report = dev_gate({"B0": baseline, "E4-AW": good, "MX-AW": bad})
    assert report["arms"]["E4-AW"]["passed"]
    assert not report["arms"]["MX-AW"]["passed"]
    assert report["continue"]


def test_teacher_cache_failure_is_visibly_incomplete(tmp_path):
    class FailedPipeline:
        def __init__(self):
            self.transformer = nn.Linear(2, 2)
            self.device = torch.device("cpu")

        def set_progress_bar_config(self, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("intentional collection failure")

    rows = [{
        "image_id": "a" * 40, "prompt": "test", "seed": 1,
        "exact_prompt_sha256": "b" * 64,
    }]
    output = tmp_path / "fit"
    with pytest.raises(RuntimeError, match="intentional"):
        collect_teacher_cache(
            FailedPipeline(), rows, output, selected_steps=(0,),
            num_steps=1, guidance_scale=1.0,
        )
    assert not output.exists()
    assert output.with_name("fit.incomplete").is_dir()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ada_plain_e4m3_scaled_mm_parity():
    torch.manual_seed(7)
    a = torch.randn(16, 64, device="cuda")
    # Preserve column-major B after casting, as required by cuBLASLt.
    b_storage = torch.randn(16, 64, device="cuda")
    qa = quantize_plain_e4m3(a)
    qb = quantize_plain_e4m3(b_storage)
    a_payload = qa.payload.view(torch.float8_e4m3fn)
    b_payload = qb.payload.view(torch.float8_e4m3fn).t()
    native = torch._scaled_mm(
        a_payload, b_payload,
        scale_a=qa.scale, scale_b=qb.scale, out_dtype=torch.float32,
    )
    reference = qa.decoded_fp32 @ qb.decoded_fp32.t()
    reference = reference * qa.scale * qb.scale
    torch.testing.assert_close(native, reference, rtol=2e-4, atol=2e-3)
