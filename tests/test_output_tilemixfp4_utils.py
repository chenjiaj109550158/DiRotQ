import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.output_tilemixfp4_utils import (
    OutputOracleFormatStats,
    build_output_weight_grams,
    fake_quantize_tile_mix_output_oracle,
    weighted_output_tile_scores_direct,
    weighted_output_tile_scores_gram,
)
from utils.quant_utils import ActQuantWrapper, ActQuantizer
from utils.tilemixfp4_utils import (
    fake_quantize_e0m3,
    fake_quantize_e2m1,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_pixart_model_utils():
    path = ROOT / "models" / "pixart-sigma" / "model_utils.py"
    spec = importlib.util.spec_from_file_location("pixart_output_oracle_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_tiles(x):
    m, k = x.shape
    padded = F.pad(x.float(), (0, (-k) % 64, 0, (-m) % 16))
    return padded.reshape(padded.shape[0] // 16, 16, -1, 64).permute(0, 2, 1, 3)


def test_activation_sse_can_choose_wrong_weighted_output_format():
    # Seed 0 has lower whole-tile activation SSE for E0M3, but column zero is
    # represented better by E2M1.  A real low weight supported only on that
    # column makes the local output oracle correctly select E2M1.
    torch.manual_seed(0)
    x = torch.randn(16, 64) * 3.0
    e2, e0 = fake_quantize_e2m1(x), fake_quantize_e0m3(x)
    assert (x - e0).square().sum() < (x - e2).square().sum()

    weight = torch.zeros(3, 64)
    weight[:, 0] = torch.tensor([1.0, -0.5, 2.0])
    assert ((x - e2) @ weight.T).square().sum() < ((x - e0) @ weight.T).square().sum()

    stats = OutputOracleFormatStats(selection_unit="tile")
    out = fake_quantize_tile_mix_output_oracle(
        x, build_output_weight_grams(weight), format_stats=stats
    )
    assert torch.equal(out, e2)
    snapshot = stats.snapshot()
    assert snapshot["e2m1_count"] == 1
    assert snapshot["e0m3_count"] == 0
    assert snapshot["weighted_output_error_selected"] == pytest.approx(
        snapshot["weighted_output_error_all_e2"]
    )


def test_gram_score_matches_direct_matmul_score():
    torch.manual_seed(301)
    delta = torch.randn(3, 2, 16, 64)
    weight = torch.randn(19, 101)
    grams = build_output_weight_grams(weight)
    direct = weighted_output_tile_scores_direct(delta, weight)
    gram = weighted_output_tile_scores_gram(delta, grams)
    torch.testing.assert_close(gram, direct, rtol=2e-5, atol=2e-4)


def test_output_oracle_dominates_both_fixed_candidates_per_tile():
    torch.manual_seed(302)
    x = torch.randn(19, 77) * 2.0
    weight = torch.randn(23, 77)
    grams = build_output_weight_grams(weight)
    oracle = fake_quantize_tile_mix_output_oracle(x, grams)
    e2, e0 = fake_quantize_e2m1(x), fake_quantize_e0m3(x)
    oracle_error = weighted_output_tile_scores_direct(_as_tiles(x - oracle), weight)
    e2_error = weighted_output_tile_scores_direct(_as_tiles(x - e2), weight)
    e0_error = weighted_output_tile_scores_direct(_as_tiles(x - e0), weight)
    torch.testing.assert_close(
        oracle_error, torch.minimum(e2_error, e0_error), rtol=2e-5, atol=3e-4
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_padding_incomplete_shape_dtype_device_and_finite(dtype):
    x = torch.zeros(2, 3, 70, dtype=dtype).transpose(0, 1)
    assert not x.is_contiguous()
    weight = torch.randn(7, 70, dtype=dtype)
    out = fake_quantize_tile_mix_output_oracle(x, build_output_weight_grams(weight))
    assert torch.equal(out, x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


def test_explicit_zero_padding_does_not_change_choice_or_valid_output():
    torch.manual_seed(303)
    x = torch.randn(15, 63)
    weight = torch.randn(11, 63)
    stats = OutputOracleFormatStats(selection_unit="tile")
    out = fake_quantize_tile_mix_output_oracle(
        x, build_output_weight_grams(weight), format_stats=stats
    )

    padded_x = F.pad(x, (0, 1, 0, 1))
    padded_weight = F.pad(weight, (0, 1))
    padded_stats = OutputOracleFormatStats(selection_unit="tile")
    padded_out = fake_quantize_tile_mix_output_oracle(
        padded_x, build_output_weight_grams(padded_weight), format_stats=padded_stats
    )
    assert torch.equal(out, padded_out[:15, :63])
    assert stats.snapshot()["e0m3_count"] == padded_stats.snapshot()["e0m3_count"]


def _make_output_wrapper(in_features=68, out_features=9, high_tail=4):
    wrapper = ActQuantWrapper(nn.Linear(in_features, out_features, bias=False))
    wrapper.quantizer.configure(
        bits=4,
        groupsize=16,
        sym=True,
        high_bits_length=high_tail,
        quant_dtype="tile-mix-output-oracle",
    )
    wrapper.output_oracle_weight_ready = True
    return wrapper


def test_high_precision_tail_does_not_participate_in_selector():
    torch.manual_seed(304)
    wrapper = _make_output_wrapper()
    low = torch.randn(2, 5, 64)
    tail_a = torch.randn(2, 5, 4)
    tail_b = torch.full_like(tail_a, 1.0e20)
    out_a = wrapper.quantize_activation_for_linear(torch.cat((low, tail_a), dim=-1))
    out_b = wrapper.quantize_activation_for_linear(torch.cat((low, tail_b), dim=-1))
    assert torch.equal(out_a[..., :64], out_b[..., :64])
    assert torch.equal(out_a[..., 64:], tail_a)
    assert torch.equal(out_b[..., 64:], tail_b)


def test_wrapper_requires_loaded_executed_weight_and_generic_quantizer_rejects_mode():
    x = torch.randn(2, 68)
    wrapper = _make_output_wrapper()
    wrapper.output_oracle_weight_ready = False
    with pytest.raises(RuntimeError, match="loaded NVFP4 E2M1 GPTQ cache"):
        wrapper.quantize_activation_for_linear(x)

    quantizer = ActQuantizer()
    quantizer.configure(
        bits=4, groupsize=16, sym=True, quant_dtype="tile-mix-output-oracle"
    )
    with pytest.raises(RuntimeError, match="ActQuantWrapper"):
        quantizer(x[..., :64])


@pytest.mark.parametrize(
    "activation_format",
    ("nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle"),
)
def test_existing_wrapper_activation_routes_are_numerically_unchanged(activation_format):
    torch.manual_seed(305)
    wrapper = ActQuantWrapper(nn.Linear(70, 13, bias=False))
    wrapper.quantizer.configure(
        bits=4, groupsize=16, sym=True, quant_dtype=activation_format
    )
    x = torch.randn(17, 70)
    expected = wrapper.quantizer(x)
    actual = wrapper.quantize_activation_for_linear(x)
    assert torch.equal(actual, expected)


class _OneLayerTransformer:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def named_modules(self):
        yield "transformer_blocks.0.attn1.to_q", self.wrapper


def test_pixart_routes_output_oracle_to_wrapper():
    model_utils = _load_pixart_model_utils()
    wrapper = ActQuantWrapper(nn.Linear(32, 32, bias=False))
    transformer = _OneLayerTransformer(wrapper)
    cfg = {
        "quantization": {"a_bits": 4},
        "dims": {"head": 32, "hidden": 32, "intermediate": 64},
        "nvfp4": {"a_groupsize": 16, "a_groupsize_attn_out": 32},
    }
    stats = OutputOracleFormatStats(selection_unit="tile")
    model_utils.configure_quantizers_by_name(
        transformer,
        high_len_hidden=16,
        high_len_head=4,
        cfg=cfg,
        nvfp4=True,
        activation_format="tile-mix-output-oracle",
        format_stats=stats,
    )
    assert wrapper.quantizer.quant_dtype == "tile-mix-output-oracle"
    assert wrapper.quantizer.format_stats is stats
    assert wrapper.quantizer.high_bits_length == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_execution_stays_on_device_without_cpu_fallback(dtype):
    wrapper = _make_output_wrapper().to("cuda", dtype=dtype)
    x = torch.randn(17, 2, 68, device="cuda", dtype=dtype).transpose(0, 1)
    assert not x.is_contiguous()
    out = wrapper.quantize_activation_for_linear(x)
    assert out.device.type == "cuda"
    assert out.dtype == dtype
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert wrapper._output_oracle_gram_cache[1].device.type == "cuda"
