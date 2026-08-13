from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace
from pathlib import Path

from utils.asymmetric_tilemix_stats import (
    ASYMMETRIC_FIXED_E0_WEIGHT_ACTIVATIONS,
    TileMixTrajectoryStats,
    validate_fixed_e0_weight_activation_format,
)
from utils.tilemixfp4_utils import fake_quantize_tile_mix_oracle
from metrics.prepare_asymmetric_tilemix_pilot32 import generation_command
from metrics.evaluate_asymmetric_tilemix_pilot32 import (
    C0,
    C2,
    _trimmed_mean,
    classify_result,
)


def test_matched_activation_controls_share_explicit_fixed_e0_cache_kind():
    cache_kind = "hardware-fixed-e0"
    cache_by_activation = {}
    for activation_format in ("e0m3", "nvfp4-hw", "tile-mix-oracle"):
        validate_fixed_e0_weight_activation_format(activation_format)
        cache_by_activation[activation_format] = cache_kind
    assert set(cache_by_activation) == ASYMMETRIC_FIXED_E0_WEIGHT_ACTIVATIONS
    assert set(cache_by_activation.values()) == {"hardware-fixed-e0"}


def test_generation_commands_keep_activation_out_of_weight_cache_key():
    args = SimpleNamespace(
        dataset=Path("dataset.json"),
        hessian_sha256="1" * 64,
        weight_cache=Path("same_hardware_fixed_e0.pt"),
    )
    commands = {
        activation: generation_command(args, activation, Path(activation), None)
        for activation in ("e0m3", "nvfp4-hw", "tile-mix-oracle")
    }
    for activation, command in commands.items():
        assert command[command.index("--activation-format") + 1] == activation
        assert command[command.index("--hardware-weight-cache-kind") + 1] == "hardware-fixed-e0"
        assert command[command.index("--quantized-cache") + 1] == "same_hardware_fixed_e0.pt"
        assert command[command.index("--hardware-weight-hessian-sha256") + 1] == "1" * 64


def test_evaluator_trim_and_classification_rules():
    assert _trimmed_mean(torch.arange(10).numpy()) == 4.5
    comparison = f"{C2}_minus_{C0}"
    paired = {comparison: {
        "psnr": {"mean_delta": .20, "trimmed_mean_10pct": .18, "ci95": [.01, .4]},
        "lpips": {"mean_delta": -.004, "trimmed_mean_10pct": -.0035,
                  "ci95": [-.008, -.0002]},
        "clip_score": {"mean_delta": .01, "trimmed_mean_10pct": .01,
                       "ci95": [-.02, .04]},
    }}
    tail = {
        "psnr_delta_below_minus_0p5_ratio": .10,
        "lpips_delta_above_plus_0p01_ratio": .10,
    }
    classification, criteria = classify_result(paired, tail, e0_ratio=.80)
    assert classification == "ASYMMETRIC MIX PROMISING"
    assert all(criteria[key] for key in (
        "mean_psnr_lpips_favorable", "trimmed_psnr_lpips_favorable",
        "minimum_effect_reached", "format_distribution_nondegenerate",
    ))


@pytest.mark.parametrize(
    "activation_format",
    ("nvfp4", "block-mix-oracle", "tile-mix-output-oracle",
     "nvfp4-4over6", "e0m3-gscale1536"),
)
def test_unmatched_activation_experiments_are_rejected(activation_format):
    with pytest.raises(ValueError, match="hardware fixed-weight runtime activation"):
        validate_fixed_e0_weight_activation_format(activation_format)


def test_trajectory_stats_are_observational_and_prompt_aligned():
    torch.manual_seed(20260813)
    x = torch.randn(4, 16, 65, dtype=torch.bfloat16)
    # The last CFG branch contains the global amax.  If statistics selected or
    # requantized rows independently, output would differ from the plain call.
    x[-1, -1, -1] = 81.0
    expected = fake_quantize_tile_mix_oracle(x)

    stats = TileMixTrajectoryStats(require_cuda=False)
    layer = stats.for_layer("transformer_blocks.0.attn.to_q")
    stats.start_batch([("prompt-a", {}), ("prompt-b", {})])
    stats._current_timestep = 999
    actual = fake_quantize_tile_mix_oracle(x, format_stats=layer)
    layer.record_reconstruction(x, actual)
    stats.end_batch()

    assert torch.equal(actual, expected)
    snapshot = stats.snapshot()
    assert snapshot["total_count"] == 8  # four row tiles x two K tiles
    assert sum(item["total_count"] for item in snapshot["per_prompt"].values()) == 8
    assert snapshot["selected_sse"] <= snapshot["all_e0_sse"] + 1e-12
    assert snapshot["selected_sse"] <= snapshot["all_e2_sse"] + 1e-12
    assert snapshot["reconstruction_sse"] >= 0.0


def test_flip_rate_uses_same_layer_tile_across_true_timesteps():
    stats = TileMixTrajectoryStats(require_cuda=False)
    layer = stats.for_layer("layer")
    stats.start_batch([("a", {}), ("b", {})])
    choices0 = torch.tensor([[True], [False], [True], [False]])
    e0 = torch.tensor([[1.0], [3.0], [1.0], [3.0]])
    e2 = torch.tensor([[2.0], [2.0], [2.0], [2.0]])
    stats._current_timestep = 100
    layer.record_tilemix(choices0, e0, e2, original_shape=(4, 16, 64))
    choices1 = torch.tensor([[False], [False], [True], [True]])
    stats._current_timestep = 90
    layer.record_tilemix(choices1, e0, e2, original_shape=(4, 16, 64))
    stats.end_batch()
    snapshot = stats.snapshot()
    assert snapshot["flip_count"] == 2
    assert snapshot["flip_total"] == 4
    assert snapshot["flip_rate"] == 0.5
    assert snapshot["e0_to_e2"] == 1
    assert snapshot["e2_to_e0"] == 1


def test_cpu_fallback_is_rejected_in_real_run_mode():
    stats = TileMixTrajectoryStats()
    stats.start_batch([("a", {})])
    stats._current_timestep = 1
    with pytest.raises(RuntimeError, match="refuse CPU fallback"):
        stats.for_layer("layer").record_tilemix(
            torch.zeros((1, 1), dtype=torch.bool),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            original_shape=(1, 16, 64),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_stats_have_no_output_effect_or_device_fallback():
    torch.manual_seed(99)
    x = torch.randn(4, 32, 73, device="cuda", dtype=torch.bfloat16)
    expected = fake_quantize_tile_mix_oracle(x)
    stats = TileMixTrajectoryStats()
    stats.start_batch([("a", {}), ("b", {})])
    stats._current_timestep = 7
    actual = fake_quantize_tile_mix_oracle(x, format_stats=stats.for_layer("layer"))
    stats.end_batch()
    assert actual.device.type == "cuda"
    assert torch.equal(actual, expected)
    assert stats.snapshot()["total_count"] > 0


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_noncontiguous_padding_zero_and_dtype_regression(dtype):
    base = torch.randn(2, 67, 17, dtype=dtype)
    x = base.transpose(1, 2)
    assert not x.is_contiguous()
    out = fake_quantize_tile_mix_oracle(x)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()
    zero = fake_quantize_tile_mix_oracle(torch.zeros_like(x))
    assert torch.count_nonzero(zero) == 0
