from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fouroversix_utils import (
    FOUR_OVER_SIX_FORMATS,
    FOUR_OVER_SIX_GLOBAL_MAX,
    M4_MAX_BLOCK_SCALE,
    M6_MAX_BLOCK_SCALE,
    FourOverSixStats,
    fake_quantize_e0m3_gscale1536,
    fake_quantize_four_over_six_activation,
    fake_quantize_nvfp4_4over6,
    fake_quantize_tile_mix_e0_e2_4over6,
    four_over_six_block_candidates,
)
from utils.quant_utils import ActQuantizer
from utils.tilemixfp4_utils import (
    E2M1_MAGNITUDES,
    fake_quantize_activation,
)


def _sse(x, y):
    return (x.float() - y.float()).square().sum()


def _tile_sse(x, y):
    m, k = x.shape
    pad_m, pad_k = (-m) % 16, (-k) % 64
    source = F.pad(x.float(), (0, pad_k, 0, pad_m))
    recon = F.pad(y.float(), (0, pad_k, 0, pad_m))
    valid = torch.zeros_like(source, dtype=torch.bool)
    valid[:m, :k] = True
    return (((source - recon).square() * valid).reshape(
        source.shape[0] // 16, 16, source.shape[1] // 64, 64
    ).sum(dim=(1, 3)))


def _official_reference_port(x):
    """Minimal port of official reference.py at dadfad6901d473a7.

    The installed torch is 2.6 and cannot import the upstream package (which
    requires torch>=2.8 and torch.float8_e8m0fnu), so this test ports only its
    pure-PyTorch NVFP4 MSE path: E4M3 max 256, M4 scale expansion 1.5, and
    strict error4 < error6 selection.
    """
    flat = x.float().reshape(-1, x.shape[-1])
    assert flat.shape[-1] % 16 == 0
    blocks = flat.reshape(-1, 16)
    amax = flat.abs().amax()
    if amax == 0:
        return torch.zeros_like(x)
    encode = 6.0 * 256.0 / amax
    decode = amax / (6.0 * 256.0)
    scales6 = (blocks.abs().amax(-1, keepdim=True) / 6.0 * encode)
    scales6 = scales6.to(torch.float8_e4m3fn).float()
    scales4 = (blocks.abs().amax(-1, keepdim=True) / 6.0 * encode * 1.5)
    scales4 = scales4.to(torch.float8_e4m3fn).float()

    def e2(code_input):
        cb = torch.tensor(E2M1_MAGNITUDES, device=x.device)
        midpoints = (cb[:-1] + cb[1:]) / 2
        index = torch.bucketize(code_input.abs().contiguous(), midpoints)
        return code_input.sign() * cb[index]

    q6 = e2(blocks / (decode * scales6)) * scales6 * decode
    q4 = e2(blocks / (decode * scales4)) * scales4 * decode
    choose4 = (blocks - q4).square().sum(-1) < (blocks - q6).square().sum(-1)
    return torch.where(choose4[:, None], q4, q6).reshape_as(x).to(x.dtype)


@pytest.mark.parametrize(
    ("values", "expected_m4"),
    [([10.0, 20.0, 30.0, 40.0], True),
     ([15.0, 30.0, 120.0, 180.0], False)],
)
def test_paper_table2_selection(values, expected_m4):
    x = torch.tensor(values)
    candidates = four_over_six_block_candidates(x)
    assert bool(candidates.choose_m4.item()) is expected_m4


def test_global_maximum_has_exact_legal_m4_m6_scales():
    x = torch.zeros(16)
    x[0] = FOUR_OVER_SIX_GLOBAL_MAX
    candidates = four_over_six_block_candidates(x)
    assert candidates.global_scale == 1
    assert candidates.scale_m6.item() == M6_MAX_BLOCK_SCALE
    assert candidates.scale_m4.item() == M4_MAX_BLOCK_SCALE
    assert torch.isfinite(candidates.m4).all()
    assert torch.isfinite(candidates.m6).all()


def test_m4_maps_maximum_to_four_without_a_over_nine_clipping():
    x = torch.tensor([1.0, 2.0, 5.0, 9.0, 17.0, 40.0] + [0.0] * 10)
    candidates = four_over_six_block_candidates(x)
    # The maximum is represented through code 4 at scale a/4.  The discarded
    # a/9 experiment instead reconstructed it near 2a/3.
    assert candidates.m4.abs().max() == pytest.approx(40.0, abs=1e-5)
    assert candidates.m4.abs().max() > 0.95 * x.abs().max()
    assert candidates.scale_m4.item() / candidates.scale_m6.item() == 1.5


def test_per_block_selection_dominates_both_candidates_and_ties_use_m6():
    torch.manual_seed(601)
    x = torch.randn(4, 67) * 80
    candidates = four_over_six_block_candidates(x)
    pad = (-x.shape[-1]) % 16
    source = F.pad(x, (0, pad)).reshape(4, -1, 16)
    m4 = F.pad(candidates.m4, (0, pad)).reshape_as(source)
    m6 = F.pad(candidates.m6, (0, pad)).reshape_as(source)
    selected = F.pad(candidates.selected, (0, pad)).reshape_as(source)
    valid = torch.zeros_like(source, dtype=torch.bool)
    valid.reshape(4, -1)[:, :x.shape[-1]] = True
    selected_error = ((source - selected).square() * valid).sum(-1)
    assert torch.all(selected_error <= ((source - m4).square() * valid).sum(-1) + 1e-6)
    assert torch.all(selected_error <= ((source - m6).square() * valid).sum(-1) + 1e-6)
    zero = four_over_six_block_candidates(torch.zeros(16))
    assert not zero.choose_m4.item()


def test_no_format_metadata_is_needed_for_nvfp4_4over6_decode():
    torch.manual_seed(602)
    x = torch.randn(3, 64)
    candidates = four_over_six_block_candidates(x)
    assert torch.equal(fake_quantize_nvfp4_4over6(x), candidates.selected)


def test_random_tensor_matches_official_pytorch_reference_equations():
    torch.manual_seed(603)
    x = torch.randn(5, 64, dtype=torch.float32) * 31
    torch.testing.assert_close(
        fake_quantize_nvfp4_4over6(x), _official_reference_port(x), rtol=0, atol=0
    )


def test_tile_selector_dominates_fair_e0_and_adaptive_e2_per_tile():
    torch.manual_seed(604)
    x = torch.randn(19, 77, dtype=torch.float32) * 5
    tile = fake_quantize_tile_mix_e0_e2_4over6(x)
    e0 = fake_quantize_e0m3_gscale1536(x)
    e2 = fake_quantize_nvfp4_4over6(x)
    assert torch.all(_tile_sse(x, tile) <= _tile_sse(x, e0) + 1e-6)
    assert torch.all(_tile_sse(x, tile) <= _tile_sse(x, e2) + 1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_format", sorted(FOUR_OVER_SIX_FORMATS))
def test_zero_extreme_noncontiguous_incomplete_shape(dtype, activation_format):
    zero = torch.zeros(3, 5, 19, dtype=dtype).transpose(0, 1)
    assert not zero.is_contiguous()
    zero_out = fake_quantize_four_over_six_activation(zero, activation_format)
    assert torch.equal(zero_out, zero)
    limit = torch.finfo(dtype).max
    base = torch.tensor(
        [limit, -limit, limit / 2, -limit / 2, 2.0**-20, 0.0], dtype=dtype
    ).repeat(7, 4)
    x = base.transpose(0, 1)
    out = fake_quantize_four_over_six_activation(x, activation_format)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("activation_format", sorted(FOUR_OVER_SIX_FORMATS))
def test_padding_and_high_precision_tail_do_not_enter_scale_or_selection(activation_format):
    torch.manual_seed(605)
    low = torch.randn(2, 3, 31)
    direct = fake_quantize_four_over_six_activation(low, activation_format)
    quantizer = ActQuantizer()
    quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=5,
        quant_dtype=activation_format,
    )
    tail_a = torch.randn(2, 3, 5)
    tail_b = torch.full_like(tail_a, 1.0e20)
    out_a = quantizer(torch.cat((low, tail_a), -1))
    out_b = quantizer(torch.cat((low, tail_b), -1))
    assert torch.equal(out_a[..., :31], direct)
    assert torch.equal(out_b[..., :31], direct)
    assert torch.equal(out_a[..., 31:], tail_a)
    assert torch.equal(out_b[..., 31:], tail_b)


def test_sidecar_is_observational_and_counts_actual_choices():
    torch.manual_seed(606)
    x = torch.randn(17, 70)
    plain = fake_quantize_tile_mix_e0_e2_4over6(x)
    stats = FourOverSixStats(selection_unit="tile")
    counted = fake_quantize_tile_mix_e0_e2_4over6(x, format_stats=stats)
    assert torch.equal(counted, plain)
    snapshot = stats.snapshot()
    candidates = four_over_six_block_candidates(x)
    assert snapshot["m4_count"] == int(candidates.choose_m4.sum())
    assert snapshot["block_count"] == candidates.choose_m4.numel()
    assert snapshot["tile_count"] == 4
    assert snapshot["tile_selected_sse"] <= snapshot["e0_tile_sse"] + 1e-6
    assert snapshot["tile_selected_sse"] <= snapshot["e2_tile_sse"] + 1e-6
    assert sum(snapshot["m4_occupancy"]) == x.numel()
    assert sum(snapshot["m6_occupancy"]) == x.numel()
    assert snapshot["m4_occupancy"][-1] == 0  # M=4 does not activate level 6


class _TimestepTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(timestep_scale=1000.0)

    def forward(self, x, *, timestep):
        return x + timestep.reshape(-1, *([1] * (x.ndim - 1))) * 0


def test_sidecar_uses_true_pipeline_timestep():
    stats = FourOverSixStats().for_layer("layer")
    transformer = _TimestepTransformer()
    pipeline = SimpleNamespace(scheduler=SimpleNamespace(timesteps=torch.tensor([100.0, 50.0])))
    stats.attach_timestep_source(transformer, pipeline)
    transformer(torch.ones(2, 4), timestep=torch.full((2,), 50_000.0))
    fake_quantize_nvfp4_4over6(torch.ones(2, 16), format_stats=stats)
    rows = stats.snapshot()["per_layer_timestep"]
    assert len(rows) == 1
    assert rows[0]["timestep_index"] == 1
    assert rows[0]["timestep"] == 50.0


@pytest.mark.parametrize(
    "activation_format",
    ["nvfp4", "nvfp4-hw", "e0m3", "block-mix-oracle",
     "tile-mix-oracle", "a16w4-residual"],
)
def test_existing_activation_formats_regression_is_unchanged(activation_format):
    torch.manual_seed(607)
    x = torch.randn(2, 3, 32)
    quantizer = ActQuantizer()
    quantizer.configure(
        bits=4, groupsize=16, sym=True, quant_dtype=activation_format
    )
    if activation_format == "a16w4-residual":
        assert torch.equal(quantizer(x), x)
    else:
        assert torch.equal(
            quantizer(x), fake_quantize_activation(x, activation_format)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_preserves_device_without_cpu_fallback(dtype):
    x = torch.randn(70, 17, device="cuda", dtype=dtype).transpose(0, 1)
    stats = FourOverSixStats(selection_unit="tile")
    out = fake_quantize_tile_mix_e0_e2_4over6(x, format_stats=stats)
    assert out.device.type == "cuda"
    assert out.dtype == dtype
    assert out.shape == x.shape
    assert stats._device is not None and stats._device.type == "cuda"
