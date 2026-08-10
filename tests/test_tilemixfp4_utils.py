import pytest
import torch
import torch.nn.functional as F

from utils.quant_utils import ActQuantizer
from utils.tilemixfp4_utils import (
    E0M3_MAGNITUDES,
    E0M3_MAX_BLOCK_SCALE,
    E2M1_MAGNITUDES,
    E2M1_MAX_BLOCK_SCALE,
    FormatSelectionStats,
    GLOBAL_SCALED_MAX,
    _e4m3_block_scale,
    _hardware_global_scale,
    _pad_k_to,
    _quantize_blocks_e4m3,
    fake_quantize_activation,
    fake_quantize_block_mix_oracle,
    fake_quantize_e0m3,
    fake_quantize_e2m1,
    fake_quantize_nvfp4_hw,
    fake_quantize_nvfp4_legacy,
    fake_quantize_tile_mix_oracle,
)


FORMATS = (
    "nvfp4", "nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle",
)
HW_FORMATS = ("nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_format", FORMATS)
def test_zero_tensor(dtype, activation_format):
    x = torch.zeros(3, 5, 37, dtype=dtype)
    out = fake_quantize_activation(x, activation_format)
    assert torch.equal(out, x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


@pytest.mark.parametrize("activation_format", HW_FORMATS)
def test_negative_zero_reconstructs_as_zero(activation_format):
    x = torch.full((2, 19), -0.0, dtype=torch.float32)
    out = fake_quantize_activation(x, activation_format)
    assert torch.equal(out, torch.zeros_like(out))
    assert not torch.signbit(out).any()


def test_e2m1_known_codebook_values():
    # max=2688 makes s32=1 and E2M1 block scale exactly 448.
    values = torch.tensor(
        E2M1_MAGNITUDES + tuple(-v for v in E2M1_MAGNITUDES), dtype=torch.float32
    ) * E2M1_MAX_BLOCK_SCALE
    out = fake_quantize_e2m1(values)
    assert torch.equal(out, values)


def test_e0m3_known_codebook_values():
    # max=2688 makes s32=1 and E0M3 block scale exactly 384.
    values = torch.tensor(
        E0M3_MAGNITUDES + tuple(-v for v in E0M3_MAGNITUDES), dtype=torch.float32
    ) * E0M3_MAX_BLOCK_SCALE
    out = fake_quantize_e0m3(values)
    assert torch.equal(out, values)


def test_global_amax_maps_to_2688():
    x = torch.tensor([[-91.0, 7.0, 8192.0], [3.0, -2.0, 1.0]])
    scaled, s32 = _hardware_global_scale(x)
    assert s32.dtype == torch.float32
    assert torch.equal(s32, x.abs().amax() / GLOBAL_SCALED_MAX)
    assert torch.equal(scaled.abs().amax(), torch.tensor(GLOBAL_SCALED_MAX))


def test_candidate_block_scale_bounds():
    x = torch.linspace(-65504.0, 65504.0, 47).reshape(1, -1)
    scaled, _ = _hardware_global_scale(x)
    padded, _ = _pad_k_to(scaled, 16)
    blocks = padded.reshape(1, -1, 16)
    e2_scale = _e4m3_block_scale(blocks, E2M1_MAGNITUDES)
    e0_scale = _e4m3_block_scale(blocks, E0M3_MAGNITUDES)
    assert e2_scale.max() <= E2M1_MAX_BLOCK_SCALE
    assert e0_scale.max() <= E0M3_MAX_BLOCK_SCALE
    assert torch.isfinite(e2_scale).all()
    assert torch.isfinite(e0_scale).all()


@pytest.mark.parametrize(
    "fn",
    [fake_quantize_e2m1, fake_quantize_e0m3,
     fake_quantize_block_mix_oracle, fake_quantize_tile_mix_oracle],
)
def test_positive_negative_symmetry(fn):
    torch.manual_seed(7)
    x = torch.randn(5, 29, dtype=torch.float32)
    assert torch.equal(fn(-x), -fn(x))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_format", FORMATS)
def test_random_shape_dtype_device_and_finite(dtype, activation_format):
    torch.manual_seed(11)
    x = torch.randn(2, 7, 67, dtype=dtype)
    out = fake_quantize_activation(x, activation_format)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("activation_format", FORMATS)
def test_non_contiguous_input(activation_format):
    base = torch.randn(4, 23, dtype=torch.float16)
    x = base.transpose(0, 1)
    assert not x.is_contiguous()
    out = fake_quantize_activation(x, activation_format)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


def test_incomplete_m_and_k_tile():
    x = torch.randn(2, 3, 19, dtype=torch.bfloat16)  # flattened M=6, K=19
    out = fake_quantize_tile_mix_oracle(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_format", HW_FORMATS)
def test_extreme_finite_magnitudes(dtype, activation_format):
    limit = torch.finfo(dtype).max
    values = torch.tensor(
        [limit, -limit, limit / 2, -limit / 2, 1.0, -1.0, 0.0], dtype=dtype
    ).repeat(3, 5)
    out = fake_quantize_activation(values, activation_format)
    assert out.shape == values.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_fixed_hw_e2m1_matches_oracle_e2_candidate_builder():
    torch.manual_seed(17)
    x = torch.randn(5, 37, dtype=torch.float32) * 103.0
    scaled, s32 = _hardware_global_scale(x)
    padded, _ = _pad_k_to(scaled, 16)
    blocks = padded.reshape(x.shape[0], -1, 16)
    candidate = (_quantize_blocks_e4m3(blocks, E2M1_MAGNITUDES) * s32)
    candidate = candidate.reshape(x.shape[0], -1)[:, :x.shape[-1]]
    assert torch.equal(fake_quantize_nvfp4_hw(x), candidate)


def _block_errors(original, reconstructed):
    k = original.shape[-1]
    pad_k = (-k) % 16
    x = F.pad(original.float(), (0, pad_k))
    q = F.pad(reconstructed.float(), (0, pad_k))
    valid = torch.zeros_like(x, dtype=torch.bool)
    valid[..., :k] = True
    return (((x - q).square() * valid).reshape(-1, x.shape[-1] // 16, 16)
            .sum(dim=-1))


def test_block_oracle_error_not_above_either_candidate():
    torch.manual_seed(19)
    x = torch.randn(3, 31, dtype=torch.float16) * 2.5
    oracle = fake_quantize_block_mix_oracle(x)
    e2 = fake_quantize_e2m1(x)
    e0 = fake_quantize_e0m3(x)
    oracle_err = _block_errors(x, oracle)
    assert torch.all(oracle_err <= _block_errors(x, e2) + 1e-6)
    assert torch.all(oracle_err <= _block_errors(x, e0) + 1e-6)


def _tile_errors(original, reconstructed):
    m, k = original.shape
    pad_m = (-m) % 16
    pad_k = (-k) % 64
    x = F.pad(original.float(), (0, pad_k, 0, pad_m))
    q = F.pad(reconstructed.float(), (0, pad_k, 0, pad_m))
    valid = torch.zeros_like(x, dtype=torch.bool)
    valid[:m, :k] = True
    return (((x - q).square() * valid).reshape(
        x.shape[0] // 16, 16, x.shape[1] // 64, 64
    ).sum(dim=(1, 3)))


def test_tile_oracle_error_not_above_either_candidate_per_tile():
    torch.manual_seed(23)
    x = torch.randn(17, 70, dtype=torch.float16) * 3.0
    oracle = fake_quantize_tile_mix_oracle(x)
    e2 = fake_quantize_e2m1(x)
    e0 = fake_quantize_e0m3(x)
    oracle_err = _tile_errors(x, oracle)
    assert torch.all(oracle_err <= _tile_errors(x, e2) + 1e-6)
    assert torch.all(oracle_err <= _tile_errors(x, e0) + 1e-6)


def _expected_block_choices(x):
    scaled, _ = _hardware_global_scale(x.float())
    padded, _ = _pad_k_to(scaled, 16)
    valid = torch.zeros_like(padded, dtype=torch.bool)
    valid[:, :x.shape[-1]] = True
    blocks = padded.reshape(x.shape[0], -1, 16)
    valid = valid.reshape_as(blocks)
    e2 = _quantize_blocks_e4m3(blocks, E2M1_MAGNITUDES)
    e0 = _quantize_blocks_e4m3(blocks, E0M3_MAGNITUDES)
    return (((blocks - e0).square() * valid).sum(-1)
            < ((blocks - e2).square() * valid).sum(-1))


def _expected_tile_choices(x):
    m, k = x.shape
    scaled, _ = _hardware_global_scale(x.float())
    pad_m, pad_k = (-m) % 16, (-k) % 64
    padded = F.pad(scaled, (0, pad_k, 0, pad_m))
    valid = torch.zeros_like(padded, dtype=torch.bool)
    valid[:m, :k] = True
    mp, kp = padded.shape
    blocks = padded.reshape(mp, kp // 16, 16)
    e2 = _quantize_blocks_e4m3(blocks, E2M1_MAGNITUDES).reshape(mp, kp)
    e0 = _quantize_blocks_e4m3(blocks, E0M3_MAGNITUDES).reshape(mp, kp)
    e2_err = (((padded - e2).square() * valid)
              .reshape(mp // 16, 16, kp // 64, 64).sum((1, 3)))
    e0_err = (((padded - e0).square() * valid)
              .reshape(mp // 16, 16, kp // 64, 64).sum((1, 3)))
    return e0_err < e2_err


@pytest.mark.parametrize(
    ("oracle_fn", "selection_unit", "expected_fn"),
    [
        (fake_quantize_block_mix_oracle, "block", _expected_block_choices),
        (fake_quantize_tile_mix_oracle, "tile", _expected_tile_choices),
    ],
)
def test_format_counter_matches_actual_choices_without_changing_output(
    oracle_fn, selection_unit, expected_fn
):
    torch.manual_seed(25)
    x = torch.randn(17, 70, dtype=torch.float32) * 4.0
    expected = expected_fn(x)
    stats = FormatSelectionStats(selection_unit=selection_unit)
    plain = oracle_fn(x)
    counted = oracle_fn(x, format_stats=stats)
    assert torch.equal(counted, plain)
    snapshot = stats.snapshot()
    assert snapshot["selection_unit"] == selection_unit
    assert snapshot["e0m3_count"] == int(expected.sum())
    assert snapshot["e2m1_count"] == expected.numel() - int(expected.sum())
    assert snapshot["total_count"] == expected.numel()
    assert snapshot["e0m3_ratio"] == pytest.approx(float(expected.float().mean()))


@pytest.mark.parametrize(
    "oracle_fn",
    [fake_quantize_block_mix_oracle, fake_quantize_tile_mix_oracle],
)
def test_zero_padding_does_not_change_global_scale_or_format_decision(oracle_fn):
    torch.manual_seed(27)
    x = torch.randn(15, 63, dtype=torch.float32) * 5.0
    out = oracle_fn(x)
    explicitly_padded = F.pad(x, (0, 1, 0, 1))
    padded_out = oracle_fn(explicitly_padded)
    assert torch.equal(out, padded_out[:15, :63])


@pytest.mark.parametrize("activation_format", HW_FORMATS)
def test_high_precision_tail_does_not_affect_hardware_global_scale(activation_format):
    torch.manual_seed(28)
    low = torch.randn(2, 3, 32, dtype=torch.float32)
    tail_a = torch.randn(2, 3, 4, dtype=torch.float32)
    tail_b = torch.full_like(tail_a, 1.0e20)

    quantizer = ActQuantizer()
    quantizer.configure(
        bits=4,
        groupsize=16,
        sym=True,
        high_bits_length=4,
        quant_dtype=activation_format,
    )
    out_a = quantizer(torch.cat([low, tail_a], dim=-1))
    out_b = quantizer(torch.cat([low, tail_b], dim=-1))
    assert torch.equal(out_a[..., :32], out_b[..., :32])
    assert torch.equal(out_a[..., 32:], tail_a)
    assert torch.equal(out_b[..., 32:], tail_b)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nvfp4_cli_path_matches_existing_grouped_activation_quantizer(dtype):
    torch.manual_seed(29)
    x = torch.randn(2, 3, 35, dtype=dtype)
    quantizer = ActQuantizer()
    quantizer.configure(
        bits=4,
        groupsize=16,
        sym=True,
        high_bits_length=3,
        quant_dtype="nvfp4",
    )
    existing = quantizer(x)
    expected = torch.cat([
        fake_quantize_nvfp4_legacy(x[..., :32], block_size=16),
        x[..., 32:],
    ], dim=-1)
    assert torch.equal(existing, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_device_preserved(dtype):
    x = torch.randn(17, 70, device="cuda", dtype=dtype).transpose(0, 1)
    stats = FormatSelectionStats(selection_unit="tile")
    out = fake_quantize_tile_mix_oracle(x, format_stats=stats)
    assert out.device == x.device
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
    assert stats._counts is not None and stats._counts.device == x.device
    snapshot = stats.snapshot()
    expected_tiles = ((x.shape[0] + 15) // 16) * ((x.shape[1] + 63) // 64)
    assert snapshot["total_count"] == expected_tiles
