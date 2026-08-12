import json

import pytest
import torch

from utils.quant_utils import _quant_group_nvfp4
from utils.weight_mixfp4 import (
    WEIGHT_GROUP_SIZE,
    expected_metadata,
    gptq_quantize_weight_tiles,
    hessian_trace_loss,
    logical_tile_to_stored_slice,
    metadata_path,
    quantize_weight_blocks,
    validate_cache_metadata,
)


def test_stored_and_logical_weight_tile_mapping():
    n_slice, k_slice = logical_tile_to_stored_slice(2, 3)
    assert (n_slice.start, n_slice.stop) == (24, 32)
    assert (k_slice.start, k_slice.stop) == (128, 192)
    stored = torch.arange(40 * 256).reshape(40, 256)
    tile = stored[n_slice, k_slice]
    assert tile.shape == (8, 64)
    assert tile.reshape(8, 4, WEIGHT_GROUP_SIZE).shape == (8, 4, 16)
    assert tile.numel() // WEIGHT_GROUP_SIZE == 32


@pytest.mark.parametrize("shape", [(8, 64), (7, 70), (3, 9)])
def test_fixed_e2_primitive_regresses_existing_quantizer(shape):
    torch.manual_seed(801)
    source = torch.randn(*shape)
    actual = quantize_weight_blocks(source, "e2")
    expected = _quant_group_nvfp4(source, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fixed_e0_codebook_scale_and_zero_partial_block():
    source = torch.tensor([
        [0., 1., 2., 3., 4., 5., 6., 7.] + [0.] * 9,
        [0.] * 17,
    ])
    quantized = quantize_weight_blocks(source, "e0")
    torch.testing.assert_close(quantized[0, :8], source[0, :8])
    assert torch.equal(quantized[1], source[1])
    assert quantized.shape == source.shape and torch.isfinite(quantized).all()


def test_hessian_trace_matches_direct_activation_error():
    torch.manual_seed(802)
    rows, k, n = 19, 13, 7
    z = torch.randn(rows, k)
    source = torch.randn(n, k)
    quantized = source + .1 * torch.randn_like(source)
    hessian = 2 / rows * (z.T @ z)
    trace = hessian_trace_loss(source, quantized, hessian)
    direct = 2 / rows * (z @ (source - quantized).T).double().square().sum()
    torch.testing.assert_close(trace, direct, rtol=2e-5, atol=1e-6)


def test_cache_metadata_isolated_by_weight_format(tmp_path):
    basis = tmp_path / "basis.pt"
    rotation = tmp_path / "rotation.pt"
    basis.write_bytes(b"basis")
    rotation.write_bytes(b"rotation")
    e2 = expected_metadata(
        model="sana-1.6b", mode="fixed-e2", calibration_count=8,
        damp_pct=.01, basis_path=basis, rotation_path=rotation,
        skip_layers=["attn2.to_k"],
    )
    e0 = expected_metadata(
        model="sana-1.6b", mode="fixed-e0", calibration_count=8,
        damp_pct=.01, basis_path=basis, rotation_path=rotation,
        skip_layers=["attn2.to_k"],
    )
    cache = tmp_path / "weightmix.pt"
    cache.write_bytes(b"cache")
    metadata_path(cache).write_text(json.dumps(e2))
    assert validate_cache_metadata(cache, e2)["weight_format"] == "fixed-e2"
    with pytest.raises(RuntimeError, match="weight_format"):
        validate_cache_metadata(cache, e0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tilemix_candidate_same_state_dominance_and_one_format_per_8x64(dtype):
    torch.manual_seed(803)
    source = torch.randn(8, 64, device="cuda", dtype=dtype)
    hessian = torch.eye(64, device="cuda")
    e2, _ = gptq_quantize_weight_tiles(source, hessian, "fixed-e2")
    e0, _ = gptq_quantize_weight_tiles(source, hessian, "fixed-e0")
    mixed, stats = gptq_quantize_weight_tiles(source, hessian, "tilemix")
    assert stats["total_count"] == 1
    assert stats["e2_count"] + stats["e0_count"] == 1
    assert stats["incremental_selected"] <= min(
        stats["incremental_e2"], stats["incremental_e0"]
    ) + 1e-6
    assert torch.equal(mixed, e0) or torch.equal(mixed, e2)
    chosen_loss = hessian_trace_loss(source, mixed, hessian)
    assert chosen_loss <= min(
        hessian_trace_loss(source, e2, hessian),
        hessian_trace_loss(source, e0, hessian),
    ) + 1e-4
    assert mixed.dtype == torch.float32 and mixed.device.type == "cuda"
    assert torch.isfinite(mixed).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_weight_mix_zero_padding_incomplete_nk_and_blockmix():
    source = torch.zeros(11, 70, device="cuda", dtype=torch.bfloat16)
    hessian = torch.eye(70, device="cuda")
    for mode in ("fixed-e2", "fixed-e0", "tilemix", "blockmix"):
        quantized, stats = gptq_quantize_weight_tiles(source, hessian, mode)
        assert quantized.shape == source.shape
        assert torch.count_nonzero(quantized) == 0
        assert torch.isfinite(quantized).all()
        if mode == "tilemix":
            # ceil(N/8) * ceil(K/64) = 2 * 2 legal tiles; padding is not a tile.
            assert stats["total_count"] == 4
            assert stats["e0_count"] == 0  # strict ties choose E2
        if mode == "blockmix":
            # N * ceil(K/16) stored 1x16 blocks.
            assert stats["total_count"] == 11 * 5
            assert stats["e0_count"] == 0


def test_gptq_weight_mix_rejects_cpu_fallback():
    with pytest.raises(RuntimeError, match="CPU fallback"):
        gptq_quantize_weight_tiles(
            torch.randn(8, 64), torch.eye(64), "tilemix"
        )
