import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from apply_dirotq import (
    gptq_hessian_cache_name,
    identity_rotation_metadata,
    quantized_weight_cache_name,
)
from utils.quant_utils import ActQuantWrapper
from utils.tilemixfp4_utils import FormatSelectionStats
from utils.fouroversix_utils import FOUR_OVER_SIX_FORMATS, FourOverSixStats


def _load_sana_model_utils():
    path = Path(__file__).parents[1] / "models" / "sana-1.6b" / "model_utils.py"
    spec = importlib.util.spec_from_file_location("sana_model_utils_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = ActQuantWrapper(nn.Linear(32, 32, bias=False))
        self.to_out = nn.ModuleList([ActQuantWrapper(nn.Linear(32, 32, bias=False))])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attention()


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])


def _orthogonal(n):
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float32))
    return q


@pytest.mark.parametrize(
    "activation_format",
    ["nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle",
     "tile-mix-output-oracle", *sorted(FOUR_OVER_SIX_FORMATS)],
)
def test_sana_routes_hardware_activation_formats(activation_format):
    model_utils = _load_sana_model_utils()
    transformer = _Transformer()
    stats = (FourOverSixStats(selection_unit="tile")
             if activation_format in FOUR_OVER_SIX_FORMATS
             else FormatSelectionStats(selection_unit="tile"))
    cfg = {
        "quantization": {"a_bits": 4},
        "dims": {"head": 32},
        "nvfp4": {"a_groupsize": 16, "a_groupsize_attn_out": 32},
    }

    model_utils.configure_quantizers_by_name(
        transformer,
        high_len_hidden=16,
        high_len_head=4,
        cfg=cfg,
        nvfp4=True,
        activation_format=activation_format,
        format_stats=stats,
    )

    q_quantizer = transformer.transformer_blocks[0].attn1.to_q.quantizer
    out_quantizer = transformer.transformer_blocks[0].attn1.to_out[0].quantizer
    assert q_quantizer.quant_dtype == activation_format
    assert out_quantizer.quant_dtype == activation_format
    assert q_quantizer.groupsize == 16
    assert out_quantizer.groupsize == 32
    assert q_quantizer.high_bits_length == 16
    assert out_quantizer.high_bits_length == 4
    assert q_quantizer.clip_ratio == out_quantizer.clip_ratio == 1.0
    if activation_format in FOUR_OVER_SIX_FORMATS:
        assert q_quantizer.format_stats._root is stats
        assert out_quantizer.format_stats._root is stats
    else:
        assert q_quantizer.format_stats is stats
        assert out_quantizer.format_stats is stats


@pytest.mark.parametrize(
    "activation_format",
    ["nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle"],
)
def test_sana_identity_routes_all_activation_formats_without_r_tensors(
    activation_format,
):
    model_utils = _load_sana_model_utils()
    transformer = _Transformer()
    hidden_basis = _orthogonal(32)
    head_basis = _orthogonal(32).unsqueeze(0)
    basis = {
        "layer.0.self_attn": hidden_basis,
        "layer.0.self_attn.value": head_basis,
    }
    metadata_only = {
        "high_len_hidden": 4,
        "high_len_head": 4,
        "high_len_down": 8,
    }
    cfg = {
        "quantization": {"a_bits": 4},
        "dims": {"num_heads": 1, "head": 32, "intermediate": 64},
        "nvfp4": {"a_groupsize": 16, "a_groupsize_attn_out": 32},
    }
    model_utils.assign_online_rotations(
        transformer, basis, metadata_only, cfg, residual_rotation="identity"
    )
    model_utils.configure_quantizers_by_name(
        transformer, 4, 4, cfg, nvfp4=True,
        activation_format=activation_format,
    )
    q_wrapper = transformer.transformer_blocks[0].attn1.to_q
    out_wrapper = transformer.transformer_blocks[0].attn1.to_out[0]
    assert torch.equal(q_wrapper.rotation, hidden_basis)
    assert torch.equal(out_wrapper.rotation_per_head, head_basis)
    assert q_wrapper.quantizer.quant_dtype == activation_format
    assert out_wrapper.quantizer.quant_dtype == activation_format


def test_sana_random_rotation_regression_is_u_times_r():
    model_utils = _load_sana_model_utils()
    transformer = _Transformer()
    hidden_basis = _orthogonal(32)
    head_basis = _orthogonal(32).unsqueeze(0)
    r1, r2 = _orthogonal(32), _orthogonal(32)
    basis = {
        "layer.0.self_attn": hidden_basis,
        "layer.0.self_attn.value": head_basis,
    }
    rotations = {"R1": r1, "R2": r2}
    cfg = {"dims": {"num_heads": 1, "head": 32}}
    model_utils.assign_online_rotations(
        transformer, basis, rotations, cfg, residual_rotation="random"
    )
    assert torch.equal(transformer.transformer_blocks[0].attn1.to_q.rotation,
                       hidden_basis @ r1)
    assert torch.equal(
        transformer.transformer_blocks[0].attn1.to_out[0].rotation_per_head,
        torch.bmm(head_basis, r2.unsqueeze(0)),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sana_random_and_identity_transformed_weights_are_unquantized_equivalent(dtype):
    torch.manual_seed(211)
    dim, high = 32, 4
    u = _orthogonal(dim)
    r = torch.block_diag(_orthogonal(dim - high), torch.eye(high))
    x = torch.randn(2, 5, dim, dtype=dtype)
    weight = torch.randn(17, dim, dtype=dtype)
    random_basis = u @ r
    identity_basis = u
    y_random = F.linear(
        (x.float() @ random_basis).to(dtype),
        (weight.float() @ random_basis).to(dtype),
    ).float()
    y_identity = F.linear(
        (x.float() @ identity_basis).to(dtype),
        (weight.float() @ identity_basis).to(dtype),
    ).float()
    tolerance = 3e-2 if dtype == torch.float16 else 2e-1
    torch.testing.assert_close(y_random, y_identity, rtol=2e-2, atol=tolerance)


def test_sana_cache_keys_separate_residual_modes_and_random_is_legacy():
    prefix = "nvfp4_g16_gptq"
    assert quantized_weight_cache_name(prefix, "random") == "nvfp4_g16_gptq_model.pt"
    assert quantized_weight_cache_name(prefix, "identity") == (
        "nvfp4_g16_gptq_rr-identity_model.pt"
    )
    assert gptq_hessian_cache_name(5120, 120, "random") == "hessians_n5120_l120.pt"
    assert gptq_hessian_cache_name(5120, 120, "identity") == (
        "hessians_n5120_l120_rr-identity.pt"
    )


def test_sana_identity_split_metadata_matches_rotation_generator_rounding():
    cfg = {
        "rotation": {"high_fraction": 0.125},
        "dims": {"hidden": 2240, "head": 32, "intermediate": 5600},
    }
    assert identity_rotation_metadata(cfg) == {
        "high_len_hidden": 280,
        "high_len_head": 4,
        "high_len_down": 700,
        "high_fraction": 0.125,
    }
