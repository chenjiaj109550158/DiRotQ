from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.flux_shared_pca_basis import (
    DOUBLE_FAMILIES,
    SINGLE_FAMILIES,
    FluxBasisConfig,
    build_flux_shared_basis,
    iter_sources,
    sharing_group,
    theoretical_basis_bytes,
    validate_flux_shared_basis,
)
from utils.quant_utils import ActQuantWrapper
from utils.shared_pca_basis import rotation_storage_report


def tiny_source():
    cfg = FluxBasisConfig(
        num_double_layers=4,
        num_single_layers=8,
        hidden=8,
        intermediate=12,
        representative_double=2,
        representative_single=4,
    )
    result = {}
    generator = torch.Generator().manual_seed(141)
    for source in iter_sources(cfg):
        width = cfg.intermediate if source.width == "down" else cfg.hidden
        x = torch.randn(2 + source.block, width, generator=generator)
        covariance = x.T @ x + torch.eye(width) * 0.1
        values, vectors = torch.linalg.eigh(covariance.double())
        result[source.key] = vectors.float()
        result[f"{source.key}.eigenvalues"] = values.float()
    return cfg, result


@pytest.mark.parametrize(
    "scheme,groups",
    [
        ("shared-width", 2),
        ("shared-operator", 12),
        ("shared-operator-stage4", 48),
        ("representative-operator", 12),
    ],
)
def test_flux_shared_basis_groups_alias_and_orthogonality(scheme, groups):
    cfg, source = tiny_source()
    before = {key: value.clone() for key, value in source.items()}
    derived, manifest = build_flux_shared_basis(source, scheme, cfg=cfg)
    report = validate_flux_shared_basis(derived, cfg=cfg, atol=2e-4)
    assert report["unique_groups"] == groups
    assert manifest["unique_group_count"] == groups
    for key, value in before.items():
        assert torch.equal(source[key], value)
    first, second = list(iter_sources(cfg))[:2]
    if sharing_group(scheme, first, cfg) == sharing_group(scheme, second, cfg):
        assert derived[first.key] is derived[second.key]


def test_flux_stage_partition_and_width_separation():
    cfg, _ = tiny_source()
    sources = list(iter_sources(cfg))
    double = next(x for x in sources if x.kind == "double" and x.block == 3 and x.family == "img_attn")
    single = next(x for x in sources if x.kind == "single" and x.block == 7 and x.family == "attn")
    down = next(x for x in sources if x.family == "img_ffn.down")
    assert sharing_group("shared-operator-stage4", double, cfg) == "double:stage3:img_attn"
    assert sharing_group("shared-operator-stage4", single, cfg) == "single:stage3:attn"
    assert sharing_group("shared-width", double, cfg) == "hidden"
    assert sharing_group("shared-width", down, cfg) == "down"


def test_flux_dense_rotation_memory_is_exact_and_monotonic():
    cfg, _ = tiny_source()
    baseline = theoretical_basis_bytes(None, cfg=cfg)
    width = theoretical_basis_bytes("shared-width", cfg=cfg)
    operator = theoretical_basis_bytes("shared-operator", cfg=cfg)
    stage = theoretical_basis_bytes("shared-operator-stage4", cfg=cfg)
    representative = theoretical_basis_bytes("representative-operator", cfg=cfg)
    assert width["unique_groups"] == 2
    assert operator["unique_groups"] == representative["unique_groups"] == 12
    assert stage["unique_groups"] == 48
    assert width["total_bytes"] < operator["total_bytes"] < stage["total_bytes"] < baseline["total_bytes"]
    assert width["total_bytes"] == (cfg.hidden**2 + cfg.intermediate**2) * 2


def test_flux_no_rotation_ffdown_memory_contract():
    cfg = FluxBasisConfig(include_down=False)
    baseline = theoretical_basis_bytes(None, cfg=cfg)
    width = theoretical_basis_bytes("shared-width", cfg=cfg)
    operator = theoretical_basis_bytes("shared-operator", cfg=cfg)
    stage = theoretical_basis_bytes("shared-operator-stage4", cfg=cfg)
    assert baseline["unique_groups"] == 228
    assert baseline["down_bytes"] == 0
    assert width["unique_groups"] == 1
    assert operator["unique_groups"] == 9
    assert stage["unique_groups"] == 36
    assert width["total_bytes"] == cfg.hidden**2 * 2


def wrapped(d):
    wrapper = ActQuantWrapper(nn.Linear(d, d, bias=False))
    wrapper.quantizer.configure(bits=4, groupsize=4, sym=True)
    return wrapper


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.to_q = wrapped(hidden)
        self.to_k = wrapped(hidden)
        self.to_v = wrapped(hidden)
        self.add_q_proj = wrapped(hidden)
        self.add_k_proj = wrapped(hidden)
        self.add_v_proj = wrapped(hidden)
        self.to_out = nn.ModuleList([wrapped(hidden)])
        self.to_add_out = wrapped(hidden)


class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        up = nn.Module()
        up.proj = wrapped(hidden)
        self.net = nn.ModuleList([up, nn.Identity(), wrapped(intermediate)])


class DoubleBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.attn = Attention(hidden)
        self.ff = FeedForward(hidden, intermediate)
        self.ff_context = FeedForward(hidden, intermediate)


class SingleAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.to_q = wrapped(hidden)
        self.to_k = wrapped(hidden)
        self.to_v = wrapped(hidden)


class SingleBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.attn = SingleAttention(hidden)
        self.proj_mlp = wrapped(hidden)
        split = nn.Module()
        split.linears = nn.ModuleList([wrapped(hidden), wrapped(intermediate)])
        self.proj_out = split


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            DoubleBlock(cfg.hidden, cfg.intermediate)
            for _ in range(cfg.num_double_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            SingleBlock(cfg.hidden, cfg.intermediate)
            for _ in range(cfg.num_single_layers)
        ])


def load_flux_model_utils():
    path = Path(__file__).resolve().parents[1] / "models/flux-schnell/model_utils.py"
    spec = importlib.util.spec_from_file_location("flux_shared_test_model_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flux_routing_preserves_shared_materialized_rotations():
    cfg, source = tiny_source()
    derived, _ = build_flux_shared_basis(source, "shared-width", cfg=cfg)
    transformer = Transformer(cfg)
    rotations = {
        "R1": torch.eye(cfg.hidden),
        "R2": torch.eye(4),
        "R_down": torch.eye(cfg.intermediate),
    }
    model_utils = load_flux_model_utils()
    assigned = model_utils.assign_online_rotations(
        transformer,
        derived,
        rotations,
        {"dims": {"num_heads": 2, "head": 4}},
        residual_rotation="random",
    )
    assert assigned == cfg.num_double_layers * 12 + cfg.num_single_layers * 6
    report = rotation_storage_report(transformer)
    assert report["unique_storages"] == 2
    assert report["unique_storage_bytes"] < report["logical_assignment_bytes"]
    double = transformer.transformer_blocks
    single = transformer.single_transformer_blocks
    assert double[0].attn.to_q.rotation is double[3].ff.net[0].proj.rotation
    assert double[0].attn.to_q.rotation is single[7].attn.to_v.rotation
    assert double[0].ff.net[2].rotation is single[7].proj_out.linears[1].rotation


def test_flux_no_rotation_ffdown_has_no_arbitrary_high_tail():
    cfg = FluxBasisConfig(
        num_double_layers=1, num_single_layers=1, hidden=8, intermediate=12,
        representative_double=0, representative_single=0, include_down=False,
    )
    source = {}
    for item in iter_sources(cfg):
        source[item.key] = torch.eye(cfg.hidden)
        source[f"{item.key}.eigenvalues"] = torch.arange(cfg.hidden).float()
    derived, _ = build_flux_shared_basis(source, "shared-width", cfg=cfg)
    transformer = Transformer(cfg)
    model_utils = load_flux_model_utils()
    model_utils.assign_online_rotations(
        transformer,
        derived,
        {"R1": torch.eye(8), "R2": torch.eye(4), "R_down": torch.eye(12)},
        {"dims": {"num_heads": 2, "head": 4}},
    )
    model_utils.configure_quantizers_by_name(
        transformer, 4, 1,
        {
            "dims": {"head": 4, "intermediate": 12},
            "quantization": {"a_bits": 4},
        },
        high_len_down=4,
    )
    assert transformer.transformer_blocks[0].ff.net[2].rotation is None
    assert transformer.transformer_blocks[0].ff.net[2].quantizer.high_bits_length == 0
    assert transformer.single_transformer_blocks[0].proj_out.linears[1].rotation is None
    assert transformer.single_transformer_blocks[0].proj_out.linears[1].quantizer.high_bits_length == 0


def test_flux_unquantized_shared_rotation_product_parity():
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(7, 8, generator=generator)
    weight = torch.randn(5, 8, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    reference = x @ weight.T
    transformed = (x @ rotation) @ (weight @ rotation).T
    assert torch.allclose(reference, transformed, atol=2e-5, rtol=2e-5)
