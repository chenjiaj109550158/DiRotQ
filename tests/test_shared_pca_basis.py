from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.quant_utils import ActQuantWrapper
from utils.shared_pca_basis import (
    FAMILIES,
    PixArtBasisConfig,
    basis_key,
    build_shared_basis,
    hessian_key,
    rotation_storage_report,
    sharing_group,
    speedup_rotation_parity,
    unique_tensor_bytes,
    validate_shared_basis,
)


def _tiny_inputs(num_layers=4):
    cfg = PixArtBasisConfig(
        num_layers=num_layers, hidden=8, num_heads=2, head_dim=4,
        intermediate=12, damping=0.01, representative_block=min(2, num_layers - 1),
    )
    source = {}
    hessians = {}
    gen = torch.Generator().manual_seed(123)
    for block in range(num_layers):
        for family in FAMILIES:
            d = cfg.intermediate if family == "ffn.down_proj" else cfg.hidden
            x = torch.randn(3 + block, d, generator=gen)
            H = x.T @ x + torch.eye(d) * (block + 1) * 0.01
            hessians[hessian_key(block, family)] = H.float()
            key = basis_key(block, family)
            if family.endswith(".value"):
                U = torch.eye(cfg.head_dim).repeat(cfg.num_heads, 1, 1)
                e = torch.arange(cfg.head_dim).repeat(cfg.num_heads, 1).float()
            else:
                U = torch.eye(d)
                e = torch.arange(d).float()
            source[key] = U
            source[f"{key}.eigenvalues"] = e
    return cfg, source, hessians


@pytest.mark.parametrize(
    "scheme,groups",
    [("shared-width", 3), ("shared-operator", 6),
     ("representative-operator", 6)],
)
def test_build_shared_basis_aliases_and_orthogonality(scheme, groups):
    cfg, source, hessians = _tiny_inputs()
    before = {key: value.clone() for key, value in source.items()}
    derived, manifest = build_shared_basis(source, hessians, scheme, cfg=cfg)
    report = validate_shared_basis(derived, cfg=cfg, atol=1e-4)
    assert report["unique_groups"] == groups
    assert manifest["unique_group_count"] == groups
    for key, value in before.items():
        assert torch.equal(source[key], value)
    # Alias storage is genuinely shared rather than copied per block.
    assert derived[basis_key(0, "ffn") ] is derived[basis_key(1, "ffn")]
    assert unique_tensor_bytes(derived) < sum(
        value.untyped_storage().nbytes()
        for value in derived.values() if isinstance(value, torch.Tensor)
    )


def test_stage4_partition_is_fixed():
    assert sharing_group("shared-operator-stage4", 0, "ffn") == "stage0:ffn"
    assert sharing_group("shared-operator-stage4", 6, "ffn") == "stage0:ffn"
    assert sharing_group("shared-operator-stage4", 7, "ffn") == "stage1:ffn"
    assert sharing_group("shared-operator-stage4", 27, "ffn") == "stage3:ffn"


def test_per_head_covariance_uses_head_blocks_only():
    cfg, source, hessians = _tiny_inputs(num_layers=2)
    for block in range(2):
        key = hessian_key(block, "self_attn.value")
        H = torch.zeros(cfg.hidden, cfg.hidden)
        H[:4, :4] = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        H[4:, 4:] = torch.diag(torch.tensor([5.0, 6.0, 7.0, 8.0]))
        H[:4, 4:] = 999  # must not enter the block-diagonal per-head PCA
        H[4:, :4] = 999
        hessians[key] = H
    derived, _ = build_shared_basis(source, hessians, "shared-operator", cfg=cfg)
    U = derived[basis_key(0, "self_attn.value")]
    assert tuple(U.shape) == (2, 4, 4)
    assert torch.allclose(U.transpose(-1, -2) @ U, torch.eye(4)[None], atol=1e-5)


def test_speedup_rotation_requires_counter_rotated_weight():
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(7, 8, generator=gen)
    weight = torch.randn(5, 8, generator=gen)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen))
    reference, correct = speedup_rotation_parity(
        x, weight, rotation, counter_rotate_weight=True,
    )
    _, literal_speed_script = speedup_rotation_parity(
        x, weight, rotation, counter_rotate_weight=False,
    )
    assert torch.allclose(reference, correct, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(reference, literal_speed_script, atol=1e-2, rtol=1e-2)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_q = ActQuantWrapper(nn.Linear(d, d, bias=False))
        self.to_k = ActQuantWrapper(nn.Linear(d, d, bias=False))
        self.to_v = ActQuantWrapper(nn.Linear(d, d, bias=False))
        self.to_out = nn.ModuleList([ActQuantWrapper(nn.Linear(d, d, bias=False))])


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn1 = _Attn(d)


class _Transformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(d), _Block(d)])


def _load_pixart_model_utils():
    path = Path(__file__).resolve().parents[1] / "models/pixart-sigma/model_utils.py"
    spec = importlib.util.spec_from_file_location("pixart_shared_test_model_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pixart_routing_reuses_materialized_shared_rotation():
    d = 8
    transformer = _Transformer(d)
    shared = torch.eye(d)
    per_head = torch.eye(4).repeat(2, 1, 1)
    basis = {
        "__shared_basis_map__": {},
    }
    for block in range(2):
        sa = basis_key(block, "self_attn")
        value = basis_key(block, "self_attn.value")
        basis[sa] = shared
        basis[f"{sa}.eigenvalues"] = torch.arange(d).float()
        basis[value] = per_head
        basis[f"{value}.eigenvalues"] = torch.arange(4).repeat(2, 1).float()
        basis["__shared_basis_map__"][sa] = "shared-sa"
        basis["__shared_basis_map__"][value] = "shared-value"
    rotations = {"R1": torch.eye(d), "R2": torch.eye(4), "R_down": torch.eye(12)}
    cfg = {"dims": {"num_heads": 2, "head": 4, "intermediate": 12}}
    model_utils = _load_pixart_model_utils()
    assigned = model_utils.assign_online_rotations(
        transformer, basis, rotations, cfg, residual_rotation="random"
    )
    assert assigned == 8
    b0, b1 = transformer.transformer_blocks
    assert b0.attn1.to_q.rotation is b0.attn1.to_k.rotation
    assert b0.attn1.to_q.rotation is b1.attn1.to_q.rotation
    assert b0.attn1.to_out[0].rotation_per_head is b1.attn1.to_out[0].rotation_per_head
    report = rotation_storage_report(transformer)
    assert report["assignments"] == 8
    assert report["unique_storages"] == 2
    assert report["unique_storage_bytes"] < report["logical_assignment_bytes"]
