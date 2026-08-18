from __future__ import annotations

import torch
from torch import nn

from metrics.build_flux_pca_hessians import name_groups, reconstruct
from utils.gptq_utils import gptq_quantize_weights
from utils.quant_utils import ActQuantWrapper, _quant_group_int


def test_reconstruct_removes_known_pca_damping():
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(31, 8, generator=generator, dtype=torch.float64)
    covariance = x.T @ x / x.shape[0]
    damping = 0.01
    damped = covariance + damping * covariance.diagonal().mean() * torch.eye(8)
    values, vectors = torch.linalg.eigh(damped)
    mapping = {"x": vectors.float(), "x.eigenvalues": values.float()}
    restored = reconstruct(mapping, "x", damping)
    assert torch.allclose(restored.double(), 2 * covariance, atol=2e-5, rtol=2e-5)


def test_flux_hessian_name_groups_have_frozen_coverage():
    groups = list(name_groups())
    names = [name for _, members in groups for name in members]
    assert len(groups) == 228
    assert len(names) == len(set(names)) == 380
    assert all(not name.endswith(".net.2") for name in names)
    assert all("proj_out.linears.1" not in name for name in names)


def test_configured_rtn_is_not_reported_as_gptq_fallback(capsys):
    model = nn.Module()
    model.down = ActQuantWrapper(nn.Linear(8, 4, bias=False))
    model.down.quantizer.configure(bits=4, groupsize=4, sym=True)
    original = model.down.module.weight.detach().float().clone()
    expected = _quant_group_int(original, 4, 4, True)
    gptq_quantize_weights(
        model,
        {},
        bits=4,
        groupsize=4,
        sym=True,
        rtn_names=["down"],
        device="cpu",
    )
    assert torch.equal(model.down.module.weight.float(), expected)
    output = capsys.readouterr().out
    assert "0 GPTQ, 1 configured RTN, 0 RTN fallback" in output
