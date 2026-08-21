from __future__ import annotations

import torch
import pytest

from utils.flux_scheme_a import (
    build_hidden_residual_rotation,
    validate_hidden_residual_rotation,
)


def test_scheme_a_rotation_is_deterministic_and_preserves_high_tail(monkeypatch):
    # The production helper calls empty_cache even on CPU; keep this test
    # independent of CUDA availability.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    a = build_hidden_residual_rotation(12, 4, seed=17, device="cpu")
    b = build_hidden_residual_rotation(12, 4, seed=17, device="cpu")
    assert torch.equal(a, b)
    report = validate_hidden_residual_rotation(a, 4, atol=1e-12)
    assert report["high_rank"] == 4
    assert report["low_rank"] == 8
    assert torch.equal(a[-4:, -4:], torch.eye(4, dtype=torch.float64))
    assert torch.count_nonzero(a[:-4, -4:]) == 0
    assert torch.count_nonzero(a[-4:, :-4]) == 0


def test_scheme_a_rotation_rejects_cross_subspace_mixing():
    rotation = torch.eye(8, dtype=torch.float64)
    rotation[0, -1] = 0.1
    with pytest.raises(RuntimeError, match="invalid residual rotation"):
        validate_hidden_residual_rotation(rotation, 2)
