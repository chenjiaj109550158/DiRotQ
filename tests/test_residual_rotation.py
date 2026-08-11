import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from apply_dirotq import (
    gptq_hessian_cache_name,
    quantized_weight_cache_name,
    residual_rotation_cache_tag,
)
from utils.quant_utils import ActQuantWrapper


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pixart_model_utils_for_test", ROOT / "models/pixart-sigma/model_utils.py"
)
PIXART_MODEL_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIXART_MODEL_UTILS)


class _OneLayerTransformer:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def named_modules(self):
        yield "transformer_blocks.0.attn1.to_q", self.wrapper


def _orthogonal(n, dtype=torch.float32):
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float32))
    return q.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_random_and_identity_residual_bases_are_unquantized_equivalent(dtype):
    torch.manual_seed(101)
    dim, high = 16, 4
    u = _orthogonal(dim)
    r = torch.block_diag(_orthogonal(dim - high), torch.eye(high))
    x = torch.randn(3, 5, dim, dtype=dtype)
    weight = torch.randn(11, dim, dtype=dtype)

    random_basis = u @ r
    identity_basis = u
    # Match the implementation's float32 basis transforms, then compare the
    # unquantized outputs at the requested activation/weight storage dtype.
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


def test_identity_assignment_does_not_access_random_rotation_tensor():
    wrapper = ActQuantWrapper(torch.nn.Linear(8, 8, bias=False))
    transformer = _OneLayerTransformer(wrapper)
    basis = {"layer.0.self_attn": _orthogonal(8)}
    rotation_metadata_only = {
        "high_len_hidden": 2,
        "high_len_head": 1,
        "high_len_down": 2,
    }
    cfg = {"dims": {"num_heads": 2, "head": 4, "intermediate": 8}}

    PIXART_MODEL_UTILS.assign_online_rotations(
        transformer, basis, rotation_metadata_only, cfg,
        residual_rotation="identity",
    )
    assert torch.equal(wrapper.rotation, basis["layer.0.self_attn"].float())


def test_random_assignment_regression_is_u_times_r():
    wrapper = ActQuantWrapper(torch.nn.Linear(8, 8, bias=False))
    transformer = _OneLayerTransformer(wrapper)
    u, r = _orthogonal(8), _orthogonal(8)
    basis = {"layer.0.self_attn": u}
    rotations = {"R1": r, "R2": _orthogonal(4), "R_down": _orthogonal(8)}
    cfg = {"dims": {"num_heads": 2, "head": 4, "intermediate": 8}}

    PIXART_MODEL_UTILS.assign_online_rotations(
        transformer, basis, rotations, cfg, residual_rotation="random"
    )
    assert torch.equal(wrapper.rotation, u @ r)


def test_residual_rotation_cache_keys_are_separate_and_random_is_legacy():
    prefix = "nvfp4_g16_gptq_skipc27eed7e"
    random_weight = quantized_weight_cache_name(prefix, "random")
    identity_weight = quantized_weight_cache_name(prefix, "identity")
    random_hessian = gptq_hessian_cache_name(5120, 224, "random")
    identity_hessian = gptq_hessian_cache_name(5120, 224, "identity")

    assert residual_rotation_cache_tag("random") == ""
    assert random_weight == "nvfp4_g16_gptq_skipc27eed7e_model.pt"
    assert random_hessian == "hessians_n5120_l224.pt"
    assert identity_weight == "nvfp4_g16_gptq_skipc27eed7e_rr-identity_model.pt"
    assert identity_hessian == "hessians_n5120_l224_rr-identity.pt"
    assert random_weight != identity_weight
    assert random_hessian != identity_hessian
