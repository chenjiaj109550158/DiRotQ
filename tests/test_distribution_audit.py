from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.distribution_audit import (
    DistributionAuditCollector,
    _exact_candidate,
    _weighted_histogram_spearman,
)
from utils.quant_utils import ActQuantWrapper
from utils.tilemixfp4_utils import E0M3_MAGNITUDES, E2M1_MAGNITUDES


def test_exact_natural_scales_map_each_codebook_maximum_to_amax():
    blocks = torch.tensor([
        [0.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 2.5] * 2,
        [0.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0] * 2,
    ])
    q2, s2, _ = _exact_candidate(blocks[:1], E2M1_MAGNITUDES)
    q0, s0, _ = _exact_candidate(blocks[1:], E0M3_MAGNITUDES)
    assert s2.item() == pytest.approx(1.0)
    assert s0.item() == pytest.approx(1.0)
    assert q2.abs().max().item() == pytest.approx(6.0)
    assert q0.abs().max().item() == pytest.approx(7.0)


def test_online_histogram_spearman_has_expected_sign():
    positive = np.eye(8) * 10
    negative = np.fliplr(positive)
    assert _weighted_histogram_spearman(positive) > 0.99
    assert _weighted_histogram_spearman(negative) < -0.99


class _Transformer(nn.Module):
    def __init__(self, wrapper, timestep_scale=1.0):
        super().__init__()
        self.layer = wrapper
        self.config = SimpleNamespace(timestep_scale=timestep_scale)

    def forward(self, x, *, timestep):
        return self.layer(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sidecar_uses_true_timestep_and_does_not_change_sse_forward(tmp_path):
    torch.manual_seed(411)
    audited_wrapper = ActQuantWrapper(nn.Linear(64, 9, bias=False)).cuda().half()
    reference_wrapper = ActQuantWrapper(nn.Linear(64, 9, bias=False)).cuda().half()
    reference_wrapper.load_state_dict(audited_wrapper.state_dict())
    for wrapper in (audited_wrapper, reference_wrapper):
        wrapper.quantizer.configure(
            bits=4, groupsize=16, sym=True, quant_dtype="tile-mix-oracle"
        )

    transformer = _Transformer(audited_wrapper, timestep_scale=1000.0).cuda()
    pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(timesteps=torch.tensor([100.0, 50.0], device="cuda"))
    )
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"test-cache")
    collector = DistributionAuditCollector(
        "unit-test", tmp_path / "audit", "datasets/mjhq_5000_samples.json", [], cache
    )
    collector.attach(transformer, pipeline)
    batch = [collector.samples[0]]
    collector.start_batch(batch)
    for timestep in (100.0, 50.0):
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.float16)
        reference = reference_wrapper(x)
        audited = transformer(
            x, timestep=torch.full((2,), timestep * 1000.0, device="cuda")
        )
        assert torch.equal(audited, reference)
    collector.end_batch()

    assert collector.timestep_values == [100.0, 50.0]
    assert collector.exclusions == {}
    assert collector.layer_acc["tile_count"].sum().item() == 4
    assert collector.transition_acc["tile_count"].sum().item() == 2
    assert collector.weight_grams[(0, torch.device("cuda:0"), 64)].device.type == "cuda"
    assert audited_wrapper.distribution_audit is collector
