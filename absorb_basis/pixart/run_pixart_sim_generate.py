"""Generate MJHQ images for PixArt-Sigma with DiRotQ-absorb-basis simulated
W4A4 NVFP4 quantization, using deepcompressor's own pipeline factory and
DiffusionEvalConfig.generate (identical protocol/seeds to the SVDQuant runs).

Per quantized linear the forward is the nunchaku-kernel semantics:
  y = Q4_act(x) @ W_q^T + (x @ lora_down) @ lora_up^T + bias
with dynamic per-group-16 fp4 activation quantization (e4m3 micro-scales).

Run in the svdquant env from deepcompressor/examples/diffusion:
  python run_pixart_sim_generate.py configs/model/pixart-sigma.yaml \
      --sim-weights <sim.pt> --gen-root <dir> \
      --eval-benchmarks MJHQ --eval-num-samples 2500 --eval-num-gpus 1 \
      --eval-batch-size 1 --skip-eval
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def act_fp4_sim(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    """Dynamic NVFP4 activation quantization (per-group-16 e4m3 scale),
    bucketize fast path (equivalent to nearest-code rounding)."""
    shp = x.shape
    xg = x.float().reshape(-1, shp[-1] // group, group)
    s = (xg.abs().amax(dim=-1, keepdim=True) / 6.0).clamp(min=1e-8)
    s = s.to(torch.float8_e4m3fn).float().clamp(min=2.0 ** -9)
    y = xg / s
    idx = torch.bucketize(y.abs(), E2M1_BOUNDS.to(x.device))
    q = E2M1_POS.to(x.device)[idx] * y.sign() * s
    return q.reshape(shp).to(x.dtype)


class SimW4A4Linear(nn.Module):
    def __init__(self, W_q, lora_down, lora_up, bias):
        super().__init__()
        self.register_buffer("W_q", W_q)
        self.register_buffer("lora_down", lora_down)
        self.register_buffer("lora_up", lora_up)
        self.register_buffer("bias_", bias if bias is not None else None)

    def forward(self, x):
        y = act_fp4_sim(x) @ self.W_q.t()
        y = y + (x @ self.lora_down) @ self.lora_up.t()
        if self.bias_ is not None:
            y = y + self.bias_
        return y


def inject(transformer, sim):
    n_replaced = 0
    for wpath, t in sim.items():
        parts = wpath.split(".")
        parent = transformer
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        old = getattr(parent, parts[-1]) if not parts[-1].isdigit() else parent[int(parts[-1])]
        assert isinstance(old, nn.Linear), wpath
        dev, dt = old.weight.device, old.weight.dtype
        mod = SimW4A4Linear(
            t["W_q"].to(dev, dt), t["lora_down"].to(dev, dt), t["lora_up"].to(dev, dt),
            old.bias.detach().clone() if old.bias is not None else None,
        )
        if parts[-1].isdigit():
            parent[int(parts[-1])] = mod
        else:
            setattr(parent, parts[-1], mod)
        n_replaced += 1
    return n_replaced


if __name__ == "__main__":
    import deepcompressor.app.diffusion.pipeline.config as _plcfg
    # up-block conv rewrite is a no-op for DiT models; skip (crashes on some
    # diffusers versions) — mirrors the FLUX bf16-reference driver.
    _plcfg.replace_up_block_conv_with_concat_conv = lambda model: model

    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--sim-weights", required=True)
    ap.add_argument("--gen-root", required=True)
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    config, _, unused_cfgs, _, unknown = DiffusionPtqRunConfig.get_parser().parse_known_args()
    assert not config.quant.is_enabled()

    config.output.lock()
    pipeline = config.pipeline.build()
    sim = torch.load(own.sim_weights, map_location="cpu", weights_only=False)
    n = inject(pipeline.transformer, sim)
    print(f"replaced {n} linears with SimW4A4Linear")

    config.eval.gen_root = own.gen_root
    config.eval.generate(pipeline, task=config.pipeline.task)
    config.output.unlock()
