"""Generate images for SANA-1.6B with DiRotQ-absorb-basis running on the REAL
nunchaku fp4 kernel (SVDQW4A4Linear), via deepcompressor's own pipeline
factory and DiffusionEvalConfig.generate (identical protocol/seeds).

SANA dims need padding (hidden 2240 -> 2304): each replaced layer wraps the
padded SVDQW4A4Linear with an input zero-pad + output slice. 1x1 convs in
GLUMBConv are handled as linears over the channel dim.

Also records transformer-only memory and per-forward latency.

Run in the svdquant env from deepcompressor/examples/diffusion:
  python run_sana_kernel_generate.py configs/model/sana-1.6b.yaml \
      --kernel-weights <kernel.pt> --gen-root <dir> --stats-out <json> \
      --eval-benchmarks MJHQ --eval-num-samples 2500 --eval-num-gpus 1 \
      --eval-batch-size 1 --skip-eval
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

STATS = {"lat_ms": [], "entry_alloc": [], "peak_alloc": []}


def attach_probes(transformer):
    ev = {}

    def pre(module, args, kwargs):
        torch.cuda.synchronize()
        STATS["entry_alloc"].append(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        ev["start"] = s
        return None

    def post(module, args, kwargs, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        e.synchronize()
        STATS["lat_ms"].append(ev["start"].elapsed_time(e))
        STATS["peak_alloc"].append(torch.cuda.max_memory_allocated())
        return None

    transformer.register_forward_pre_hook(pre, with_kwargs=True)
    transformer.register_forward_hook(post, with_kwargs=True)


def module_bytes(m: nn.Module) -> int:
    return sum(t.numel() * t.element_size()
               for t in list(m.parameters()) + list(m.buffers()))


class KernelLinear(nn.Module):
    """Padded SVDQW4A4Linear behind an input-pad / output-slice shim."""

    def __init__(self, lin, ic, oc):
        super().__init__()
        self.lin, self.ic, self.oc = lin, ic, oc
        self.ic_p = lin.in_features

    def forward(self, x):
        if self.ic_p > self.ic:
            x = F.pad(x, (0, self.ic_p - self.ic))
        return self.lin(x)[..., : self.oc]


class KernelConv1x1(nn.Module):
    """1x1 Conv2d as a channel-dim linear on the padded kernel."""

    def __init__(self, lin, ic, oc):
        super().__init__()
        self.lin, self.ic, self.oc = lin, ic, oc
        self.ic_p = lin.in_features

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.ic_p > c:
            y = F.pad(y, (0, self.ic_p - c))
        y = self.lin(y)[..., : self.oc]
        return y.reshape(b, h, w, self.oc).permute(0, 3, 1, 2)


def inject_kernel(transformer, packed_dict, torch_dtype=torch.bfloat16):
    from nunchaku.models.linear import SVDQW4A4Linear

    from absorb_basis.build_checkpoint import pack_perm_vector

    n_replaced = 0
    for wpath, t in packed_dict.items():
        parts = wpath.split(".")
        parent = transformer
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        old = getattr(parent, parts[-1]) if not parts[-1].isdigit() else parent[int(parts[-1])]
        is_conv = isinstance(old, nn.Conv2d)
        assert is_conv or isinstance(old, nn.Linear), wpath
        oc, ic, oc_p, ic_p = t["oc"], t["ic"], t["oc_p"], t["ic_p"]
        has_bias = old.bias is not None
        lin = SVDQW4A4Linear(
            ic_p, oc_p, rank=t["lora_down"].shape[1], bias=has_bias,
            precision="nvfp4", torch_dtype=torch_dtype, device="cuda",
        )
        with torch.no_grad():
            lin.qweight.copy_(t["qweight"].to("cuda"))
            lin.wscales.copy_(
                t["wscales"].to("cuda").view(lin.wscales.shape).to(lin.wscales.dtype))
            lin.smooth_factor.copy_(t["smooth"].to("cuda", torch_dtype))
            lin.smooth_factor_orig.copy_(t["smooth_orig"].to("cuda", torch_dtype))
            lin.proj_down.copy_(t["lora_down"].to("cuda", torch_dtype))
            lin.proj_up.copy_(t["lora_up"].to("cuda", torch_dtype))
            if has_bias:
                if "bias" in t and t["bias"] is not None:
                    lin.bias.copy_(t["bias"].view(-1).to("cuda", torch_dtype))
                else:
                    b = old.bias.detach().view(-1).to("cuda", torch_dtype)
                    b = F.pad(b, (0, oc_p - oc))
                    lin.bias.copy_(b[pack_perm_vector(oc_p).to(b.device)])
        lin.wtscale = float(t["wtscale"])
        mod = (KernelConv1x1 if is_conv else KernelLinear)(lin, ic, oc)
        if parts[-1].isdigit():
            parent[int(parts[-1])] = mod
        else:
            setattr(parent, parts[-1], mod)
        n_replaced += 1
    return n_replaced


if __name__ == "__main__":
    import deepcompressor.app.diffusion.pipeline.config as _plcfg
    _plcfg.replace_up_block_conv_with_concat_conv = lambda model: model

    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--kernel-weights", required=True)
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--stats-out", required=True)
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    config, _, unused_cfgs, _, unknown = DiffusionPtqRunConfig.get_parser().parse_known_args()
    assert not config.quant.is_enabled()

    config.output.lock()
    pipeline = config.pipeline.build()
    dt = pipeline.transformer.dtype
    fp_bytes = module_bytes(pipeline.transformer)
    packed = torch.load(own.kernel_weights, map_location="cpu", weights_only=False)
    n = inject_kernel(pipeline.transformer, packed, torch_dtype=dt)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    kernel_bytes = module_bytes(pipeline.transformer)
    mem = {
        "transformer_16bit_gib": fp_bytes / 2**30,
        "transformer_kernel_gib": kernel_bytes / 2**30,
    }
    print(f"replaced {n} layers with SVDQW4A4Linear (dtype={dt})")
    print("transformer memory:", json.dumps(mem))

    attach_probes(pipeline.transformer)
    config.eval.gen_root = own.gen_root
    config.eval.generate(pipeline, task=config.pipeline.task)
    config.output.unlock()

    lat = sorted(STATS["lat_ms"])
    n_f = len(lat)
    summary = {
        "transformer_mem": mem,
        "num_forwards": n_f,
        "lat_ms_median": lat[n_f // 2] if n_f else None,
        "lat_ms_mean": (sum(lat) / n_f) if n_f else None,
        "entry_alloc_gib_max": max(STATS["entry_alloc"]) / 2**30 if n_f else None,
        "peak_alloc_gib_max": max(STATS["peak_alloc"]) / 2**30 if n_f else None,
    }
    print("SANA_KERNEL_STATS:", json.dumps(summary, indent=2))
    with open(own.stats_out, "w") as f:
        json.dump({"summary": summary, "lat_ms": STATS["lat_ms"]}, f)
