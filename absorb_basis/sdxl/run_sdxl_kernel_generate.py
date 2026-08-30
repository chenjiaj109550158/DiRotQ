"""Generate images for SDXL-Turbo with DiRotQ-absorb-basis: transformer
linears on the REAL nunchaku fp4 kernel (SVDQW4A4Linear), resnet convs as
fp16 convs executing the fused dequantized-grid weights (mirroring
nunchaku's own SDXL deployment semantics). deepcompressor eval protocol.

Run in the svdquant env from deepcompressor/examples/diffusion:
  python run_sdxl_kernel_generate.py configs/model/sdxl-turbo.yaml \
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

STATS = {"lat_ms": [], "entry_alloc": [], "peak_alloc": []}


def attach_probes(unet):
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

    unet.register_forward_pre_hook(pre, with_kwargs=True)
    unet.register_forward_hook(post, with_kwargs=True)


def module_bytes(m: nn.Module) -> int:
    return sum(t.numel() * t.element_size()
               for t in list(m.parameters()) + list(m.buffers()))


def inject_kernel(unet, packed_dict, torch_dtype=torch.float16):
    from nunchaku.models.linear import SVDQW4A4Linear

    from absorb_basis.build_checkpoint import pack_perm_vector

    n_lin = n_conv = 0
    for wpath, t in packed_dict.items():
        parts = wpath.split(".")
        parent = unet
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        old = getattr(parent, parts[-1]) if not parts[-1].isdigit() else parent[int(parts[-1])]
        if t["type"] == "conv":
            assert isinstance(old, nn.Conv2d), wpath
            with torch.no_grad():
                old.weight.copy_(t["weight"].to(old.weight.device, old.weight.dtype))
            n_conv += 1
            continue
        assert isinstance(old, nn.Linear), wpath
        ic, oc = old.in_features, old.out_features
        lin = SVDQW4A4Linear(
            ic, oc, rank=t["lora_down"].shape[1], bias=old.bias is not None,
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
            if "wcscales" in t and t["wcscales"] is not None:
                lin.wcscales.copy_(t["wcscales"].view(-1).to("cuda", lin.wcscales.dtype))
            if old.bias is not None:
                if "bias" in t and t["bias"] is not None:
                    lin.bias.copy_(t["bias"].view(-1).to("cuda", torch_dtype))
                else:
                    b = old.bias.detach().to("cuda", torch_dtype)
                    lin.bias.copy_(b[pack_perm_vector(oc).to(b.device)])
        lin.wtscale = float(t["wtscale"])
        if parts[-1].isdigit():
            parent[int(parts[-1])] = lin
        else:
            setattr(parent, parts[-1], lin)
        n_lin += 1
    return n_lin, n_conv


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
    unet = pipeline.unet
    dt = unet.dtype
    fp_bytes = module_bytes(unet)
    packed = torch.load(own.kernel_weights, map_location="cpu", weights_only=False)
    n_lin, n_conv = inject_kernel(unet, packed, torch_dtype=dt)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem = {
        "unet_16bit_gib": fp_bytes / 2**30,
        "unet_kernel_gib": module_bytes(unet) / 2**30,
    }
    print(f"replaced {n_lin} linears (SVDQW4A4Linear) + {n_conv} convs (fused fp16)")
    print("unet memory:", json.dumps(mem))

    attach_probes(unet)
    config.eval.gen_root = own.gen_root
    config.eval.generate(pipeline, task=config.pipeline.task)
    config.output.unlock()

    lat = sorted(STATS["lat_ms"])
    n_f = len(lat)
    summary = {
        "unet_mem": mem,
        "num_forwards": n_f,
        "lat_ms_median": lat[n_f // 2] if n_f else None,
        "lat_ms_mean": (sum(lat) / n_f) if n_f else None,
        "peak_alloc_gib_max": max(STATS["peak_alloc"]) / 2**30 if n_f else None,
    }
    print("SDXL_KERNEL_STATS:", json.dumps(summary, indent=2))
    with open(own.stats_out, "w") as f:
        json.dump({"summary": summary, "lat_ms": STATS["lat_ms"]}, f)
