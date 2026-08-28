"""Generate MJHQ images for PixArt-Sigma with DiRotQ-absorb-basis running on
the REAL nunchaku fp4 kernel (SVDQW4A4Linear: fused act-quant + rank-32 lora
+ W4A4 GEMM), using deepcompressor's own pipeline factory and
DiffusionEvalConfig.generate (identical protocol/seeds to all prior runs).

Also records transformer-only memory and per-forward latency with the same
probes as the FLUX runs (run_nvfp4_nunchaku.py).

Run in the svdquant env from deepcompressor/examples/diffusion:
  python run_pixart_kernel_generate.py configs/model/pixart-sigma.yaml \
      --kernel-weights <kernel.pt> --gen-root <dir> --stats-out <json> \
      --eval-benchmarks MJHQ --eval-num-samples 500 --eval-num-gpus 1 \
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


def inject_kernel(transformer, packed_dict):
    from nunchaku.models.linear import SVDQW4A4Linear

    from absorb_basis.build_checkpoint import pack_perm_vector

    n_replaced = 0
    for wpath, t in packed_dict.items():
        parts = wpath.split(".")
        parent = transformer
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        old = getattr(parent, parts[-1]) if not parts[-1].isdigit() else parent[int(parts[-1])]
        assert isinstance(old, nn.Linear), wpath
        ic, oc = old.in_features, old.out_features
        lin = SVDQW4A4Linear(
            ic, oc, rank=t["lora_down"].shape[1], bias=old.bias is not None,
            precision="nvfp4", torch_dtype=torch.float16, device="cuda",
        )
        with torch.no_grad():
            lin.qweight.copy_(t["qweight"].to("cuda"))
            lin.wscales.copy_(
                t["wscales"].to("cuda").view(lin.wscales.shape).to(lin.wscales.dtype))
            lin.smooth_factor.copy_(t["smooth"].to("cuda", torch.float16))
            lin.smooth_factor_orig.copy_(t["smooth_orig"].to("cuda", torch.float16))
            lin.proj_down.copy_(t["lora_down"].to("cuda", torch.float16))
            lin.proj_up.copy_(t["lora_up"].to("cuda", torch.float16))
            if old.bias is not None:
                # the kernel epilogue reads the bias in nunchaku's packed
                # (pack_scale) channel order. A checkpoint may carry an
                # already-packed bias (e.g. converted SVDQuant, where the
                # shift-folded bias lives in the dict); otherwise permute
                # the fp16 model's bias.
                if "bias" in t and t["bias"] is not None:
                    lin.bias.copy_(t["bias"].view(-1).to("cuda", torch.float16))
                else:
                    b = old.bias.detach().to("cuda", torch.float16)
                    lin.bias.copy_(b[pack_perm_vector(oc).to(b.device)])
        lin.wtscale = float(t["wtscale"])
        if parts[-1].isdigit():
            parent[int(parts[-1])] = lin
        else:
            setattr(parent, parts[-1], lin)
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
    fp16_bytes = module_bytes(pipeline.transformer)
    packed = torch.load(own.kernel_weights, map_location="cpu", weights_only=False)
    n = inject_kernel(pipeline.transformer, packed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    kernel_bytes = module_bytes(pipeline.transformer)
    mem = {
        "transformer_fp16_gib": fp16_bytes / 2**30,
        "transformer_kernel_gib": kernel_bytes / 2**30,
    }
    print(f"replaced {n} linears with SVDQW4A4Linear")
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
        "forward_act_overhead_gib_max": (
            max(p - e for p, e in zip(STATS["peak_alloc"], STATS["entry_alloc"])) / 2**30
            if n_f else None
        ),
    }
    print("PIXART_KERNEL_STATS:", json.dumps(summary, indent=2))
    with open(own.stats_out, "w") as f:
        json.dump({"summary": summary, "lat_ms": STATS["lat_ms"],
                   "entry_alloc": STATS["entry_alloc"],
                   "peak_alloc": STATS["peak_alloc"]}, f)
