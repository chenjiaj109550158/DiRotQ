"""
optimize_sign_flips.py - Optimize Hadamard sign flips for ff.net.2 layers.

Uses calibration data to find sign flips that minimize post-Hadamard quantization
error via evolutionary search (mutation + selection).

Memory-safe: processes blocks in groups of 7, simple truncation for token cap.
GPU-efficient: all optimization on GPU, no per-iteration .item() syncs.
"""

import os
import sys
import gc
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import yaml
import importlib

_ROOT = Path(__file__).resolve().parents[2]  # DiRotQ root
sys.path.insert(0, str(_ROOT))
from utils.hadamard_utils import fast_hadamard_transform, generate_sign_flips


def load_model_config(model_name):
    cfg_path = _ROOT / "models" / model_name / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def collect_ffn_down_inputs_for_blocks(transformer, calib_dir, block_indices,
                                        num_files=128, batch_size=4,
                                        max_tokens_per_block=20000, device="cuda"):
    """Collect ff.net.2 inputs for a subset of blocks. Simple append + truncate."""
    hooks = []
    raw_inputs = {}  # bidx -> list of tensors
    token_counts = {}  # bidx -> total collected

    for name, module in transformer.named_modules():
        parts = name.split(".")
        if "transformer_blocks" not in parts:
            continue
        suffix_start = parts.index("transformer_blocks") + 2
        suffix = ".".join(parts[suffix_start:])
        if suffix != "ff.net.2" or not isinstance(module, nn.Linear):
            continue
        block_idx = int(parts[parts.index("transformer_blocks") + 1])
        if block_idx not in block_indices:
            continue

        raw_inputs[block_idx] = []
        token_counts[block_idx] = 0

        def make_hook(bidx, dim):
            def hook_fn(mod, inp, out):
                if token_counts[bidx] >= max_tokens_per_block:
                    return  # already have enough, skip
                x = inp[0].detach().float().reshape(-1, dim).cpu()
                raw_inputs[bidx].append(x)
                token_counts[bidx] += x.shape[0]
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(block_idx, module.in_features)))

    print(f"  Hooks on blocks: {sorted(block_indices)} ({len(hooks)} hooks)")

    calib_files = sorted(Path(calib_dir).glob("*.pt"))[:num_files]
    print(f"  Running {len(calib_files)} calibration files (batch_size={batch_size})...")

    transformer.eval()
    batch_args = []
    batch_kwargs = {}

    def run_batch():
        if not batch_args:
            return
        latents = torch.cat([a[0] for a in batch_args], dim=0).to(device)
        kw = {}
        for k in batch_kwargs:
            vals = batch_kwargs[k]
            if isinstance(vals[0], torch.Tensor):
                kw[k] = torch.cat(vals, dim=0).to(device)
            else:
                kw[k] = vals[0]
        with torch.no_grad():
            transformer(latents, **kw)
        batch_args.clear()
        for k in batch_kwargs:
            batch_kwargs[k].clear()

    for f in tqdm(calib_files, desc="  Calibration"):
        # Early stop if all blocks have enough
        if all(token_counts[b] >= max_tokens_per_block for b in block_indices):
            break
        data = torch.load(f, map_location="cpu", weights_only=False)
        batch_args.append([a.half() for a in data["input_args"]])
        for k, v in data["input_kwargs"].items():
            if k not in batch_kwargs:
                batch_kwargs[k] = []
            batch_kwargs[k].append(v.half() if isinstance(v, torch.Tensor) else v)
        if len(batch_args) >= batch_size:
            run_batch()
    run_batch()

    for h in hooks:
        h.remove()

    # Concatenate and truncate
    result = {}
    for bidx in sorted(raw_inputs.keys()):
        all_tokens = torch.cat(raw_inputs[bidx], dim=0)
        if all_tokens.shape[0] > max_tokens_per_block:
            all_tokens = all_tokens[:max_tokens_per_block]
        result[bidx] = all_tokens
        print(f"    Block {bidx}: {all_tokens.shape[0]} tokens")

    del raw_inputs
    gc.collect()
    return result


@torch.no_grad()
def quant_mse_gpu(x, bits=4):
    """Per-token asymmetric INT4 quantization MSE. Returns scalar tensor on GPU."""
    maxq = 2 ** bits - 1
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-5) / maxq
    zero = torch.round(-xmin / scale)
    x_q = scale * (torch.clamp(torch.round(x / scale) + zero, 0, maxq) - zero)
    return (x - x_q).pow(2).mean()  # stays on GPU, no .item()


@torch.no_grad()
def optimize_signs_evolutionary(activations, low_dim=4096, bits=4, block_idx=0,
                                 num_generations=3000, max_gpu_tokens=50000,
                                 device="cuda"):
    """Optimize sign flips via evolutionary search, fully on GPU."""
    x_all = activations[:, :low_dim]
    # Subsample for GPU if too many tokens
    if x_all.shape[0] > max_gpu_tokens:
        perm = torch.randperm(x_all.shape[0])[:max_gpu_tokens]
        x_all = x_all[perm]
    x_low = x_all.to(device)  # [N, 4096]

    # Baseline: random sign flips
    baseline_signs = generate_sign_flips(low_dim, seed=42 + block_idx).to(device)
    base_mse = quant_mse_gpu(fast_hadamard_transform(x_low * baseline_signs), bits)
    nosf_mse = quant_mse_gpu(fast_hadamard_transform(x_low), bits)
    print(f"  Baselines: random_signs={base_mse.item():.8f}, no_signs={nosf_mse.item():.8f}")

    current = baseline_signs.clone()
    current_mse = base_mse.clone()
    best_mse = base_mse.clone()
    best_signs = baseline_signs.clone()

    for gen in range(num_generations):
        K = max(1, int(50 * (1 - gen / num_generations) + 1))
        candidate = current.clone()
        flip_idx = torch.randint(0, low_dim, (K,), device=device)
        candidate[flip_idx] *= -1

        cand_mse = quant_mse_gpu(fast_hadamard_transform(x_low * candidate), bits)

        # All comparisons on GPU tensors — no .item() sync
        if cand_mse < current_mse:
            current = candidate
            current_mse = cand_mse
            if current_mse < best_mse:
                best_mse = current_mse.clone()
                best_signs = current.clone()

        if (gen + 1) % 500 == 0:
            # Only sync for logging
            improvement = ((base_mse - best_mse) / base_mse * 100).item()
            print(f"    gen {gen+1}: best={best_mse.item():.8f} ({improvement:+.2f}%), K={K}")

    improvement = ((base_mse - best_mse) / base_mse * 100).item()
    n_flipped = (best_signs != baseline_signs).sum().item()
    print(f"  -> Improvement: {improvement:+.2f}% MSE, "
          f"{n_flipped}/{low_dim} signs flipped ({n_flipped/low_dim*100:.0f}%)")
    return best_signs.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize Hadamard sign flips for ff.net.2 layers")
    parser.add_argument("--model", default="pixart-sigma")
    parser.add_argument("--num-calib-files", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--num-generations", type=int, default=3000)
    parser.add_argument("--blocks-per-group", type=int, default=7)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    model_id = cfg["model_id"]
    calib_dir = str(_ROOT / cfg["calib"]["cache_dir"])
    intermediate_dim = cfg["dims"]["intermediate"]  # 4608
    low_dim = 1 << (intermediate_dim.bit_length() - 1)  # 4096
    num_blocks = cfg["dims"]["num_layers"]  # 28
    a_bits = cfg["quantization"]["a_bits"]  # 4

    if args.output is None:
        args.output = str(_ROOT / "models" / args.model / "basis" /
                          "optimized_sign_flips.pt")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from partial results
    optimized_flips = {}
    if output_path.exists():
        optimized_flips = torch.load(output_path, map_location="cpu", weights_only=False)
        print(f"Resumed: blocks {sorted(optimized_flips.keys())} already done.")

    blocks_todo = [b for b in range(num_blocks) if b not in optimized_flips]
    if not blocks_todo:
        print("All blocks already optimized. Done.")
        sys.exit(0)
    print(f"Blocks to optimize: {blocks_todo}")

    # Load model
    pipeline_cls_path = cfg["pipeline_class"]
    mod_name, cls_name = pipeline_cls_path.rsplit(".", 1)
    PipelineClass = getattr(importlib.import_module(mod_name), cls_name)

    print(f"Loading pipeline ({pipeline_cls_path}) in fp16...")
    pipe = PipelineClass.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe.transformer = pipe.transformer.to("cuda")

    # Process blocks in groups
    bpg = args.blocks_per_group
    for group_start in range(0, len(blocks_todo), bpg):
        group = blocks_todo[group_start:group_start + bpg]
        print(f"\n{'='*60}")
        print(f"Group: blocks {group}")
        print(f"{'='*60}")

        block_inputs = collect_ffn_down_inputs_for_blocks(
            pipe.transformer, calib_dir, block_indices=set(group),
            num_files=args.num_calib_files,
            max_tokens_per_block=args.max_tokens,
        )

        for bidx in group:
            if bidx not in block_inputs:
                print(f"\nBlock {bidx}: no activations collected, skipping.")
                continue
            print(f"\nBlock {bidx}/{num_blocks - 1}:")
            flips = optimize_signs_evolutionary(
                block_inputs[bidx],
                low_dim=low_dim,
                bits=a_bits,
                block_idx=bidx,
                num_generations=args.num_generations,
            )
            optimized_flips[bidx] = flips
            torch.save(optimized_flips, output_path)
            print(f"  Saved ({len(optimized_flips)}/{num_blocks} blocks done)")

        del block_inputs
        gc.collect()
        torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nDone! Optimized sign flips for {len(optimized_flips)} blocks "
          f"saved to {output_path}")
