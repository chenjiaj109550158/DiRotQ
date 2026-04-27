"""
diagnose_layer_qsnr.py

Per-layer QSNR diagnostic: run the same 10 MJHQ prompts through both
BF16 baseline and DiRotQ-quantized FLUX, compare transformer block outputs
at each denoising step.

Reports:
  1. Per-block QSNR at each step (which blocks introduce the most error)
  2. Per-step total QSNR (how error accumulates across steps)
  3. Final image PSNR / SSIM

Usage:
    python diagnose_layer_qsnr.py --model flux-schnell --num-images 10
"""

import argparse
import gc
import json
import importlib
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from utils.quant_utils import ActQuantWrapper, add_actquant, find_qlayers


def qsnr_db(ref, test):
    """Compute QSNR in dB: 10*log10(||ref||^2 / ||ref - test||^2)."""
    ref_f = ref.float()
    test_f = test.float()
    signal = (ref_f ** 2).sum()
    noise = ((ref_f - test_f) ** 2).sum()
    if noise == 0:
        return float('inf')
    return 10 * torch.log10(signal / noise).item()


def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def ssim_simple(img1, img2):
    """Simplified SSIM on grayscale."""
    from scipy.ndimage import uniform_filter
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    if img1.ndim == 3:
        img1 = img1.mean(axis=2)
        img2 = img2.mean(axis=2)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = uniform_filter(img1, size=11)
    mu2 = uniform_filter(img2, size=11)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = uniform_filter(img1 ** 2, size=11) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=11) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=11) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return np.mean(num / den)


def hash_str_to_int(s):
    modulus = 10 ** 9 + 7
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) % modulus
    return h


def load_samples(dataset_json, n):
    with open(dataset_json) as f:
        samples = json.load(f)
    return list(samples.items())[:n]


class BlockOutputCapture:
    """Hook that captures transformer block outputs at each denoising step."""

    def __init__(self):
        self.step = 0
        self.outputs = defaultdict(dict)  # {step: {block_name: tensor}}
        self._hooks = []

    def register(self, transformer):
        # Double blocks
        for i, block in enumerate(transformer.transformer_blocks):
            name = f"double.{i}"
            self._hooks.append(
                block.register_forward_hook(self._make_hook(name))
            )
        # Single blocks
        for i, block in enumerate(transformer.single_transformer_blocks):
            name = f"single.{i}"
            self._hooks.append(
                block.register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name):
        def hook(module, inp, out):
            # Double blocks return (encoder_hidden_states, hidden_states)
            # Single blocks return hidden_states
            if isinstance(out, tuple):
                # Take hidden_states (image stream) — more relevant for image quality
                tensor = out[1] if len(out) > 1 else out[0]
            else:
                tensor = out
            self.outputs[self.step][name] = tensor.detach().cpu()
        return hook

    def next_step(self):
        self.step += 1

    def reset(self):
        self.step = 0
        self.outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class StepCounter:
    """Hook on the transformer to count denoising steps."""

    def __init__(self, capture: BlockOutputCapture):
        self.capture = capture
        self.call_count = 0
        self._hook = None

    def register(self, transformer):
        self._hook = transformer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        self.call_count += 1
        self.capture.next_step()

    def remove(self):
        if self._hook:
            self._hook.remove()


def run_pipeline(pipe, samples, generation_params, capture):
    """Generate images and capture per-block outputs at each step."""
    images = {}
    pipe.set_progress_bar_config(disable=True)

    for img_id, info in samples:
        capture.reset()
        seed = hash_str_to_int(img_id)
        gen = torch.Generator().manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                [info["prompt"]],
                generator=[gen],
                **generation_params,
            )

        images[img_id] = {
            "image": np.array(result.images[0]),
            "block_outputs": dict(capture.outputs),  # {step: {block: tensor}}
        }

    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flux-schnell")
    parser.add_argument("--dataset", default=str(_ROOT / "datasets" / "mjhq_5000_samples.json"))
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--quantized-cache", default=None)
    parser.add_argument("--a-bits", type=int, default=8)
    parser.add_argument("--nvfp4", action="store_true", default=False)
    args = parser.parse_args()

    cfg_path = _ROOT / "models" / args.model / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    _spec = importlib.util.spec_from_file_location(
        "model_utils", _ROOT / "models" / args.model / "model_utils.py")
    model_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(model_utils)

    generation_params = model_utils.generation_params
    samples = load_samples(args.dataset, args.num_images)
    print(f"Using {len(samples)} prompts for diagnosis.")

    pipeline_cls_path = cfg["pipeline_class"]
    mod_name, cls_name = pipeline_cls_path.rsplit(".", 1)
    PipelineClass = getattr(importlib.import_module(mod_name), cls_name)
    dtype_str = cfg.get("dtype", "fp16")
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

    # ── Phase 1: BF16 baseline ──────────────────────────────────────────
    print("\n=== Phase 1: BF16 baseline ===")
    print(f"Loading {cfg['model_id']} in {dtype_str}...")
    pipe_fp = PipelineClass.from_pretrained(
        cfg["model_id"], torch_dtype=torch_dtype, use_safetensors=True)
    pipe_fp = pipe_fp.to("cuda")

    capture_fp = BlockOutputCapture()
    capture_fp.register(pipe_fp.transformer)
    step_counter_fp = StepCounter(capture_fp)
    step_counter_fp.register(pipe_fp.transformer)

    print(f"Generating {len(samples)} BF16 images...")
    fp_results = run_pipeline(pipe_fp, samples, generation_params, capture_fp)

    capture_fp.remove()
    step_counter_fp.remove()
    del pipe_fp
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: Quantized model ────────────────────────────────────────
    print("\n=== Phase 2: DiRotQ quantized ===")
    from dirotq_fused_unrotation_fast import patch_forward_fast as patch_forward, preconvert_rotations_to_device

    print(f"Loading {cfg['model_id']} in {dtype_str}...")
    pipe_q = PipelineClass.from_pretrained(
        cfg["model_id"], torch_dtype=torch_dtype, use_safetensors=True)

    preprocess = getattr(model_utils, "preprocess_transformer", None)
    if preprocess:
        preprocess(pipe_q.transformer, cfg)

    if args.nvfp4:
        nvfp4_cfg = cfg.get("nvfp4", {})
        skip_layers = nvfp4_cfg.get("skip_layers", cfg["quantization"]["skip_layers"])
    else:
        skip_layers = cfg["quantization"]["skip_layers"]

    if args.a_bits is not None:
        cfg["quantization"]["a_bits"] = args.a_bits

    basis_path = str(_ROOT / cfg["basis"]["output_path"])
    rotation_path = str(_ROOT / cfg["rotation"]["output_path"])

    basis_dict = torch.load(basis_path, map_location="cpu", weights_only=False)
    rotation_dict = torch.load(rotation_path, map_location="cpu", weights_only=False)

    from apply_dirotq import apply_dirotq_to_model
    apply_dirotq_to_model(pipe_q.transformer, basis_dict, rotation_dict, cfg,
                           model_utils.assign_online_rotations, skip_layers=skip_layers)

    high_len_hidden = rotation_dict["high_len_hidden"]
    high_len_head = rotation_dict["high_len_head"]
    high_len_down = rotation_dict.get("high_len_down", 0)
    model_utils.configure_quantizers_by_name(
        pipe_q.transformer, high_len_hidden, high_len_head, cfg,
        nvfp4=args.nvfp4, high_len_down=high_len_down)

    # Load quantized weights
    if args.quantized_cache is None:
        fmt_tag = "nvfp4" if args.nvfp4 else "int4"
        w_gs = cfg.get("nvfp4", {}).get("w_groupsize", 16) if args.nvfp4 else cfg["quantization"]["w_groupsize"]
        a_tag = f"_a{args.a_bits}" if args.a_bits != 4 else ""
        cache_name = f"{fmt_tag}_g{w_gs}_gptq{a_tag}_model.pt"
        args.quantized_cache = str(_ROOT / "models" / args.model / "quantized_cache" / cache_name)

    print(f"Loading quantized weights from {args.quantized_cache}...")
    state = torch.load(args.quantized_cache, map_location="cpu", weights_only=False)
    pipe_q.transformer.load_state_dict(state, strict=False)
    for _, mod in pipe_q.transformer.named_modules():
        if isinstance(mod, ActQuantWrapper) and (
            mod.rotation is not None or mod.rotation_per_head is not None
            or getattr(mod, 'use_hadamard', False)):
            mod._unrot_fused = True
    del state

    patch_forward()
    pipe_q = pipe_q.to("cuda")
    preconvert_rotations_to_device(pipe_q.transformer, device="cuda")

    capture_q = BlockOutputCapture()
    capture_q.register(pipe_q.transformer)
    step_counter_q = StepCounter(capture_q)
    step_counter_q.register(pipe_q.transformer)

    print(f"Generating {len(samples)} quantized images...")
    q_results = run_pipeline(pipe_q, samples, generation_params, capture_q)

    capture_q.remove()
    step_counter_q.remove()

    # ── Phase 3: Analysis ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Image-level metrics
    print("\n── Image-level PSNR / SSIM ──")
    psnr_vals, ssim_vals = [], []
    for img_id, _ in samples:
        fp_img = fp_results[img_id]["image"]
        q_img = q_results[img_id]["image"]
        p = psnr(fp_img, q_img)
        s = ssim_simple(fp_img, q_img)
        psnr_vals.append(p)
        ssim_vals.append(s)
        print(f"  {img_id[:12]}...: PSNR={p:.2f} dB, SSIM={s:.4f}")
    print(f"  Average: PSNR={np.mean(psnr_vals):.2f} dB, SSIM={np.mean(ssim_vals):.4f}")

    # Per-step transformer output QSNR
    num_steps = len(fp_results[samples[0][0]]["block_outputs"])
    print(f"\n── Per-step QSNR (averaged over {len(samples)} images) ──")

    # Per-block QSNR averaged over images and steps
    block_qsnr = defaultdict(list)  # block_name -> list of QSNR values
    step_qsnr = defaultdict(list)   # step -> list of QSNR values

    for img_id, _ in samples:
        fp_blocks = fp_results[img_id]["block_outputs"]
        q_blocks = q_results[img_id]["block_outputs"]
        for step in sorted(fp_blocks.keys()):
            if step not in q_blocks:
                continue
            for block_name in sorted(fp_blocks[step].keys()):
                if block_name not in q_blocks[step]:
                    continue
                q = qsnr_db(fp_blocks[step][block_name], q_blocks[step][block_name])
                block_qsnr[block_name].append(q)
                step_qsnr[step].append(q)

    for step in sorted(step_qsnr.keys()):
        vals = step_qsnr[step]
        print(f"  Step {step}: mean QSNR={np.mean(vals):.2f} dB, "
              f"min={np.min(vals):.2f} dB")

    # Top-20 worst blocks
    print(f"\n── Worst 20 blocks by QSNR (averaged over images & steps) ──")
    block_avg = {name: np.mean(vals) for name, vals in block_qsnr.items()}
    for name, avg in sorted(block_avg.items(), key=lambda x: x[1])[:20]:
        print(f"  {name:25s}: {avg:.2f} dB")

    # Top-20 best blocks
    print(f"\n── Best 10 blocks by QSNR ──")
    for name, avg in sorted(block_avg.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:25s}: {avg:.2f} dB")

    print("\nDone.")


if __name__ == "__main__":
    main()
