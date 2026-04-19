"""
collect_calibration_dataset.py

Collect calibration cache for FLUX.1-schnell by running bf16 inference on
128 prompts and recording the transformer's inputs at every denoising step.

Adapted from the PixArt-Sigma collector which itself mirrors SVDQuant's
deepcompressor/app/diffusion/dataset/collect/calib.py.

Output structure:
  <output_dir>/
    caches/   {prompt_id}-{repeat}-{step:05d}-{guidance}.pt   (~512 files)
    samples/  {prompt_id}-{repeat}.png                         (128 files)

Each .pt cache file contains:
  {
    'input_args':   [hidden_states],            # [1, L, 64] bf16, packed latent
    'input_kwargs': {
        'encoder_hidden_states': ...,           # [1, 512, 4096] bf16 (T5)
        'pooled_projections':    ...,           # [1, 768]        bf16 (CLIP)
        'timestep':              ...,           # [1]             bf16
        'img_ids':               ...,           # [L, 3]          bf16
        'txt_ids':               ...,           # [512, 3]        bf16
        'guidance':              ...,           # [1] float or None (schnell=None)
        'joint_attention_kwargs': None,
        'return_dict':           False,
    },
    'outputs': [noise_pred],
    'filename': '0006-0',
    'step': 0,       # diffusion step index (0 = highest noise)
    'guidance': 0,   # 0 = only branch (schnell uses CFG=0, no split)
  }

Usage:
    python collect_calibration_dataset.py \\
        --model-id black-forest-labs/FLUX.1-schnell \\
        --prompts  models/flux-schnell/calib_prompts.yaml \\
        --output   models/flux-schnell/calibration_dataset \\
        --num-samples 128 \\
        --num-steps   4 \\
        --guidance-scale 0
"""

import argparse
import inspect
import os
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Taken from deepcompressor/utils/common.py
# ---------------------------------------------------------------------------

def hash_str_to_int(s: str) -> int:
    """Polynomial hash matching deepcompressor/utils/common.py."""
    modulus = 10**9 + 7
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def _tree_map(fn, obj):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    elif isinstance(obj, dict):
        return {k: _tree_map(fn, v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        mapped = [_tree_map(fn, x) for x in obj]
        return type(obj)(mapped)
    else:
        return obj


# ---------------------------------------------------------------------------
# Adapted from deepcompressor/app/diffusion/dataset/collect/utils.py
# ---------------------------------------------------------------------------

class CollectHook:
    """Forward hook that captures (input_args, input_kwargs, outputs) per call."""

    def __init__(self, caches: list) -> None:
        self.caches = caches

    def __call__(
        self,
        module: nn.Module,
        input_args: tuple,
        input_kwargs: dict,
        output: tp.Any,
    ) -> None:
        signature = inspect.signature(module.forward)
        bound = signature.bind(*input_args, **input_kwargs)
        arguments = bound.arguments
        args_to_kwargs = {k: v for k, v in arguments.items() if k not in input_kwargs}
        input_kwargs = dict(input_kwargs)
        input_kwargs.update(args_to_kwargs)

        new_args = []
        if isinstance(module, FluxTransformer2DModel):
            new_args.append(input_kwargs.pop("hidden_states"))
        else:
            raise ValueError(f"Unsupported model type: {type(module)}")

        cache = _tree_map(
            lambda x: x.cpu(),
            {"input_args": new_args, "input_kwargs": input_kwargs, "outputs": output},
        )
        self.caches.append(cache)


# ---------------------------------------------------------------------------
# Calibration driver
# ---------------------------------------------------------------------------

def _process(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through numpy to detach any non-contiguous storage."""
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def collect_calib_cache(
    pipeline: FluxPipeline,
    prompts: list[str],
    filenames: list[str],
    output_dir: str,
    num_steps: int = 4,
    guidance_scale: float = 0.0,
    batch_size: int = 1,
    image_size: int = 1024,
    max_sequence_length: int = 256,
) -> None:
    """
    Run inference on `prompts` and save per-step transformer inputs to disk.

    Notes:
      - FLUX.1-schnell is a 4-step guidance-distilled model (guidance_scale=0).
        At CFG=0 the pipeline runs the conditional branch only, so there is
        exactly one "guidance" branch per step (as opposed to PixArt's 2).
    """
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    samples_dir = os.path.join(output_dir, "samples")
    caches_dir  = os.path.join(output_dir, "caches")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(caches_dir,  exist_ok=True)

    caches: list[dict] = []
    pipeline.transformer.register_forward_hook(
        CollectHook(caches=caches), with_kwargs=True
    )
    pipeline.set_progress_bar_config(
        desc="Denoising", leave=False, dynamic_ncols=True, position=1
    )

    assert batch_size == 1, (
        "FLUX calibration collector only supports batch_size=1 "
        "(img_ids/txt_ids are sequence-length, not per-batch)."
    )

    for batch_start in tqdm(range(0, len(prompts)), desc="Prompts",
                            total=len(prompts), dynamic_ncols=True):
        prompt   = prompts[batch_start]
        filename = filenames[batch_start]

        seed = hash_str_to_int(filename)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        result_images = pipeline(
            [prompt],
            generator=[generator],
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=image_size,
            width=image_size,
            max_sequence_length=max_sequence_length,
        ).images

        # At CFG=0, schnell runs one branch per step, so caches should have
        # exactly num_steps entries after each prompt.
        num_guidances = len(caches) // num_steps
        assert len(caches) == num_steps * num_guidances, (
            f"Unexpected cache count: {len(caches)} != {num_steps} * {num_guidances}"
        )

        result_images[0].save(os.path.join(samples_dir, f"{filename}.png"))
        for s in range(num_steps):
            for g in range(num_guidances):
                c = caches[s * num_guidances + g]
                c["filename"] = filename
                c["step"]     = s
                c["guidance"] = g
                c = _tree_map(lambda x: _process(x), c)
                torch.save(c, os.path.join(caches_dir, f"{filename}-{s:05d}-{g}.pt"))

        caches.clear()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_prompts(prompt_path: str, num_samples: int) -> tuple[list[str], list[str]]:
    """Load prompts from a YAML file. Returns (prompts, filenames).

    Supports:
      - Dict mapping id → prompt text (calib_prompts.yaml)
      - List of strings or {prompt: ...} dicts (qdiff.yaml)
    """
    with open(prompt_path) as f:
        data = yaml.safe_load(f)

    prompts, filenames = [], []

    if isinstance(data, dict):
        for key, value in list(data.items())[:num_samples]:
            prompts.append(str(value).strip())
            filenames.append(f"{key}-0")
    else:
        for i, entry in enumerate(data[:num_samples]):
            if isinstance(entry, str):
                prompt = entry
            elif isinstance(entry, dict):
                prompt = entry.get("prompt", entry.get("text", ""))
            else:
                prompt = str(entry)
            prompts.append(str(prompt).strip())
            filenames.append(f"{i:04d}-0")

    return prompts, filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect FLUX.1-schnell calibration cache")
    parser.add_argument("--model-id",            default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--prompts",             required=True,
                        help="Path to prompts YAML file (e.g. calib_prompts.yaml)")
    parser.add_argument("--output",              required=True,
                        help="Output directory for caches/ and samples/")
    parser.add_argument("--num-samples",         type=int,   default=128)
    parser.add_argument("--prompt-id",           type=str,   default=None,
                        help="Regenerate only this prompt ID (e.g. '0960')")
    parser.add_argument("--num-steps",           type=int,   default=25)
    parser.add_argument("--guidance-scale",      type=float, default=3.5)
    parser.add_argument("--batch-size",          type=int,   default=1)
    parser.add_argument("--image-size",          type=int,   default=1024)
    parser.add_argument("--max-sequence-length", type=int,   default=256)
    args = parser.parse_args()

    prompts, filenames = load_prompts(args.prompts, args.num_samples)
    if args.prompt_id:
        pairs = [(p, f) for p, f in zip(prompts, filenames) if f.startswith(args.prompt_id + "-")]
        if not pairs:
            raise ValueError(f"Prompt ID '{args.prompt_id}' not found in {args.prompts}")
        prompts, filenames = zip(*pairs)
        prompts, filenames = list(prompts), list(filenames)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    print(f"Loading {args.model_id} in bf16...")
    pipe = FluxPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.eval()
    pipe.transformer.requires_grad_(False)

    collect_calib_cache(
        pipeline=pipe,
        prompts=prompts,
        filenames=filenames,
        output_dir=args.output,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_sequence_length=args.max_sequence_length,
    )

    # At guidance_scale=0 the FLUX scheduler runs one branch per step.
    expected = len(prompts) * args.num_steps * 1
    actual   = len(list(Path(args.output, "caches").glob("*.pt")))
    print(f"Done. Saved {actual}/{expected} cache files to {args.output}/caches/")
