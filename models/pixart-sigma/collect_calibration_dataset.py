"""
collect_calib.py

Collect calibration cache for PixArt-Sigma by running FP16 inference on
128 prompts and recording the transformer's inputs at every denoising step.

Taken from the SVDQuant repository:
  https://github.com/mit-han-lab/deepcompressor
  deepcompressor/app/diffusion/dataset/collect/calib.py
  deepcompressor/app/diffusion/dataset/collect/utils.py

Output structure:
  <output_dir>/
    caches/   {prompt_id}-{repeat}-{step:05d}-{guidance}.pt   (5120 files)
    samples/  {prompt_id}-{repeat}.png                         (128 files)

Each .pt cache file contains:
  {
    'input_args':   [latent_tensor],          # [1, 4, 128, 128] fp16
    'input_kwargs': {
        'encoder_hidden_states': ...,         # [1, 300, 4096] fp16
        'encoder_attention_mask': ...,        # [1, 300] int64
        'timestep': ...,                      # int64 scalar
        'added_cond_kwargs': {...},
        'return_dict': False,
    },
    'outputs': [noise_pred],                  # [1, 8, 128, 128] fp16
    'filename': '0006-0',
    'step': 3,      # diffusion step index (0 = highest noise)
    'guidance': 1,  # 0 = unconditional, 1 = conditional
  }

Usage:
    python collect_calib.py \\
        --model-id PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \\
        --prompts  /path/to/qdiff.yaml \\
        --output   /path/to/output_dir \\
        --num-samples 128 \\
        --num-steps 20 \\
        --guidance-scale 4.5
"""

import argparse
import inspect
import os
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from diffusers import PixArtSigmaPipeline
from diffusers.models.transformers import PixArtTransformer2DModel
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


def _tree_split(obj):
    """Split a batch-1 dict of tensors into a list of per-sample dicts."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        split_vals = [_tree_split(obj[k]) for k in keys]
        n = max(len(v) for v in split_vals)
        result = []
        for i in range(n):
            # Use last element for values with fewer splits (e.g. broadcast timestep)
            result.append({k: split_vals[j][min(i, len(split_vals[j]) - 1)] for j, k in enumerate(keys)})
        return result
    elif isinstance(obj, (list, tuple)):
        split_items = [_tree_split(x) for x in obj]
        n = max(len(x) for x in split_items)
        result = []
        for i in range(n):
            result.append(type(obj)(x[min(i, len(x) - 1)] for x in split_items))
        return result
    elif isinstance(obj, torch.Tensor):
        return [obj[i : i + 1] for i in range(obj.shape[0])]
    else:
        return [obj]


# ---------------------------------------------------------------------------
# Taken from deepcompressor/app/diffusion/dataset/collect/utils.py
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
        # Promote positional args that were bound by name into kwargs
        args_to_kwargs = {k: v for k, v in arguments.items() if k not in input_kwargs}
        input_kwargs = dict(input_kwargs)
        input_kwargs.update(args_to_kwargs)

        new_args = []
        if isinstance(module, PixArtTransformer2DModel):
            new_args.append(input_kwargs.pop("hidden_states"))
        else:
            raise ValueError(f"Unsupported model type: {type(module)}")

        cache = _tree_map(
            lambda x: x.cpu(),
            {"input_args": new_args, "input_kwargs": input_kwargs, "outputs": output},
        )
        self.caches.extend(_tree_split(cache))


# ---------------------------------------------------------------------------
# Taken from deepcompressor/app/diffusion/dataset/collect/calib.py
# ---------------------------------------------------------------------------

def _process(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through numpy to detach any non-contiguous storage."""
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def collect_calib_cache(
    pipeline: PixArtSigmaPipeline,
    prompts: list[str],
    filenames: list[str],
    output_dir: str,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
    batch_size: int = 1,
    image_size: int = 1024,
) -> None:
    """
    Run inference on `prompts` and save per-step transformer inputs to disk.

    Args:
        pipeline:       Loaded PixArtSigmaPipeline (fp16, on CUDA).
        prompts:        List of text prompts.
        filenames:      Corresponding filename stems (e.g. "0006-0").
        output_dir:     Root directory; caches/ and samples/ sub-dirs are created.
        num_steps:      Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        batch_size:     Number of prompts to process at once (default 1).
        image_size:     Output image resolution.
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

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Prompts",
                            total=total_batches, dynamic_ncols=True):
        batch_prompts   = prompts[batch_start : batch_start + batch_size]
        batch_filenames = filenames[batch_start : batch_start + batch_size]
        bs = len(batch_prompts)

        seeds      = [hash_str_to_int(name) for name in batch_filenames]
        generators = [
            torch.Generator(device=pipeline.device).manual_seed(seed)
            for seed in seeds
        ]

        result_images = pipeline(
            batch_prompts,
            generator=generators,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=image_size,
            width=image_size,
        ).images

        num_guidances = (len(caches) // bs) // num_steps
        assert len(caches) == bs * num_steps * num_guidances, (
            f"Unexpected cache count: {len(caches)} != {bs} * {num_steps} * {num_guidances}"
        )

        for j, (filename, image) in enumerate(zip(batch_filenames, result_images)):
            image.save(os.path.join(samples_dir, f"{filename}.png"))
            for s in range(num_steps):
                for g in range(num_guidances):
                    c = caches[s * bs * num_guidances + g * bs + j]
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

    Supports two formats:
      - Dict mapping id → prompt text  (e.g. calib_prompts.yaml: {'0006': 'Two sinks...'})
      - List of strings or dicts with a 'prompt' key  (e.g. qdiff.yaml)
    """
    with open(prompt_path) as f:
        data = yaml.safe_load(f)

    prompts, filenames = [], []

    if isinstance(data, dict):
        # {prompt_id: prompt_text} — preserve original IDs as filename stems
        for key, value in list(data.items())[:num_samples]:
            prompts.append(str(value).strip())
            filenames.append(f"{key}-0")
    else:
        # list of strings or dicts
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
    parser = argparse.ArgumentParser(description="Collect PixArt-Sigma calibration cache")
    parser.add_argument("--model-id",       default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--prompts",        required=True,
                        help="Path to prompts YAML file (e.g. qdiff.yaml)")
    parser.add_argument("--output",         required=True,
                        help="Output directory for caches/ and samples/")
    parser.add_argument("--num-samples",    type=int, default=128)
    parser.add_argument("--prompt-id",      type=str, default=None,
                        help="Regenerate only this prompt ID (e.g. '0960')")
    parser.add_argument("--num-steps",      type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--batch-size",     type=int, default=1)
    parser.add_argument("--image-size",     type=int, default=1024)
    args = parser.parse_args()

    prompts, filenames = load_prompts(args.prompts, args.num_samples)
    if args.prompt_id:
        pairs = [(p, f) for p, f in zip(prompts, filenames) if f.startswith(args.prompt_id + "-")]
        if not pairs:
            raise ValueError(f"Prompt ID '{args.prompt_id}' not found in {args.prompts}")
        prompts, filenames = zip(*pairs)
        prompts, filenames = list(prompts), list(filenames)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    print(f"Loading {args.model_id} in fp16...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, use_safetensors=True
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
    )

    expected = len(prompts) * args.num_steps * 2
    actual   = len(list(Path(args.output, "caches").glob("*.pt")))
    print(f"Done. Saved {actual}/{expected} cache files to {args.output}/caches/")
