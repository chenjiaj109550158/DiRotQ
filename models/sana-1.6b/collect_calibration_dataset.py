"""
Collect calibration cache for Sana-1.6B.

Runs BF16 inference on 128 prompts and records the transformer's inputs at every
denoising step.  Adapted from the PixArt-Sigma collector (same structure).

Output:
  <output_dir>/
    caches/   {prompt_id}-{step:05d}-{guidance}.pt   (5120 files = 128 × 20 × 2)
    samples/  {prompt_id}.png                         (128 files)

Each .pt cache file:
  {
    'input_args':   [hidden_states],              # [1, T, 2240] bf16
    'input_kwargs': {
        'encoder_hidden_states': ...,             # [1, T_text, 2240] bf16
        'timestep': ...,                          # [1] int64
        'encoder_attention_mask': ...,
        ...
    },
    'outputs': [...],
    'filename': '0006-0',
    'step': 3,
    'guidance': 1,  # 0=unconditional, 1=conditional
  }

Usage:
    python collect_calibration_dataset.py \\
        --model-id Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers \\
        --prompts  models/sana-1.6b/calib_prompts.yaml \\
        --output   models/sana-1.6b/calibration_dataset
"""

import argparse
import inspect
import os
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from diffusers import SanaPipeline
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel
from tqdm import tqdm


def hash_str_to_int(s: str) -> int:
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
    """Split a batched dict/list of tensors into per-sample entries."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        split_vals = [_tree_split(obj[k]) for k in keys]
        n = max(len(v) for v in split_vals)
        return [
            {k: split_vals[j][min(i, len(split_vals[j]) - 1)] for j, k in enumerate(keys)}
            for i in range(n)
        ]
    elif isinstance(obj, (list, tuple)):
        split_items = [_tree_split(x) for x in obj]
        n = max(len(x) for x in split_items)
        return [
            type(obj)(x[min(i, len(x) - 1)] for x in split_items)
            for i in range(n)
        ]
    elif isinstance(obj, torch.Tensor):
        return [obj[i: i + 1] for i in range(obj.shape[0])]
    else:
        return [obj]


class CollectHook:
    """Forward hook that captures (input_args, input_kwargs, outputs) per transformer call."""

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
        # Promote positional args bound by name into kwargs
        args_to_kwargs = {k: v for k, v in arguments.items() if k not in input_kwargs}
        input_kwargs = dict(input_kwargs)
        input_kwargs.update(args_to_kwargs)

        new_args = []
        if isinstance(module, SanaTransformer2DModel):
            new_args.append(input_kwargs.pop("hidden_states"))
        else:
            raise ValueError(f"Unsupported model type: {type(module)}")

        cache = _tree_map(
            lambda x: x.cpu(),
            {"input_args": new_args, "input_kwargs": input_kwargs, "outputs": output},
        )
        self.caches.extend(_tree_split(cache))


def _process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def collect_calib_cache(
    pipeline: SanaPipeline,
    prompts: list,
    filenames: list,
    output_dir: str,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
    batch_size: int = 1,
    image_size: int = 1024,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    caches_dir  = os.path.join(output_dir, "caches")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(caches_dir,  exist_ok=True)

    caches: list = []
    pipeline.transformer.register_forward_hook(
        CollectHook(caches=caches), with_kwargs=True
    )
    pipeline.set_progress_bar_config(
        desc="Denoising", leave=False, dynamic_ncols=True, position=1
    )

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Prompts",
                            total=total_batches, dynamic_ncols=True):
        batch_prompts   = prompts[batch_start: batch_start + batch_size]
        batch_filenames = filenames[batch_start: batch_start + batch_size]
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


def load_prompts(prompt_path: str, num_samples: int) -> tuple:
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
    parser = argparse.ArgumentParser(description="Collect Sana-1.6B calibration cache")
    parser.add_argument("--model-id",
                        default="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers")
    parser.add_argument("--prompts",        required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--num-samples",    type=int, default=128)
    parser.add_argument("--prompt-id",      type=str, default=None,
                        help="Regenerate only this prompt ID")
    parser.add_argument("--num-steps",      type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--batch-size",     type=int, default=1)
    parser.add_argument("--image-size",     type=int, default=1024)
    args = parser.parse_args()

    prompts, filenames = load_prompts(args.prompts, args.num_samples)
    if args.prompt_id:
        pairs = [(p, f) for p, f in zip(prompts, filenames)
                 if f.startswith(args.prompt_id + "-")]
        if not pairs:
            raise ValueError(f"Prompt ID '{args.prompt_id}' not found in {args.prompts}")
        prompts, filenames = zip(*pairs)
        prompts, filenames = list(prompts), list(filenames)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    print(f"Loading {args.model_id} in bf16...")
    pipe = SanaPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, use_safetensors=True
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

    expected = len(prompts) * args.num_steps * 2   # 2 guidances (CFG)
    actual   = len(list(Path(args.output, "caches").glob("*.pt")))
    print(f"Done. Saved {actual}/{expected} cache files to {args.output}/caches/")
