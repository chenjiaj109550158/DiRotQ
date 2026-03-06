"""
apply_dirotq.py

DiRotQ: PCA-based rotation + GPTQ/RTN mixed-precision W4A4 quantization.

Model-specific logic (rotation assignment, quantizer config, generation params)
is loaded from models/<model>/model_utils.py.

Supports:
  --model NAME: model name (subdirectory under models/)
  --gptq: GPTQ weight quantization (default: RTN)
  --max-images N: generate only N images (for testing)
"""

import os
import sys
import json
import subprocess
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from utils.quant_utils import (
    ActQuantWrapper, add_actquant, find_qlayers, rtn_quantize_weights
)
from dirotq_fused_unrotation import fuse_unrotation_into_weights, patch_forward
from utils.gptq_utils import collect_hessians, gptq_quantize_weights

_ROOT = Path(__file__).parent


def load_model_config(model_name: str):
    """Load config from models/<model_name>/config.yaml and return derived constants."""
    cfg_path = _ROOT / "models" / model_name / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg



def hash_str_to_int(s: str) -> int:
    """Deterministic seeding matching deepcompressor's hash_str_to_int."""
    modulus = 10**9 + 7
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def apply_dirotq_to_model(transformer, basis_dict, rotation_dict, cfg, assign_online_rotations):
    """Wrap linear layers with ActQuantWrapper and assign online PCA rotations."""
    skip_layers = cfg["quantization"]["skip_layers"]
    transformer.eval()
    print("Wrapping linear layers with ActQuantWrapper...")
    add_actquant(transformer, skip_names=skip_layers)
    qlayers = find_qlayers(transformer, layers=[ActQuantWrapper])
    print(f"Wrapped {len(qlayers)} layers with ActQuantWrapper.")
    print("Assigning online PCA rotations to quantizers...")
    assign_online_rotations(transformer, basis_dict, rotation_dict, cfg)
    return transformer


def generate_images(pipeline, output_dir, dataset_json, generation_params, max_images=None):
    """Generate images with deterministic seeding, skipping existing ones."""
    with open(dataset_json) as f:
        samples = json.load(f)

    output_dir = Path(output_dir)
    existing = sum(1 for img_id, info in samples.items()
                   if (output_dir / info["category"] / f"{img_id}.png").exists())
    print(f"Found {existing}/{len(samples)} images already generated.")

    to_generate = [(img_id, info) for img_id, info in samples.items()
                   if not (output_dir / info["category"] / f"{img_id}.png").exists()]

    if max_images is not None:
        to_generate = to_generate[:max_images]
        print(f"Limiting to {max_images} images for testing.")

    if not to_generate:
        print("Nothing to generate.")
        return

    print(f"Generating {len(to_generate)} images...")
    pipeline.set_progress_bar_config(disable=True)

    for img_id, info in tqdm(to_generate):
        category = info["category"]
        prompt   = info["prompt"]
        out_path = output_dir / category / f"{img_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        seed = hash_str_to_int(img_id)
        generator = torch.Generator().manual_seed(seed)

        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                generator=generator,
                **generation_params,
            ).images[0]

        image.save(out_path)

    print(f"Done. Images saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pixart-sigma",
                        help="Model name (subdirectory under models/, default: pixart-sigma)")
    parser.add_argument("--dataset", default=str(_ROOT / "datasets" / "mjhq_5000_samples.json"),
                        help="Path to dataset JSON (default: datasets/mjhq_5000_samples.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: models/pixart-sigma/generated_images_gptq or models/pixart-sigma/generated_images_rtn)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Generate only N images (for testing)")
    parser.add_argument("--generate", action="store_true", default=True)
    parser.add_argument("--no-generate", action="store_false", dest="generate")
    parser.add_argument("--gptq", action="store_true", default=False,
                        help="Use GPTQ weight quantization instead of RTN")
    parser.add_argument("--gptq-calib-files", type=int, default=5120) # 128 samples x 20 steps
    parser.add_argument("--gptq-batch-size", type=int, default=1)
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--gptq-damp-pct", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42)")
    parser.add_argument("--quantized-cache", default=None,
                        help="Path to save/load quantized transformer weights. "
                             "If file exists, skips GPTQ/RTN and loads from cache.")
    args = parser.parse_args()

    cfg = load_model_config(args.model)

    _spec = importlib.util.spec_from_file_location(
        "model_utils", _ROOT / "models" / args.model / "model_utils.py")
    model_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(model_utils)
    assign_online_rotations = model_utils.assign_online_rotations
    configure_quantizers_by_name = model_utils.configure_quantizers_by_name
    generation_params = model_utils.generation_params

    model_id      = cfg["model_id"]
    basis_path    = str(_ROOT / cfg["basis"]["output_path"])
    rotation_path = str(_ROOT / cfg["rotation"]["output_path"])
    calib_dir     = str(_ROOT / cfg["calib"]["cache_dir"])
    w_bits        = cfg["quantization"]["w_bits"]
    w_groupsize   = cfg["quantization"]["w_groupsize"]
    skip_layers   = cfg["quantization"]["skip_layers"]

    if args.quantized_cache is None:
        cache_name = "gptq_model.pt" if args.gptq else "rtn_model.pt"
        args.quantized_cache = str(_ROOT / "models" / args.model / "quantized_cache" / cache_name)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.output_dir is None: 
        args.output_dir = str(_ROOT / "models" / args.model / ("generated_images_gptq" if args.gptq else "generated_images_rtn"))

    if not os.path.exists(rotation_path):
        print(f"Rotation file not found at {rotation_path}, generating...")
        subprocess.run([sys.executable, str(_ROOT / "gen_rotation.py"), "--model", args.model], check=True)

    if not os.path.exists(basis_path):
        print(f"PCA basis not found at {basis_path}, generating...")
        subprocess.run([sys.executable, str(_ROOT / "get_basis.py"), "--model", args.model], check=True)

    print(f"Loading PCA basis from {basis_path}...")
    basis_dict = torch.load(basis_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(basis_dict)} basis matrices.")

    print(f"Loading rotations from {rotation_path}...")
    rotation_dict = torch.load(rotation_path, map_location="cpu", weights_only=False)
    print("Rotations:", {k: v.shape if isinstance(v, torch.Tensor) else v
                         for k, v in rotation_dict.items()})

    pipeline_cls_path = cfg["pipeline_class"]
    mod_name, cls_name = pipeline_cls_path.rsplit(".", 1)
    PipelineClass = getattr(importlib.import_module(mod_name), cls_name)

    print(f"Loading pipeline ({pipeline_cls_path}) in fp16...")
    pipe = PipelineClass.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    )

    apply_dirotq_to_model(pipe.transformer, basis_dict, rotation_dict, cfg, assign_online_rotations)

    high_len_hidden = rotation_dict["high_len_hidden"]
    high_len_head   = rotation_dict["high_len_head"]

    configure_quantizers_by_name(pipe.transformer, high_len_hidden, high_len_head, cfg)

    cache_path = Path(args.quantized_cache)
    if cache_path.exists():
        print(f"Loading quantized weights from cache: {cache_path}")
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        pipe.transformer.load_state_dict(state, strict=False)
        for _, mod in pipe.transformer.named_modules():
            if isinstance(mod, ActQuantWrapper) and (mod.rotation is not None or mod.rotation_per_head is not None):
                mod._unrot_fused = True
    else:
        if args.gptq:
            print("Moving transformer to CUDA for GPTQ calibration...")
            pipe.transformer = pipe.transformer.to("cuda")

            print(f"Collecting GPTQ Hessians from {args.gptq_calib_files} calibration files...")
            hessians = collect_hessians(
                pipe.transformer,
                calib_dir=calib_dir,
                device="cuda",
                num_calib_files=args.gptq_calib_files,
                batch_size=args.gptq_batch_size,
            )

            print("Applying GPTQ weight quantization (W4, group_size=64)...")
            gptq_quantize_weights(
                pipe.transformer, hessians,
                bits=w_bits, groupsize=w_groupsize, sym=True,
                skip_names=skip_layers,
                damp_pct=args.gptq_damp_pct,
                block_size=args.gptq_block_size,
                device="cuda",
            )
            del hessians
        else:
            print("Applying RTN weight quantization (W4, group_size=64)...")
            rtn_quantize_weights(pipe.transformer, bits=w_bits, groupsize=w_groupsize,
                                 sym=True, skip_names=skip_layers)

        print("Fusing unrotation into weights...")
        fuse_unrotation_into_weights(pipe.transformer)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving quantized weights to cache: {cache_path}")
        torch.save(pipe.transformer.state_dict(), cache_path)

    patch_forward()

    if args.generate:
        print("Moving pipeline to CUDA...")
        pipe = pipe.to("cuda")
        generate_images(pipe, args.output_dir, args.dataset, generation_params, max_images=args.max_images)

    print("All done.")
