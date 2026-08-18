"""
get_basis.py

Compute PCA basis from calibration activations for a given model.
If calibration caches don't exist, auto-runs the collection script.

Usage:
    python get_basis.py --model pixart-sigma
    python get_basis.py --model pixart-sigma --max-files 256
"""

import os
import sys
import glob
import argparse
import importlib
import importlib.util
import subprocess
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).parent

sys.path.insert(0, os.path.dirname(__file__))


def load_model_config(model_name: str) -> dict:
    config_path = _ROOT / "models" / model_name / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for model '{model_name}' at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_collect_fn(model_name: str):
    """Import collect_basis from models/<model_name>/basis_utils.py."""
    module_path = f"models.{model_name.replace('-', '_')}.basis_utils"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        spec_path = _ROOT / "models" / model_name / "basis_utils.py"
        spec = importlib.util.spec_from_file_location("basis_utils", spec_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod.collect_basis


def collect_calibration_caches(cfg, cache_dir):
    """Auto-run the calibration collection script if no caches exist."""
    calib_cfg = cfg.get("calib", {})
    collect_script = calib_cfg.get("collect_script")
    prompts_path = calib_cfg.get("prompts")
    calib_output_dir = calib_cfg.get("output_dir")

    if not collect_script or not prompts_path or not calib_output_dir:
        raise FileNotFoundError(
            f"No calibration caches in {cache_dir} and config is missing "
            "calib.collect_script, calib.prompts, or calib.output_dir"
        )

    cmd = [
        sys.executable, str(_ROOT / collect_script),
        "--model-id", cfg["model_id"],
        "--prompts", str(_ROOT / prompts_path),
        "--output", str(_ROOT / calib_output_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not cache_files:
        raise RuntimeError(f"Calibration collection finished but no .pt files found in {cache_dir}")
    return cache_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model name (must match a folder under models/)")
    parser.add_argument(
        "--model-id", default=None,
        help=("Override the configured Hugging Face ID with an immutable local "
              "snapshot or alternate model ID."),
    )
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of calibration cache files (for testing)")
    parser.add_argument("--output", default=None,
                        help="Override basis output path from config")
    parser.add_argument("--cache-dir", default=None,
                        help="Override calibration cache dir from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Forward-pass batch size for basis collection (default: model-specific)")
    parser.add_argument(
        "--exclude-down", action="store_true",
        help=("Do not collect FFN-down PCA sources. This is the FLUX speed-path "
              "no-rotation-ffdown contract, not a generic approximation."),
    )
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    collect_basis = load_model_collect_fn(args.model)

    basis_path = args.output or cfg["basis"]["output_path"]
    cache_dir = args.cache_dir or cfg["basis"]["cache_dir"]

    if os.path.exists(basis_path):
        print(f"Basis already exists at {basis_path}")
        print("Delete the file to recompute.")
    else:
        cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

        if not cache_files:
            print(f"No calibration caches found in {cache_dir}, generating them...")
            cache_files = collect_calibration_caches(cfg, cache_dir)

        if args.max_files is not None:
            cache_files = cache_files[:args.max_files]
        print(f"Found {len(cache_files)} calibration files in {cache_dir}")

        pipeline_cls_path = cfg["pipeline_class"]
        module_name, cls_name = pipeline_cls_path.rsplit(".", 1)
        PipelineClass = getattr(importlib.import_module(module_name), cls_name)

        _dtype_map = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                      "fp16": torch.float16,  "float16":  torch.float16,
                      "fp32": torch.float32,  "float32":  torch.float32}
        torch_dtype = _dtype_map.get(cfg.get("dtype", "float32"), torch.float32)
        model_id = args.model_id or cfg["model_id"]
        print(f"Loading {model_id} ({pipeline_cls_path}) in {torch_dtype}...")
        pipe = PipelineClass.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
        pipe = pipe.to("cuda")
        pipe.transformer.eval()
        pipe.transformer.requires_grad_(False)

        collect_kwargs = {}
        if args.batch_size is not None:
            collect_kwargs["batch_size"] = args.batch_size
        if args.exclude_down:
            collect_kwargs["include_down"] = False
        basis_dict = collect_basis(pipe.transformer, cache_files, cfg, **collect_kwargs)

        del pipe
        torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(basis_path), exist_ok=True)
        torch.save(basis_dict, basis_path)
        print(f"Saved basis to {basis_path}  ({len(basis_dict)} keys)")

    print("Done.")
