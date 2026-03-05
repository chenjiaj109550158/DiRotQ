"""
gen_rotation_v2.py

Generate DiRotQ rotation matrices (v2: R_high = Identity).

Rationale:
  After PCA rotation, the last `high_bits_length` dims already correspond to the
  highest-variance principal components. Applying a random rotation R_high on top
  serves no purpose for quantization (those dims are kept at 16-bit anyway) and
  slightly mixes the carefully ordered principal components. Using R_high = I keeps
  the 16-bit branch exactly in the PCA basis.

  Only the low-precision (4-bit) subspace gets random rotation (R_low unchanged),
  which helps reduce outliers before quantization.

Usage:
    python gen_rotation_v2.py --model pixart-sigma
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from utils.hadamard_utils import random_orthogonal_matrix


def load_model_config(model_name: str) -> dict:
    config_path = Path(__file__).parent / "models" / model_name / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for model '{model_name}' at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_rotation_matrices(cfg: dict, device="cuda", seed=42) -> dict:
    """
    Block-diagonal rotation matrices where only the low-variance subspace
    gets a random orthogonal rotation; the high-variance subspace uses Identity.

    R = block_diag(R_low, I_high)
    """
    torch.manual_seed(seed)

    dims = cfg["dims"]
    high_fraction = cfg["rotation"]["high_fraction"]

    hidden_dim       = dims["hidden"]
    head_dim         = dims["head"]
    intermediate_dim = dims["intermediate"]

    # Hidden dim
    high_len_hidden = round(high_fraction * hidden_dim)
    low_len_hidden  = hidden_dim - high_len_hidden
    R1 = torch.block_diag(
        random_orthogonal_matrix(low_len_hidden, device),
        torch.eye(high_len_hidden, device=device),
    )

    # Head dim
    high_len_head = round(high_fraction * head_dim)
    low_len_head  = head_dim - high_len_head
    R2 = torch.block_diag(
        random_orthogonal_matrix(low_len_head, device),
        torch.eye(high_len_head, device=device),
    )

    # Intermediate dim
    high_len_down = round(high_fraction * intermediate_dim)
    low_len_down  = intermediate_dim - high_len_down
    R_down = torch.block_diag(
        random_orthogonal_matrix(low_len_down, device),
        torch.eye(high_len_down, device=device),
    )

    return {
        "R1":              R1.cpu(),
        "R2":              R2.cpu(),
        "R_down":          R_down.cpu(),
        "high_len_hidden": high_len_hidden,
        "high_len_head":   high_len_head,
        "high_len_down":   high_len_down,
        "high_fraction":   high_fraction,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model name (must match a folder under models/)")
    parser.add_argument("--output", default=None,
                        help="Override output path from config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    output_path = args.output or cfg["rotation"]["output_path"]

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}  (delete to regenerate)")
    else:
        print(f"Generating rotation matrices for '{args.model}' (R_high = Identity)...")
        d = generate_rotation_matrices(cfg, device=args.device, seed=args.seed)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(d, output_path)
        print(f"Saved to {output_path}")
        for k, v in d.items():
            print(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else v}")

    print("Done.")
