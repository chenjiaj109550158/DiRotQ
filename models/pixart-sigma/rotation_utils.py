"""
PixArt-Sigma specific rotation utilities for DiRotQ.
Applies PCA basis rotations to each BasicTransformerBlock.
"""

import torch
from utils.rotation_utils import (
    rotate_linear_input_side,
    rotate_linear_output_side,
    rotate_linear_output_per_head,
    rotate_linear_input_per_head,
)


def apply_block_rotations(block, block_idx, basis_dict, R1, R2, R_down,
                          num_heads=16, head_dim=72):
    """
    Apply all DiRotQ rotations to a single BasicTransformerBlock.
    Each block's rotation is self-contained (undone at output for residual consistency).

    Args:
        block: BasicTransformerBlock module
        block_idx: Block index (0-27)
        basis_dict: Dict mapping keys like "layer.0.self_attn" to eigenvector matrices
        R1: [1152, 1152] block-diagonal random rotation for hidden dim
        R2: [72, 72] block-diagonal random rotation for head dim
        R_down: [4608, 4608] block-diagonal random rotation for intermediate dim
        num_heads: number of attention heads
        head_dim: dimension per head
    """
    i = block_idx
    device = "cuda"

    # --- Self-attention ---
    key_sa = f"layer.{i}.self_attn"
    U_sa = basis_dict[key_sa].to(device) @ R1.to(device)  # [1152, 1152]

    rotate_linear_input_side(block.attn1.to_q, U_sa)
    rotate_linear_input_side(block.attn1.to_k, U_sa)
    rotate_linear_input_side(block.attn1.to_v, U_sa)

    # V-output / O-input per-head rotation
    key_val = f"layer.{i}.self_attn.value"
    if key_val in basis_dict:
        U_val = basis_dict[key_val].to(device)  # [num_heads, head_dim, head_dim]
        R2_dev = R2.to(device)
        U_val_R = torch.bmm(U_val, R2_dev.unsqueeze(0).expand(num_heads, -1, -1))

        rotate_linear_output_per_head(block.attn1.to_v, U_val_R, num_heads, head_dim)
        rotate_linear_input_per_head(block.attn1.to_out[0], U_val_R, num_heads, head_dim)

    # Undo input rotation at attn1 output (residual consistency)
    rotate_linear_output_side(block.attn1.to_out[0], U_sa)

    # --- Cross-attention ---
    key_ca = f"layer.{i}.cross_attn_q"
    if key_ca in basis_dict:
        U_ca = basis_dict[key_ca].to(device) @ R1.to(device)  # [1152, 1152]

        rotate_linear_input_side(block.attn2.to_q, U_ca)

        key_val_ca = f"layer.{i}.cross_attn_q.value"
        if key_val_ca in basis_dict:
            U_val_ca = basis_dict[key_val_ca].to(device)  # [num_heads, head_dim, head_dim]
            U_val_ca_R = torch.bmm(U_val_ca, R2_dev.unsqueeze(0).expand(num_heads, -1, -1))

            rotate_linear_output_per_head(block.attn2.to_v, U_val_ca_R, num_heads, head_dim)
            rotate_linear_input_per_head(block.attn2.to_out[0], U_val_ca_R, num_heads, head_dim)

        # Undo cross-attn rotation at output
        rotate_linear_output_side(block.attn2.to_out[0], U_ca)

    # --- FFN ---
    key_ffn = f"layer.{i}.ffn"
    U_ffn = basis_dict[key_ffn].to(device) @ R1.to(device)  # [1152, 1152]

    rotate_linear_input_side(block.ff.net[0].proj, U_ffn)

    key_down = f"layer.{i}.ffn.down_proj"
    if key_down in basis_dict:
        U_down = basis_dict[key_down].to(device) @ R_down.to(device)  # [4608, 4608]
        rotate_linear_input_side(block.ff.net[2], U_down)

    # Undo FFN rotation at output (residual consistency)
    rotate_linear_output_side(block.ff.net[2], U_ffn)
