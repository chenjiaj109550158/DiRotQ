"""
FLUX.1-schnell specific rotation utilities for DiRotQ (offline weight-space).

Applies PCA basis rotations to each Flux(Single)TransformerBlock. Each block's
rotation is self-contained: input-side rotations on Q/K/V-like projections are
undone at the corresponding output projection for residual consistency.

Note: the online path in model_utils.py:assign_online_rotations is what the
current pipeline uses. These helpers mirror the PixArt offline pattern and are
kept for parity / experimentation.
"""

import torch
from utils.rotation_utils import (
    rotate_linear_input_side,
    rotate_linear_output_side,
    rotate_linear_output_per_head,
    rotate_linear_input_per_head,
)


def apply_double_block_rotations(block, block_idx, basis_dict, R1, R2, R_down,
                                  num_heads=24, head_dim=128):
    """
    Apply DiRotQ rotations to a single FluxTransformerBlock (double / joint-attn).

    Layer mapping:
      attn.to_q/k/v                        <- layer.{i}.img_attn        (hidden)
      attn.add_q_proj/add_k_proj/add_v_proj <- layer.{i}.txt_attn       (hidden)
      attn.to_v  (output) / attn.to_out[0] (input)
                                           <- layer.{i}.img_attn.value  (per-head)
      attn.add_v_proj (output) / attn.to_add_out (input)
                                           <- layer.{i}.txt_attn.value  (per-head)
      ff.net[0].proj                       <- layer.{i}.img_ffn         (hidden)
      ff.net[2]                            <- layer.{i}.img_ffn.down    (intermediate)
      ff_context.net[0].proj               <- layer.{i}.txt_ffn         (hidden)
      ff_context.net[2]                    <- layer.{i}.txt_ffn.down    (intermediate)
    """
    i = block_idx
    device = "cuda"
    R1_dev     = R1.to(device)
    R2_dev     = R2.to(device)
    R_down_dev = R_down.to(device)

    # --- Image-stream attention ---
    key_img_attn = f"layer.{i}.img_attn"
    U_img = basis_dict[key_img_attn].to(device) @ R1_dev  # [hidden, hidden]

    rotate_linear_input_side(block.attn.to_q, U_img)
    rotate_linear_input_side(block.attn.to_k, U_img)
    rotate_linear_input_side(block.attn.to_v, U_img)

    key_img_val = f"layer.{i}.img_attn.value"
    if key_img_val in basis_dict:
        U_img_val = basis_dict[key_img_val].to(device)  # [H, d, d]
        U_img_val_R = torch.bmm(U_img_val, R2_dev.unsqueeze(0).expand(num_heads, -1, -1))

        rotate_linear_output_per_head(block.attn.to_v,      U_img_val_R, num_heads, head_dim)
        rotate_linear_input_per_head (block.attn.to_out[0], U_img_val_R, num_heads, head_dim)

    # Undo image input rotation at image output (residual consistency)
    rotate_linear_output_side(block.attn.to_out[0], U_img)

    # --- Text-stream attention ---
    key_txt_attn = f"layer.{i}.txt_attn"
    U_txt = basis_dict[key_txt_attn].to(device) @ R1_dev  # [hidden, hidden]

    rotate_linear_input_side(block.attn.add_q_proj, U_txt)
    rotate_linear_input_side(block.attn.add_k_proj, U_txt)
    rotate_linear_input_side(block.attn.add_v_proj, U_txt)

    key_txt_val = f"layer.{i}.txt_attn.value"
    if key_txt_val in basis_dict:
        U_txt_val = basis_dict[key_txt_val].to(device)  # [H, d, d]
        U_txt_val_R = torch.bmm(U_txt_val, R2_dev.unsqueeze(0).expand(num_heads, -1, -1))

        rotate_linear_output_per_head(block.attn.add_v_proj, U_txt_val_R, num_heads, head_dim)
        rotate_linear_input_per_head (block.attn.to_add_out, U_txt_val_R, num_heads, head_dim)

    # Undo text input rotation at text output
    rotate_linear_output_side(block.attn.to_add_out, U_txt)

    # --- Image FFN ---
    key_img_ffn = f"layer.{i}.img_ffn"
    U_img_ffn = basis_dict[key_img_ffn].to(device) @ R1_dev  # [hidden, hidden]
    rotate_linear_input_side(block.ff.net[0].proj, U_img_ffn)

    key_img_down = f"layer.{i}.img_ffn.down"
    if key_img_down in basis_dict:
        U_img_down = basis_dict[key_img_down].to(device) @ R_down_dev  # [inter, inter]
        rotate_linear_input_side(block.ff.net[2], U_img_down)

    rotate_linear_output_side(block.ff.net[2], U_img_ffn)

    # --- Text FFN ---
    key_txt_ffn = f"layer.{i}.txt_ffn"
    U_txt_ffn = basis_dict[key_txt_ffn].to(device) @ R1_dev  # [hidden, hidden]
    rotate_linear_input_side(block.ff_context.net[0].proj, U_txt_ffn)

    key_txt_down = f"layer.{i}.txt_ffn.down"
    if key_txt_down in basis_dict:
        U_txt_down = basis_dict[key_txt_down].to(device) @ R_down_dev  # [inter, inter]
        rotate_linear_input_side(block.ff_context.net[2], U_txt_down)

    rotate_linear_output_side(block.ff_context.net[2], U_txt_ffn)


def apply_single_block_rotations(block, block_idx, basis_dict, R1):
    """
    Apply DiRotQ input-side rotations to a single FluxSingleTransformerBlock.

    Layer mapping:
      attn.to_q/k/v    <- single.{i}.attn  (hidden)
      proj_mlp         <- single.{i}.mlp   (hidden)

    proj_out is skipped: its input is [attn_out || mlp_out], a heterogeneous
    concatenation that cannot be rotated by a single orthogonal basis.

    Residual consistency is maintained at the block level by the parallel
    structure: since attn and mlp share the same normed hidden-state input
    (re-normalized upstream) and proj_out is left un-rotated, no output-side
    "undo" is needed here — but because attn + mlp share the same hidden input,
    both must use the same rotation. We bind them to single.{i}.attn so the
    fused input projection (conceptually a QKV+MLP-up block) is consistent.
    """
    i = block_idx
    device = "cuda"
    R1_dev = R1.to(device)

    key_attn = f"single.{i}.attn"
    U_attn = basis_dict[key_attn].to(device) @ R1_dev  # [hidden, hidden]
    rotate_linear_input_side(block.attn.to_q, U_attn)
    rotate_linear_input_side(block.attn.to_k, U_attn)
    rotate_linear_input_side(block.attn.to_v, U_attn)

    key_mlp = f"single.{i}.mlp"
    if key_mlp in basis_dict:
        U_mlp = basis_dict[key_mlp].to(device) @ R1_dev  # [hidden, hidden]
        rotate_linear_input_side(block.proj_mlp, U_mlp)
