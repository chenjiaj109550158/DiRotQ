"""
Model-specific utilities for PixArt-Sigma.

Provides:
  - assign_online_rotations: map PCA basis + rotation matrices to PixArt layers
  - configure_quantizers_by_name: set mixed-precision quantizer config per layer type
  - GENERATION_PARAMS: PixArt-Sigma generation settings (steps, guidance, resolution)
"""

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.quant_utils import ActQuantWrapper


# PixArt-Sigma generation parameters (passed to generate_images)
generation_params = dict(
    num_inference_steps=20,
    guidance_scale=4.5,
    height=1024,
    width=1024,
)


def assign_online_rotations(transformer, basis_dict, rotation_dict, cfg):
    """Assign online PCA rotation matrices to each ActQuantWrapper.

    PixArt-Sigma layer mapping:
      - attn1.to_q/k/v:   self-attention QKV   -> hidden-dim rotation (U @ R1)
      - attn2.to_q:        cross-attention Q    -> hidden-dim rotation (U @ R1)
      - attn1.to_out.0:    self-attn out proj   -> per-head rotation (U @ R2)
      - attn2.to_out.0:    cross-attn out proj  -> per-head rotation (U @ R2)
      - ff.net.0.proj:     FFN up projection    -> hidden-dim rotation (U @ R1)
      - ff.net.2:          FFN down projection  -> intermediate-dim rotation (U @ R_down)
    """
    num_heads = cfg["dims"]["num_heads"]
    head_dim = cfg["dims"]["head"]

    R1     = rotation_dict["R1"].float()
    R2     = rotation_dict["R2"].float()
    R_down = rotation_dict["R_down"].float()

    assigned = 0
    for name, module in transformer.named_modules():
        if not isinstance(module, ActQuantWrapper):
            continue

        parts = name.split(".")
        if "transformer_blocks" not in parts:
            continue
        try:
            block_idx = int(parts[parts.index("transformer_blocks") + 1])
        except (ValueError, IndexError):
            continue

        layer_suffix = ".".join(parts[parts.index("transformer_blocks") + 2:])

        if layer_suffix in ("attn1.to_q", "attn1.to_k", "attn1.to_v"):
            evec = basis_dict[f"layer.{block_idx}.self_attn"].float()
            module.rotation = evec @ R1
            assigned += 1
        elif layer_suffix == "attn2.to_q":
            evec = basis_dict[f"layer.{block_idx}.cross_attn_q"].float()
            module.rotation = evec @ R1
            assigned += 1
        elif layer_suffix == "attn1.to_out.0":
            evec_val = basis_dict[f"layer.{block_idx}.self_attn.value"].float()
            module.rotation_per_head = torch.bmm(evec_val, R2.unsqueeze(0).expand(num_heads, -1, -1))
            module.num_heads = num_heads
            module.head_dim  = head_dim
            assigned += 1
        elif layer_suffix == "attn2.to_out.0":
            evec_val_ca = basis_dict[f"layer.{block_idx}.cross_attn_q.value"].float()
            module.rotation_per_head = torch.bmm(evec_val_ca, R2.unsqueeze(0).expand(num_heads, -1, -1))
            module.num_heads = num_heads
            module.head_dim  = head_dim
            assigned += 1
        elif "ff.net" in layer_suffix and layer_suffix.endswith(".proj"):
            evec = basis_dict[f"layer.{block_idx}.ffn"].float()
            module.rotation = evec @ R1
            assigned += 1
        elif "ff.net" in layer_suffix and layer_suffix.endswith(".2"):
            evec = basis_dict[f"layer.{block_idx}.ffn.down_proj"].float()
            module.rotation = evec @ R_down
            assigned += 1

    print(f"Assigned rotations to {assigned} ActQuantWrapper layers.")
    return assigned


def configure_quantizers_by_name(transformer, high_len_hidden, high_len_head, cfg,
                                 nvfp4=False):
    """Configure mixed-precision activation quantizers by PixArt-Sigma layer type.

    When nvfp4=False (INT4):
      - QKV and FFN up-proj: groupsize=-1 (per-token), asymmetric
      - attn to_out: groupsize=head_dim (per head per token), asymmetric
      - FFN down-proj: groupsize=-1, no mixed-precision

    When nvfp4=True (NF4):
      - All layers: groupsize=16 (except to_out: groupsize=72/head_dim), symmetric
      - NF4 codebook quantization
    """
    a_bits = cfg["quantization"]["a_bits"]
    high_bits = cfg["quantization"]["high_bits"]
    head_dim = cfg["dims"]["head"]

    if nvfp4:
        nvfp4_cfg = cfg.get("nvfp4", {})
        a_gs = nvfp4_cfg.get("a_groupsize", 16)
        a_gs_out = nvfp4_cfg.get("a_groupsize_attn_out", head_dim)
        qdt = "nvfp4"
    else:
        a_gs = -1
        a_gs_out = head_dim
        qdt = "int"

    for name, module in transformer.named_modules():
        if not isinstance(module, ActQuantWrapper):
            continue

        is_self_attn_qkv = (".attn1.to_q" in name or ".attn1.to_k" in name
                             or ".attn1.to_v" in name or ".attn2.to_q" in name)
        is_attn_out  = ".attn1.to_out" in name or ".attn2.to_out" in name
        is_ffn_up    = ".ff." in name and ".net." in name and name.endswith(".proj")
        is_ffn_down  = ".ff." in name and ".net." in name and name.endswith(".2")

        if is_self_attn_qkv:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden, high_bits=high_bits,
                low_bits_length=0, low_bits=high_bits,
                quant_dtype=qdt,
            )
        elif is_attn_out:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs_out, sym=nvfp4,
                high_bits_length=high_len_head, high_bits=high_bits,
                low_bits_length=0, low_bits=high_bits,
                quant_dtype=qdt,
            )
        elif is_ffn_up:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden, high_bits=high_bits,
                low_bits_length=0, low_bits=high_bits,
                quant_dtype=qdt,
            )
        elif is_ffn_down:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=0, high_bits=high_bits,
                low_bits_length=0, low_bits=high_bits,
                quant_dtype=qdt,
            )
        else:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=0, high_bits=high_bits,
                quant_dtype=qdt,
            )
