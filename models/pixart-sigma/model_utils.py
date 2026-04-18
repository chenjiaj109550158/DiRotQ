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
from utils.hadamard_utils import generate_sign_flips


# PixArt-Sigma generation parameters (passed to generate_images)
generation_params = dict(
    num_inference_steps=20,
    guidance_scale=4.5,
    height=1024,
    width=1024,
)


def assign_online_rotations(transformer, basis_dict, rotation_dict, cfg,
                             hadamard_layers=None, sign_flips_dict=None):
    """Assign online PCA rotation matrices to each ActQuantWrapper.

    PixArt-Sigma layer mapping:
      - attn1.to_q/k/v:   self-attention QKV   -> hidden-dim rotation (U @ R1)
      - attn2.to_q:        cross-attention Q    -> hidden-dim rotation (U @ R1)
      - attn1.to_out.0:    self-attn out proj   -> per-head rotation (U @ R2)
      - attn2.to_out.0:    cross-attn out proj  -> per-head rotation (U @ R2)
      - ff.net.0.proj:     FFN up projection    -> hidden-dim rotation (U @ R1)
      - ff.net.2:          FFN down projection  -> intermediate-dim rotation (U @ R_down)
                            OR Hadamard if in hadamard_layers

    Args:
        hadamard_layers: list of layer suffix patterns to use Hadamard instead of PCA.
                         e.g. ["ff.net.2"] for FFN down projection only.
        sign_flips_dict: optional dict {block_idx: tensor} of optimized sign flips.
                         If provided, uses these instead of random sign flips.
    """
    if hadamard_layers is None:
        hadamard_layers = []
    if sign_flips_dict is None:
        sign_flips_dict = {}

    num_heads = cfg["dims"]["num_heads"]
    head_dim = cfg["dims"]["head"]
    intermediate_dim = cfg["dims"]["intermediate"]

    R1     = rotation_dict["R1"].float()
    R2     = rotation_dict["R2"].float()
    R_down = rotation_dict["R_down"].float()

    assigned = 0
    hadamard_count = 0
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

        # Check if this layer should use Hadamard
        use_had = any(pat in layer_suffix for pat in hadamard_layers)

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
            if use_had:
                # Hadamard rotation: O(D log D) instead of O(D²)
                # low_dim = largest power of 2 <= D (here 4096 for D=4608)
                low_dim = 1 << (intermediate_dim.bit_length() - 1)  # 4096
                module.use_hadamard = True
                module.hadamard_low_dim = low_dim
                if block_idx in sign_flips_dict:
                    module.hadamard_sign_flips = sign_flips_dict[block_idx]
                else:
                    module.hadamard_sign_flips = generate_sign_flips(low_dim, seed=42 + block_idx)
                hadamard_count += 1
            else:
                evec = basis_dict[f"layer.{block_idx}.ffn.down_proj"].float()
                module.rotation = evec @ R_down
            assigned += 1

    print(f"Assigned rotations to {assigned} ActQuantWrapper layers "
          f"({hadamard_count} Hadamard, {assigned - hadamard_count} PCA).")
    return assigned


def configure_quantizers_by_name(transformer, high_len_hidden, high_len_head, cfg,
                                 nvfp4=False, hadamard_layers=None, a_groupsize=None,
                                 high_len_down=0, skip_quant_layers=None):
    """Configure mixed-precision activation quantizers by PixArt-Sigma layer type.

    When nvfp4=False (INT4):
      - QKV and FFN up-proj: groupsize=64, asymmetric
      - attn to_out: groupsize=head_dim (per head per token), asymmetric
      - FFN down-proj: groupsize=64, no mixed-precision

    When nvfp4=True (NF4):
      - All layers: groupsize=16 (except to_out: groupsize=72/head_dim), symmetric
      - NF4 codebook quantization
    """
    a_bits = cfg["quantization"]["a_bits"]
    head_dim = cfg["dims"]["head"]

    if nvfp4:
        nvfp4_cfg = cfg.get("nvfp4", {})
        a_gs = nvfp4_cfg.get("a_groupsize", 16)
        a_gs_out = nvfp4_cfg.get("a_groupsize_attn_out", head_dim)
        qdt = "nvfp4"
    else:
        a_gs = 64
        a_gs_out = head_dim
        qdt = "int"

    if a_groupsize is not None:
        a_gs = a_groupsize
        # Keep to_out at per-head groupsize — PCA rotation is per-head,
        # so groups must not cross head boundaries
        a_gs_out = head_dim

    # Align high_bits_length so the 4-bit region is divisible by groupsize.
    # Only hidden and down are aligned — per-head layers (to_out) always use
    # d_q = head_dim - high_len_head as their effective groupsize in the fused
    # forward, so aligning high_len_head to a_gs_out would wrongly zero out the
    # entire 4-bit region (e.g. _ceil_to_gs(9, 72) = 72 → d_q = 0).
    if a_gs > 0:
        def _ceil_to_gs(high, gs):
            if high == 0:
                return 0
            return ((high + gs - 1) // gs) * gs
        high_len_hidden_aligned = _ceil_to_gs(high_len_hidden, a_gs)
        high_len_down_aligned   = _ceil_to_gs(high_len_down,   a_gs)
        if high_len_hidden_aligned != high_len_hidden or high_len_down_aligned != high_len_down:
            print(f"Aligned high_bits_length to gs={a_gs}: "
                  f"hidden {high_len_hidden}->{high_len_hidden_aligned} "
                  f"({high_len_hidden_aligned}/{cfg['dims']['hidden']}="
                  f"{high_len_hidden_aligned/cfg['dims']['hidden']*100:.1f}%), "
                  f"down {high_len_down}->{high_len_down_aligned} "
                  f"({high_len_down_aligned}/{cfg['dims']['intermediate']}="
                  f"{high_len_down_aligned/cfg['dims']['intermediate']*100:.1f}%)")
        high_len_hidden = high_len_hidden_aligned
        high_len_down   = high_len_down_aligned

    if hadamard_layers is None:
        hadamard_layers = []
    if skip_quant_layers is None:
        skip_quant_layers = []
    intermediate_dim = cfg["dims"]["intermediate"]

    n_skipped = 0
    for name, module in transformer.named_modules():
        if not isinstance(module, ActQuantWrapper):
            continue

        # Skip activation quantization for specified layers (bits=16 = no-op)
        if any(pat in name for pat in skip_quant_layers):
            module.quantizer.configure(bits=16, groupsize=-1, sym=True)
            n_skipped += 1
            continue

        is_self_attn_qkv = (".attn1.to_q" in name or ".attn1.to_k" in name
                             or ".attn1.to_v" in name or ".attn2.to_q" in name)
        is_attn_out  = ".attn1.to_out" in name or ".attn2.to_out" in name
        is_ffn_up    = ".ff." in name and ".net." in name and name.endswith(".proj")
        is_ffn_down  = ".ff." in name and ".net." in name and name.endswith(".2")

        if is_self_attn_qkv:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden,
                quant_dtype=qdt,
            )
        elif is_attn_out:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs_out, sym=nvfp4,
                high_bits_length=high_len_head,
                quant_dtype=qdt,
            )
        elif is_ffn_up:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden,
                quant_dtype=qdt,
            )
        elif is_ffn_down:
            use_had = any(pat in name for pat in hadamard_layers)
            if use_had:
                # Hadamard: split = [low_dim (4096, quantized) | high_dim (512, 16-bit)]
                low_dim = 1 << (intermediate_dim.bit_length() - 1)
                had_high_len = intermediate_dim - low_dim  # 512
            else:
                # PCA: last high_len_down channels (576 = 1/8 of 4608) kept at 16-bit
                had_high_len = high_len_down
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=had_high_len,
                quant_dtype=qdt,
            )
        else:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=0,
                quant_dtype=qdt,
            )

    if n_skipped:
        print(f"Skipped activation quantization for {n_skipped} layers (bits=16).")
