"""
Model-specific utilities for Sana-1.6B.

Provides:
  - assign_online_rotations: map PCA basis + rotation matrices to Sana attn layers
  - configure_quantizers_by_name: set mixed-precision quantizer config per layer type
  - generation_params: Sana generation settings (steps, guidance, resolution)

Architecture notes:
  Quantized layers (matching SVDQuant):
    attn1.to_q/k/v, attn1.to_out.0  (self-attention)
    attn2.to_q, attn2.to_out.0      (cross-attn image-side Q and output)
  Skipped:
    attn2.to_k, attn2.to_v          (text K/V from fixed embeddings — SVDQuant attn_add skip)
  Not wrappable:
    ff.conv_inverted, ff.conv_point  (GLUMBConv Conv2d — add_actquant only handles nn.Linear)
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.quant_utils import (
    ActQuantWrapper, perm_idx_from_eigendecomp, perm_idx_from_eigendecomp_per_head,
)


# Generation parameters from SVDQuant sana-1.6b.yaml
generation_params = dict(
    num_inference_steps=20,
    guidance_scale=4.5,
    height=1024,
    width=1024,
)


def assign_online_rotations(transformer, basis_dict, rotation_dict, cfg,
                             hadamard_layers=None, sign_flips_dict=None,
                             pca_only_layers=None, residual_rotation="random"):
    """Assign online PCA rotation matrices to each ActQuantWrapper.

    Sana layer mapping:
      - attn1.to_q/k/v:  self-attn QKV input      -> hidden-dim rotation (U_sa  @ R1)
      - attn1.to_out.0:  self-attn out proj        -> per-head rotation   (U_sa_val @ R2)
      - attn2.to_q:      cross-attn query input    -> hidden-dim rotation (U_ca  @ R1)
      - attn2.to_out.0:  cross-attn out proj       -> per-head rotation   (U_ca_val @ R2)

    When a layer suffix matches any pattern in pca_only_layers, a channel permutation
    index is assigned instead of a rotation matrix (O(D) gather at inference vs O(D²) matmul).
    Requires eigenvalues in basis_dict (produced by current get_basis.py); falls back to
    rotation if eigenvalues are absent (old basis cache).
    """
    if residual_rotation not in {"random", "identity"}:
        raise ValueError(f"unsupported residual rotation: {residual_rotation}")
    if pca_only_layers is None:
        pca_only_layers = []

    num_heads = cfg["dims"]["num_heads"]
    head_dim  = cfg["dims"]["head"]

    # Identity mode deliberately does not access R1/R2.  It keeps the PCA
    # channel order and high-precision tail split, removing only residual R.
    if residual_rotation == "random":
        R1 = rotation_dict["R1"].float()
        R2 = rotation_dict["R2"].float()

    def _residual_basis(evec, rotation):
        return evec if residual_rotation == "identity" else evec @ rotation

    def _use_perm(suffix):
        return any(pat.strip() in suffix for pat in pca_only_layers)

    assigned = 0
    n_perm = 0
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
            evals_key = f"layer.{block_idx}.self_attn.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp(evec, basis_dict[evals_key].float())
                n_perm += 1
            else:
                module.rotation = _residual_basis(
                    evec, R1 if residual_rotation == "random" else None
                )
            assigned += 1
        elif layer_suffix == "attn1.to_out.0":
            evec_val = basis_dict[f"layer.{block_idx}.self_attn.value"].float()
            evals_key = f"layer.{block_idx}.self_attn.value.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp_per_head(
                    evec_val, basis_dict[evals_key].float()
                )
                n_perm += 1
            else:
                module.rotation_per_head = (
                    evec_val if residual_rotation == "identity" else
                    torch.bmm(evec_val, R2.unsqueeze(0).expand(num_heads, -1, -1))
                )
                module.num_heads = num_heads
                module.head_dim  = head_dim
            assigned += 1
        elif layer_suffix == "attn2.to_q":
            evec = basis_dict[f"layer.{block_idx}.cross_attn"].float()
            evals_key = f"layer.{block_idx}.cross_attn.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp(evec, basis_dict[evals_key].float())
                n_perm += 1
            else:
                module.rotation = _residual_basis(
                    evec, R1 if residual_rotation == "random" else None
                )
            assigned += 1
        elif layer_suffix == "attn2.to_out.0":
            evec_val = basis_dict[f"layer.{block_idx}.cross_attn.value"].float()
            evals_key = f"layer.{block_idx}.cross_attn.value.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp_per_head(
                    evec_val, basis_dict[evals_key].float()
                )
                n_perm += 1
            else:
                module.rotation_per_head = (
                    evec_val if residual_rotation == "identity" else
                    torch.bmm(evec_val, R2.unsqueeze(0).expand(num_heads, -1, -1))
                )
                module.num_heads = num_heads
                module.head_dim  = head_dim
            assigned += 1

    print(f"Assigned rotations to {assigned} ActQuantWrapper layers "
          f"({n_perm} perm_idx, {assigned - n_perm} rotation, "
          f"residual_rotation={residual_rotation}).")
    return assigned


def configure_quantizers_by_name(transformer, high_len_hidden, high_len_head, cfg,
                                 nvfp4=False, hadamard_layers=None, a_groupsize=None,
                                 high_len_down=0, skip_quant_layers=None,
                                 activation_format="nvfp4", format_stats=None):
    """Configure mixed-precision activation quantizers by Sana layer type.

    INT4 mode:
      - attn1.to_q/k/v, attn2.to_q: groupsize=64, high_bits_length=high_len_hidden
      - attn1.to_out.0, attn2.to_out.0: groupsize=head_dim (32), high_bits_length=high_len_head

    FP4 mode:
      - attn1.to_q/k/v, attn2.to_q: groupsize=16, symmetric
      - attn1.to_out.0, attn2.to_out.0: groupsize=head_dim (32), symmetric
      - activation_format selects the shared legacy/hardware FP4 fake quantizer;
        weight quantization, PCA, and rotation remain unchanged
    """
    a_bits   = cfg["quantization"]["a_bits"]
    head_dim = cfg["dims"]["head"]

    if nvfp4:
        allowed_formats = {
            "nvfp4", "nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle",
            "tile-mix-output-oracle", "a16w4-residual",
        }
        if activation_format not in allowed_formats:
            raise ValueError(
                f"SANA activation format {activation_format!r} is not enabled; "
                f"choose one of {sorted(allowed_formats)}"
            )
        nvfp4_cfg = cfg.get("nvfp4", {})
        a_gs      = nvfp4_cfg.get("a_groupsize", 16)
        a_gs_out  = nvfp4_cfg.get("a_groupsize_attn_out", head_dim)
        qdt       = activation_format
    else:
        a_gs     = 64
        a_gs_out = head_dim
        qdt      = "int"

    if a_groupsize is not None:
        a_gs     = a_groupsize
        a_gs_out = head_dim  # always keep per-head for to_out

    if a_gs > 0 and high_len_hidden > 0:
        high_len_hidden = ((high_len_hidden + a_gs - 1) // a_gs) * a_gs

    if hadamard_layers is None:
        hadamard_layers = []
    if skip_quant_layers is None:
        skip_quant_layers = []

    n_skipped = 0
    for name, module in transformer.named_modules():
        if not isinstance(module, ActQuantWrapper):
            continue
        module.quantizer.format_stats = format_stats

        if any(pat in name for pat in skip_quant_layers):
            module.quantizer.configure(bits=16, groupsize=-1, sym=True)
            n_skipped += 1
            continue

        is_self_attn_qkv = (
            ".attn1.to_q" in name or ".attn1.to_k" in name or ".attn1.to_v" in name
            or ".attn2.to_q" in name
        )
        is_attn_out = ".attn1.to_out" in name or ".attn2.to_out" in name

        # perm_idx layers always use flat (hidden-dim) groupsize and high_len_hidden
        # since the per-head structure is dissolved into a global channel sort.
        if getattr(module, 'perm_idx', None) is not None:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden,
                quant_dtype=qdt,
            )
        elif is_self_attn_qkv:
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
        else:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=0,
                quant_dtype=qdt,
            )

        # The hardware scale hierarchy is defined for the unclipped operand.
        # Fail before cache loading/quantization or generation if a future SANA
        # config changes the inherited ActQuantizer default.
        if qdt in {"nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle",
                   "tile-mix-output-oracle"}:
            if module.quantizer.clip_ratio != 1.0:
                raise ValueError(
                    f"{name}: hardware FP4 requires activation clip_ratio=1.0, "
                    f"got {module.quantizer.clip_ratio}"
                )

    if n_skipped:
        print(f"Skipped activation quantization for {n_skipped} layers (bits=16).")
