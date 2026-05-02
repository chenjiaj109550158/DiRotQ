"""
Model-specific utilities for FLUX.1-schnell.

Provides:
  - assign_online_rotations: map PCA basis + rotation matrices to FLUX layers
  - configure_quantizers_by_name: set mixed-precision quantizer config per layer type
  - generation_params: FLUX.1-schnell generation settings (4 steps, guidance=0)

Layer coverage mirrors deepcompressor/svdquant flux.1-schnell.yaml. Skip lists are
applied in config.yaml (`quantization.skip_layers`); this file covers the layers
that ARE quantized.

FLUX.1-schnell architecture:
  transformer_blocks (19, FluxTransformerBlock, joint-attention):
    attn.to_q/to_k/to_v                     -- image stream Q/K/V
    attn.add_q_proj/add_k_proj/add_v_proj   -- text stream Q/K/V
    attn.to_out.0                           -- image stream out proj
    attn.to_add_out                         -- text stream out proj
    ff.net.0.proj, ff.net.2                 -- image FFN up / down
    ff_context.net.0.proj, ff_context.net.2 -- text FFN up / down

  single_transformer_blocks (38, FluxSingleTransformerBlock, parallel):
    attn.to_q/to_k/to_v                     -- pre_only (no to_out)
    proj_mlp                                -- MLP up proj
    proj_out (SPLIT)                        -- FluxProjOutSplit with two Linears:
      proj_out.linears.0 (attn half)          hidden -> hidden, per-head rotation
      proj_out.linears.1 (mlp half)           inter  -> hidden, dense rotation
"""

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.quant_utils import (
    ActQuantWrapper, perm_idx_from_eigendecomp, perm_idx_from_eigendecomp_per_head,
)

sys.path.insert(0, os.path.dirname(__file__))
from split_proj_out import split_flux_single_proj_out


def preprocess_transformer(transformer, cfg):
    """Model-level structural preprocessing run before ActQuantWrapper wrapping.

    For FLUX, this splits the fused single-block proj_out into two Linear
    branches so each half gets its own rotation + activation quantizer, matching
    SVDQuant's ConcatLinear treatment.
    """
    n = split_flux_single_proj_out(transformer)
    if n > 0:
        print(f"Split {n} single-block proj_out layers into attn/mlp halves.")
    return transformer


# FLUX.1-schnell generation parameters
generation_params = dict(
    num_inference_steps=4,
    guidance_scale=0.0,
    height=1024,
    width=1024,
    max_sequence_length=256,
)


def _is_single_block(name: str) -> bool:
    return "single_transformer_blocks" in name


def _block_idx(name: str) -> int | None:
    parts = name.split(".")
    for container in ("single_transformer_blocks", "transformer_blocks"):
        if container in parts:
            i = parts.index(container)
            try:
                return int(parts[i + 1])
            except (ValueError, IndexError):
                return None
    return None


def assign_online_rotations(transformer, basis_dict, rotation_dict, cfg,
                             hadamard_layers=None, sign_flips_dict=None,
                             pca_only_layers=None):
    """Assign online PCA rotation matrices to each ActQuantWrapper.

    FLUX layer mapping (only applied if the matching basis key is present):

      Double blocks (transformer_blocks.{i}.*):
        attn.to_q/k/v                     -> "layer.{i}.img_attn"        (hidden)
        attn.add_q_proj/add_k_proj/add_v_proj -> "layer.{i}.txt_attn"    (hidden)
        attn.to_out.0                     -> "layer.{i}.img_attn.value"  (per-head)
        attn.to_add_out                   -> "layer.{i}.txt_attn.value"  (per-head)
        ff.net.0.proj                     -> "layer.{i}.img_ffn"         (hidden)
        ff.net.2                          -> "layer.{i}.img_ffn.down"    (intermediate)
        ff_context.net.0.proj             -> "layer.{i}.txt_ffn"         (hidden)
        ff_context.net.2                  -> "layer.{i}.txt_ffn.down"    (intermediate)

      Single blocks (single_transformer_blocks.{i}.*):
        attn.to_q/k/v                     -> "single.{i}.attn"             (hidden)
        proj_mlp                          -> "single.{i}.mlp"              (hidden)
        proj_out.linears.0 (attn half)    -> "single.{i}.attn_out.value"   (per-head)
        proj_out.linears.1 (mlp  half)    -> "single.{i}.mlp.down"         (intermediate)

    If basis_dict is empty or a key is missing, the layer is left without a
    rotation (falls through to standard activation quantization).

    When a layer suffix matches any pattern in pca_only_layers, a perm_idx is
    assigned instead of a rotation matrix. Falls back to rotation if eigenvalues
    are absent in basis_dict (old basis cache without .eigenvalues entries).
    """
    if sign_flips_dict is None:
        sign_flips_dict = {}
    if pca_only_layers is None:
        pca_only_layers = []

    num_heads = cfg["dims"]["num_heads"]
    head_dim  = cfg["dims"]["head"]

    R1     = rotation_dict["R1"].float()     if rotation_dict and "R1"     in rotation_dict else None
    R2     = rotation_dict["R2"].float()     if rotation_dict and "R2"     in rotation_dict else None
    R_down = rotation_dict["R_down"].float() if rotation_dict and "R_down" in rotation_dict else None

    def _hidden(evec):
        return evec.float() @ R1 if R1 is not None else evec.float()

    def _down(evec):
        return evec.float() @ R_down if R_down is not None else evec.float()

    def _use_perm(suffix):
        return any(pat.strip() in suffix for pat in pca_only_layers)

    def _assign_perm_or_rot(module, key, suffix, rot_fn):
        """Assign perm_idx if pca_only matches and eigenvalues exist, else rotation."""
        if key not in basis_dict:
            return False
        if _use_perm(suffix):
            evals_key = f"{key}.eigenvalues"
            if evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp(
                    basis_dict[key].float(), basis_dict[evals_key].float()
                )
                return True
        module.rotation = rot_fn(basis_dict[key])
        return True

    assigned = 0
    n_perm = 0
    for name, module in transformer.named_modules():
        if not isinstance(module, ActQuantWrapper):
            continue

        bi = _block_idx(name)
        if bi is None:
            continue

        is_single = _is_single_block(name)
        parts = name.split(".")
        container = "single_transformer_blocks" if is_single else "transformer_blocks"
        start = parts.index(container) + 2
        suffix = ".".join(parts[start:])

        prev_perm = module.perm_idx

        if is_single:
            if suffix in ("attn.to_q", "attn.to_k", "attn.to_v"):
                if _assign_perm_or_rot(module, f"single.{bi}.attn", suffix, _hidden):
                    assigned += 1
            elif suffix == "proj_mlp":
                if _assign_perm_or_rot(module, f"single.{bi}.mlp", suffix, _hidden):
                    assigned += 1
            elif suffix == "proj_out.linears.0":
                if _assign_perm_or_rot(module, f"single.{bi}.attn_out.value", suffix, _hidden):
                    assigned += 1
            elif suffix == "proj_out.linears.1":
                if _assign_perm_or_rot(module, f"single.{bi}.mlp.down", suffix, _down):
                    assigned += 1
        else:
            if suffix in ("attn.to_q", "attn.to_k", "attn.to_v"):
                if _assign_perm_or_rot(module, f"layer.{bi}.img_attn", suffix, _hidden):
                    assigned += 1
            elif suffix in ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"):
                if _assign_perm_or_rot(module, f"layer.{bi}.txt_attn", suffix, _hidden):
                    assigned += 1
            elif suffix == "attn.to_out.0":
                if _assign_perm_or_rot(module, f"layer.{bi}.img_attn.value", suffix, _hidden):
                    assigned += 1
            elif suffix == "attn.to_add_out":
                if _assign_perm_or_rot(module, f"layer.{bi}.txt_attn.value", suffix, _hidden):
                    assigned += 1
            elif suffix == "ff.net.0.proj":
                if _assign_perm_or_rot(module, f"layer.{bi}.img_ffn", suffix, _hidden):
                    assigned += 1
            elif suffix == "ff.net.2":
                if _assign_perm_or_rot(module, f"layer.{bi}.img_ffn.down", suffix, _down):
                    assigned += 1
            elif suffix == "ff_context.net.0.proj":
                if _assign_perm_or_rot(module, f"layer.{bi}.txt_ffn", suffix, _hidden):
                    assigned += 1
            elif suffix == "ff_context.net.2":
                if _assign_perm_or_rot(module, f"layer.{bi}.txt_ffn.down", suffix, _down):
                    assigned += 1

        if module.perm_idx is not None and prev_perm is None:
            n_perm += 1

    print(f"Assigned rotations to {assigned} ActQuantWrapper layers "
          f"({n_perm} perm_idx, {assigned - n_perm} rotation).")
    return assigned


def configure_quantizers_by_name(transformer, high_len_hidden, high_len_head, cfg,
                                 nvfp4=False, hadamard_layers=None, a_groupsize=None,
                                 high_len_down=0, skip_quant_layers=None):
    """Configure mixed-precision activation quantizers by FLUX layer type.

    Layer buckets:
      QKV-hidden (per-token on hidden dim):
        double attn.to_q/k/v, double attn.add_q/k/v_proj,
        single attn.to_q/k/v, single proj_mlp,
        double ff[ _context ].net.0.proj
      Attn-out (per-head groupsize):
        double attn.to_out.0, double attn.to_add_out
      FFN-down:
        double ff[ _context ].net.2
      Single-block split proj_out:
        proj_out.linears.0 — attn-out half, per-head groupsize + high_len_head slice
        proj_out.linears.1 — mlp half, dense groupsize + high_len_down slice
    """
    a_bits   = cfg["quantization"]["a_bits"]
    head_dim = cfg["dims"]["head"]

    if nvfp4:
        nvfp4_cfg = cfg.get("nvfp4", {})
        a_gs     = nvfp4_cfg.get("a_groupsize", 16)
        a_gs_out = nvfp4_cfg.get("a_groupsize_attn_out", head_dim)
        qdt      = "nvfp4"
    else:
        a_gs     = 64
        a_gs_out = head_dim
        qdt      = "int"

    if a_groupsize is not None:
        a_gs     = a_groupsize
        a_gs_out = head_dim

    # Align hidden and down so the 4-bit region is divisible by groupsize.
    # high_len_head is NOT aligned — per-head layers always use d_q as their
    # effective groupsize, so aligning to a_gs_out would zero out d_q entirely.
    if a_gs > 0:
        def _ceil(high, gs):
            return 0 if high == 0 else ((high + gs - 1) // gs) * gs
        high_len_hidden_aligned = _ceil(high_len_hidden, a_gs)
        high_len_down_aligned   = _ceil(high_len_down,   a_gs)
        if high_len_hidden_aligned != high_len_hidden or high_len_down_aligned != high_len_down:
            print(f"Aligned high_bits_length to gs={a_gs}: "
                  f"hidden {high_len_hidden}->{high_len_hidden_aligned}, "
                  f"down {high_len_down}->{high_len_down_aligned}")
        high_len_hidden = high_len_hidden_aligned
        high_len_down   = high_len_down_aligned

    if skip_quant_layers is None:
        skip_quant_layers = []

    n_skipped = 0
    for name, module in transformer.named_modules():
        if not isinstance(module, ActQuantWrapper):
            continue

        if any(pat in name for pat in skip_quant_layers):
            module.quantizer.configure(bits=16, groupsize=-1, sym=True)
            n_skipped += 1
            continue

        is_single = _is_single_block(name)

        # Attention bucket
        is_attn_qkv = (
            ".attn.to_q" in name or ".attn.to_k" in name or ".attn.to_v" in name
            or ".attn.add_q_proj" in name
            or ".attn.add_k_proj" in name
            or ".attn.add_v_proj" in name
        )
        is_attn_out = (".attn.to_out" in name) or (".attn.to_add_out" in name)

        # Feed-forward buckets
        is_double_ff_up   = (not is_single) and name.endswith(".net.0.proj")
        is_double_ff_down = (not is_single) and name.endswith(".net.2")

        # Single-block buckets
        is_single_mlp_up         = is_single and name.endswith(".proj_mlp")
        is_single_proj_out_attn  = is_single and name.endswith(".proj_out.linears.0")
        is_single_proj_out_mlp   = is_single and name.endswith(".proj_out.linears.1")

        # perm_idx layers always use flat hidden-dim groupsize and high_len_hidden.
        # For FFN-down perm_idx layers, high_len_hidden is used instead of high_len_down
        # since we're sorting globally by variance rather than using the down-proj basis.
        if getattr(module, 'perm_idx', None) is not None:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden,
                quant_dtype=qdt,
            )
        elif is_attn_qkv or is_double_ff_up or is_single_mlp_up:
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden,
                quant_dtype=qdt,
            )
        elif is_attn_out or is_single_proj_out_attn:
            # Attn-out uses global (flat) rotation, same as QKV/FFN layers.
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_hidden,
                quant_dtype=qdt,
            )
        elif is_double_ff_down or is_single_proj_out_mlp:
            # MLP half of split proj_out is the FFN-down input — same bucket
            # as double-block ff.net.2.
            module.quantizer.configure(
                bits=a_bits, groupsize=a_gs, sym=nvfp4,
                high_bits_length=high_len_down,
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
