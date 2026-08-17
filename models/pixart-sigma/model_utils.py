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
from utils.quant_utils import (
    ActQuantWrapper, perm_idx_from_eigendecomp, perm_idx_from_eigendecomp_per_head,
)
from utils.hadamard_utils import generate_sign_flips


# PixArt-Sigma generation parameters (passed to generate_images)
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

    PixArt-Sigma layer mapping:
      - attn1.to_q/k/v:   self-attention QKV   -> hidden-dim rotation (U @ R1)
      - attn2.to_q:        cross-attention Q    -> hidden-dim rotation (U @ R1)
      - attn1.to_out.0:    self-attn out proj   -> per-head rotation (U @ R2)
      - attn2.to_out.0:    cross-attn out proj  -> per-head rotation (U @ R2)
      - ff.net.0.proj:     FFN up projection    -> hidden-dim rotation (U @ R1)
      - ff.net.2:          FFN down projection  -> intermediate-dim rotation (U @ R_down)
                            OR Hadamard if in hadamard_layers

    When a layer suffix matches any pattern in pca_only_layers, a perm_idx is assigned
    instead of a rotation matrix. Falls back to rotation if eigenvalues absent in basis_dict.
    """
    if hadamard_layers is None:
        hadamard_layers = []
    if sign_flips_dict is None:
        sign_flips_dict = {}
    if pca_only_layers is None:
        pca_only_layers = []

    num_heads = cfg["dims"]["num_heads"]
    head_dim = cfg["dims"]["head"]
    intermediate_dim = cfg["dims"]["intermediate"]

    if residual_rotation not in {"random", "identity"}:
        raise ValueError(f"unsupported residual rotation: {residual_rotation}")

    # Identity mode deliberately does not access any R tensor.  It retains the
    # PCA basis/order (and therefore the identical high-precision split) while
    # removing only the random orthogonal transform inside the residual space.
    if residual_rotation == "random":
        R1 = rotation_dict["R1"].float()
        R2 = rotation_dict["R2"].float()
        R_down = rotation_dict["R_down"].float()

    # Derived shared-basis artifacts map every legacy per-layer key to a
    # canonical group.  Cache the post-residual rotation too: merely aliasing
    # PCA tensors on disk is not a memory reduction if ``evec @ R`` is
    # materialized again for every Linear.
    shared_map = basis_dict.get("__shared_basis_map__", {})
    rotation_cache = {}

    def _canonical(key):
        return shared_map.get(key, key)

    def _residual_basis(key, evec, rotation, kind):
        cache_key = (_canonical(key), residual_rotation, kind)
        if cache_key not in rotation_cache:
            rotation_cache[cache_key] = (
                evec if residual_rotation == "identity" else evec @ rotation
            )
        return rotation_cache[cache_key]

    def _residual_basis_per_head(key, evec, rotation):
        cache_key = (_canonical(key), residual_rotation, "per_head")
        if cache_key not in rotation_cache:
            rotation_cache[cache_key] = (
                evec if residual_rotation == "identity" else
                torch.bmm(evec, rotation.unsqueeze(0).expand(num_heads, -1, -1))
            )
        return rotation_cache[cache_key]

    def _use_perm(suffix):
        return any(pat.strip() in suffix for pat in pca_only_layers)

    assigned = 0
    hadamard_count = 0
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

        # Check if this layer should use Hadamard (only applies to ff.net.2)
        use_had = any(pat in layer_suffix for pat in hadamard_layers)

        if layer_suffix in ("attn1.to_q", "attn1.to_k", "attn1.to_v"):
            basis_key = f"layer.{block_idx}.self_attn"
            evec = basis_dict[basis_key].float()
            evals_key = f"{basis_key}.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp(evec, basis_dict[evals_key].float())
                n_perm += 1
            else:
                module.rotation = _residual_basis(
                    basis_key, evec, R1 if residual_rotation == "random" else None, "hidden"
                )
            assigned += 1
        elif layer_suffix == "attn2.to_q":
            basis_key = f"layer.{block_idx}.cross_attn_q"
            evec = basis_dict[basis_key].float()
            evals_key = f"{basis_key}.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp(evec, basis_dict[evals_key].float())
                n_perm += 1
            else:
                module.rotation = _residual_basis(
                    basis_key, evec, R1 if residual_rotation == "random" else None, "hidden"
                )
            assigned += 1
        elif layer_suffix == "attn1.to_out.0":
            basis_key = f"layer.{block_idx}.self_attn.value"
            evec_val = basis_dict[basis_key].float()
            evals_key = f"{basis_key}.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp_per_head(
                    evec_val, basis_dict[evals_key].float()
                )
                n_perm += 1
            else:
                module.rotation_per_head = _residual_basis_per_head(
                    basis_key, evec_val, R2 if residual_rotation == "random" else None
                )
                module.num_heads = num_heads
                module.head_dim  = head_dim
            assigned += 1
        elif layer_suffix == "attn2.to_out.0":
            basis_key = f"layer.{block_idx}.cross_attn_q.value"
            evec_val_ca = basis_dict[basis_key].float()
            evals_key = f"{basis_key}.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp_per_head(
                    evec_val_ca, basis_dict[evals_key].float()
                )
                n_perm += 1
            else:
                module.rotation_per_head = _residual_basis_per_head(
                    basis_key, evec_val_ca, R2 if residual_rotation == "random" else None
                )
                module.num_heads = num_heads
                module.head_dim  = head_dim
            assigned += 1
        elif "ff.net" in layer_suffix and layer_suffix.endswith(".proj"):
            basis_key = f"layer.{block_idx}.ffn"
            evec = basis_dict[basis_key].float()
            evals_key = f"{basis_key}.eigenvalues"
            if _use_perm(layer_suffix) and evals_key in basis_dict:
                module.perm_idx = perm_idx_from_eigendecomp(evec, basis_dict[evals_key].float())
                n_perm += 1
            else:
                module.rotation = _residual_basis(
                    basis_key, evec, R1 if residual_rotation == "random" else None, "hidden"
                )
            assigned += 1
        elif "ff.net" in layer_suffix and layer_suffix.endswith(".2"):
            if use_had:
                # Hadamard rotation: O(D log D) instead of O(D²)
                low_dim = 1 << (intermediate_dim.bit_length() - 1)  # 4096 for D=4608
                module.use_hadamard = True
                module.hadamard_low_dim = low_dim
                if block_idx in sign_flips_dict:
                    module.hadamard_sign_flips = sign_flips_dict[block_idx]
                else:
                    module.hadamard_sign_flips = generate_sign_flips(low_dim, seed=42 + block_idx)
                hadamard_count += 1
            else:
                basis_key = f"layer.{block_idx}.ffn.down_proj"
                evec = basis_dict[basis_key].float()
                evals_key = f"{basis_key}.eigenvalues"
                if _use_perm(layer_suffix) and evals_key in basis_dict:
                    module.perm_idx = perm_idx_from_eigendecomp(evec, basis_dict[evals_key].float())
                    n_perm += 1
                else:
                    module.rotation = _residual_basis(
                        basis_key, evec,
                        R_down if residual_rotation == "random" else None,
                        "down",
                    )
            assigned += 1

    print(f"Assigned rotations to {assigned} ActQuantWrapper layers "
          f"({n_perm} perm_idx, {hadamard_count} Hadamard, "
          f"{assigned - n_perm - hadamard_count} PCA rotation, "
          f"residual_rotation={residual_rotation}).")
    if shared_map:
        print(f"Shared PCA basis: {len(set(shared_map.values()))} canonical groups, "
              f"{len(rotation_cache)} materialized post-R rotations.")
    return assigned


def configure_quantizers_by_name(transformer, high_len_hidden, high_len_head, cfg,
                                 nvfp4=False, hadamard_layers=None, a_groupsize=None,
                                 high_len_down=0, skip_quant_layers=None,
                                 activation_format="nvfp4", format_stats=None):
    """Configure mixed-precision activation quantizers by PixArt-Sigma layer type.

    When nvfp4=False (INT4):
      - QKV and FFN up-proj: groupsize=64, asymmetric
      - attn to_out: groupsize=head_dim (per head per token), asymmetric
      - FFN down-proj: groupsize=64, no mixed-precision

    When nvfp4=True (FP4):
      - All layers: groupsize=16 (except to_out: groupsize=72/head_dim), symmetric
      - ``activation_format`` selects legacy E2M1 or a hardware-faithful
        fixed/mixed FP4 mode; weight quantization and rotation are unchanged
    """
    a_bits = cfg["quantization"]["a_bits"]
    head_dim = cfg["dims"]["head"]

    if nvfp4:
        nvfp4_cfg = cfg.get("nvfp4", {})
        a_gs = nvfp4_cfg.get("a_groupsize", 16)
        a_gs_out = nvfp4_cfg.get("a_groupsize_attn_out", head_dim)
        allowed_formats = {
            "nvfp4", "nvfp4-hw", "e0m3", "block-mix-oracle",
            "tile-mix-oracle", "tile-mix-output-oracle", "a16w4-residual",
            "e0a-w16-residual",
            "nvfp4-4over6", "e0m3-gscale1536",
            "tile-mix-e0-e2-4over6",
        }
        if activation_format not in allowed_formats:
            raise ValueError(f"unsupported PixArt activation format: {activation_format}")
        qdt = activation_format
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
        module.quantizer.format_stats = (
            format_stats.for_layer(name)
            if format_stats is not None and hasattr(format_stats, "for_layer")
            else format_stats
        )

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

        # perm_idx layers use flat hidden-dim layout: high_len_hidden + a_gs groupsize
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
