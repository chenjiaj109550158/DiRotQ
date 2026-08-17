"""
dirotq_fused_unrotation.py

Optimized DiRotQ generation: the unrotation matrix is absorbed into each
linear layer's weight during weight quantization itself (see
utils.quant_utils._rotate_and_split_W), so the per-forward-pass unrotation
matmul disappears. This module provides the matching forward hook that
skips the unrotation step and reads the already-rotated weight.

Math:
  Original: x_rot = x @ U → quantize → x_unrot = x_quant @ U.T → y = x_unrot @ W.T
  Fused:    x_rot = x @ U → quantize →                          → y = x_quant @ W_fused.T

  where W_fused is built during rtn/gptq quantization as
  Q(W_rot[:, :low]) stitched with W_rot[:, tail] in fp16.

The forward rotation (x @ U) is still computed at each forward pass because
activation quantization calibration (find_params) needs the rotated
activations.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.quant_utils import ActQuantWrapper
from utils.hadamard_utils import fast_hadamard_transform


# ---------------------------------------------------------------------------
# Pre-convert rotations to fp32 on GPU (avoids redundant .to() every call)
# ---------------------------------------------------------------------------

def preconvert_rotations_to_device(transformer, device="cuda"):
    """Convert all rotation matrices and sign flips to fp32 on GPU once."""
    n = 0
    converted = {}

    def _convert_once(tensor, target_dtype):
        storage = tensor.untyped_storage()
        key = (
            tensor.device.type, tensor.device.index, storage.data_ptr(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(target_dtype), str(device),
        )
        if key not in converted:
            converted[key] = tensor.to(device=device, dtype=target_dtype)
        return converted[key]

    for name, mod in transformer.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue
        # A bits=16 wrapper takes the ordinary Linear path and never reads
        # its assigned rotation.  Keeping a skipped layer's (notably the
        # PixArt 4608x4608 ff.net.2) basis on CPU avoids charging dead storage
        # to the generation-time rotation footprint.
        if mod.quantizer.bits >= 16:
            continue
        if mod.rotation is not None:
            mod.rotation = _convert_once(mod.rotation, torch.float32)
            n += 1
        elif mod.rotation_per_head is not None:
            mod.rotation_per_head = _convert_once(mod.rotation_per_head, torch.float32)
            n += 1
        elif getattr(mod, 'perm_idx', None) is not None:
            # perm_idx is int64 — move to device but keep dtype
            mod.perm_idx = mod.perm_idx.to(device=device)
            n += 1
        if getattr(mod, 'hadamard_sign_flips', None) is not None:
            mod.hadamard_sign_flips = mod.hadamard_sign_flips.to(device=device, dtype=torch.float32)
    print(f"Preconverted {n} rotation/permutation assignments to fp32/int64 on "
          f"{device} using {len(converted)} unique tensor storages.")
    return n


# ---------------------------------------------------------------------------
# Patched forward (skips unrotation, uses fused weight directly)
# ---------------------------------------------------------------------------

def _fused_forward(self, x):
    """
    Drop-in replacement for ActQuantWrapper.forward when unrotation is fused.

    Changes vs. original:
      - Standard rotation:   skips  x_unrot = x_quant @ U.T
      - Per-head rotation:   skips  einsum unrotation
      - Everything else:     identical to original
    """
    x_dtype = x.dtype

    if getattr(self, '_unrot_fused', False) and self.quantizer.bits < 16:

        if self.rotation is not None:
            # Forward rotation → quantize → (unrotation SKIPPED, absorbed in W)
            init_shape = x.shape
            x_fp32 = x.float().reshape(-1, init_shape[-1])
            x_rot = (x_fp32 @ self.rotation).reshape(init_shape)
            if self.quantizer.bits < 16:
                self.quantizer.find_params(x_rot)
                x = self.quantize_activation_for_linear(x_rot).to(x_dtype)
            else:
                x = x_rot.to(x_dtype)

        elif self.rotation_per_head is not None:
            # Per-head forward rotation → quantize → (unrotation SKIPPED)
            B_T = x.shape[:-1]
            H, d = self.num_heads, self.head_dim
            x_heads = x.float().reshape(*B_T, H, d)
            x_rot_flat = torch.einsum('...hd,hde->...he', x_heads, self.rotation_per_head).reshape(*B_T, H * d)
            if self.quantizer.bits < 16:
                self.quantizer.find_params(x_rot_flat)
                x = self.quantize_activation_for_linear(x_rot_flat).to(x_dtype)
            else:
                x = x_rot_flat.to(x_dtype)

        elif getattr(self, 'use_hadamard', False):
            # Hadamard forward rotation → quantize → (inverse SKIPPED, absorbed in W)
            init_shape = x.shape
            D = init_shape[-1]
            low_dim = self.hadamard_low_dim
            x_fp32 = x.float()

            if low_dim < D:
                x_low = x_fp32[..., :low_dim]
                x_high = x_fp32[..., low_dim:]
            else:
                x_low = x_fp32
                x_high = None

            if self.hadamard_sign_flips is not None:
                x_low = x_low * self.hadamard_sign_flips
            x_rot_low = fast_hadamard_transform(x_low)

            if x_high is not None:
                x_rot = torch.cat([x_rot_low, x_high], dim=-1)
            else:
                x_rot = x_rot_low

            if self.quantizer.bits < 16:
                self.quantizer.find_params(x_rot)
                x = self.quantize_activation_for_linear(x_rot).to(x_dtype)
            else:
                x = x_rot.to(x_dtype)

        elif getattr(self, 'perm_idx', None) is not None:
            # PCA-only permutation: O(D) gather, weight already pre-permuted.
            x_perm = x[..., self.perm_idx]
            if self.quantizer.bits < 16:
                self.quantizer.find_params(x_perm)
                x = self.quantize_activation_for_linear(x_perm).to(x_dtype)
            else:
                x = x_perm.to(x_dtype)

        else:
            # Fused but no rotation assigned — just quantize if needed
            if self.quantizer.bits < 16:
                self.quantizer.find_params(x)
                x = self.quantize_activation_for_linear(x).to(x_dtype)

    else:
        # No fusion: standard activation quantization
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantize_activation_for_linear(x).to(x_dtype)

    return self.module(x).to(x_dtype)


def patch_forward():
    """Monkey-patch ActQuantWrapper.forward with the fused version."""
    ActQuantWrapper.forward = _fused_forward
    print("Patched ActQuantWrapper.forward to use fused unrotation.")
