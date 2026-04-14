"""
dirotq_fused_unrotation_fast.py

Optimized version of dirotq_fused_unrotation.py with two speed improvements:

1. fp16 rotations: rotation matrices stored in fp16 on GPU, eliminating per-call
   fp32 casts and getting ~2x faster matmuls. Safe because the fused path only does
   forward rotation -> quantize (no unrotation accumulation).

2. Cached quantizer buffers: skip free() after each call so scale/zero tensors
   are reused instead of deallocated and reallocated every forward pass.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.quant_utils import ActQuantWrapper
from utils.hadamard_utils import fast_hadamard_transform


def preconvert_rotations_to_fp16(transformer, device="cuda"):
    """Convert all rotation matrices to fp16 on GPU once, before generation."""
    n = 0
    for name, mod in transformer.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue

        if mod.rotation is not None:
            mod.rotation = mod.rotation.to(device=device, dtype=torch.float16)
            n += 1
        elif mod.rotation_per_head is not None:
            mod.rotation_per_head = mod.rotation_per_head.to(device=device, dtype=torch.float16)
            n += 1

        if getattr(mod, 'hadamard_sign_flips', None) is not None:
            mod.hadamard_sign_flips = mod.hadamard_sign_flips.to(device=device, dtype=torch.float16)

    print(f"Preconverted {n} rotation matrices to fp16 on {device}.")
    return n


def _fused_forward_fast(self, x):
    """
    Drop-in replacement for ActQuantWrapper.forward — fp16 rotations, no free().

    Changes vs. _fused_forward:
      - Rotation matrices already fp16 on GPU (no .to() per call)
      - Sign flips already fp16 on GPU (no .to() per call)
      - quantizer.free() removed (buffers reused across calls)
    """
    x_dtype = x.dtype

    if getattr(self, '_unrot_fused', False) and self.quantizer.bits < 16:

        if self.rotation is not None:
            init_shape = x.shape
            x_flat = x.reshape(-1, init_shape[-1])
            x_rot = (x_flat @ self.rotation).reshape(init_shape)
            self.quantizer.find_params(x_rot)
            x = self.quantizer(x_rot).to(x_dtype)

        elif self.rotation_per_head is not None:
            B_T = x.shape[:-1]
            H, d = self.num_heads, self.head_dim
            x_heads = x.reshape(*B_T, H, d)
            x_rot_flat = torch.einsum('...hd,hde->...he', x_heads, self.rotation_per_head).reshape(*B_T, H * d)
            self.quantizer.find_params(x_rot_flat)
            x = self.quantizer(x_rot_flat).to(x_dtype)

        elif getattr(self, 'use_hadamard', False):
            init_shape = x.shape
            D = init_shape[-1]
            low_dim = self.hadamard_low_dim

            if low_dim < D:
                x_low = x[..., :low_dim]
                x_high = x[..., low_dim:]
            else:
                x_low = x
                x_high = None

            if self.hadamard_sign_flips is not None:
                x_low = x_low * self.hadamard_sign_flips
            x_rot_low = fast_hadamard_transform(x_low.float()).to(x_dtype)

            if x_high is not None:
                x_rot = torch.cat([x_rot_low, x_high], dim=-1)
            else:
                x_rot = x_rot_low

            self.quantizer.find_params(x_rot)
            x = self.quantizer(x_rot).to(x_dtype)

    else:
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)

    x = self.module(x).to(x_dtype)

    if self.out_quantizer.bits < 16:
        self.out_quantizer.find_params(x)
        x = self.out_quantizer(x).to(x_dtype)

    return x


def patch_forward_fast():
    """Monkey-patch ActQuantWrapper.forward with the fast fused version."""
    ActQuantWrapper.forward = _fused_forward_fast
    print("Patched ActQuantWrapper.forward with fast fused version (fp16 rotations, no free).")
