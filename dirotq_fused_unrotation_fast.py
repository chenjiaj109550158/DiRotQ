"""
dirotq_fused_unrotation_fast.py

Optimized version of dirotq_fused_unrotation.py with two speed improvements:

1. fp16/bf16 rotations: rotation matrices stored in the model's compute
   dtype (fp16 or bf16) on GPU, eliminating per-call fp32 casts and getting
   ~2x faster matmuls. Safe because the fused path only does forward rotation
   -> quantize (no unrotation accumulation).

2. Cached quantizer buffers: skip free() after each call so scale/zero tensors
   are reused instead of deallocated and reallocated every forward pass.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.quant_utils import ActQuantWrapper
from utils.hadamard_utils import fast_hadamard_transform


def preconvert_rotations_to_device(transformer, device="cuda", dtype=None):
    """Convert all rotation matrices to the model's compute dtype on GPU once.

    If dtype is None, infer from the transformer's parameters.
    """
    if dtype is None:
        dtype = next(transformer.parameters()).dtype

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
        # Skipped bits=16 wrappers bypass every rotation branch in forward.
        # Do not materialize their otherwise-unused basis on the GPU.
        if mod.quantizer.bits >= 16:
            continue

        if mod.rotation is not None:
            mod.rotation = _convert_once(mod.rotation, dtype)
            n += 1
        elif mod.rotation_per_head is not None:
            mod.rotation_per_head = _convert_once(mod.rotation_per_head, dtype)
            n += 1
        elif getattr(mod, 'perm_idx', None) is not None:
            # perm_idx is int64 — move to device but keep dtype (no fp conversion)
            mod.perm_idx = mod.perm_idx.to(device=device)
            n += 1

        if getattr(mod, 'hadamard_sign_flips', None) is not None:
            mod.hadamard_sign_flips = mod.hadamard_sign_flips.to(device=device, dtype=dtype)

    print(f"Preconverted {n} rotation/permutation assignments to {device} "
          f"using {len(converted)} unique tensor storages.")
    return n


def _fused_forward_fast(self, x):
    """
    Drop-in replacement for ActQuantWrapper.forward — fp16/bf16 rotations, no free().

    Changes vs. _fused_forward:
      - Rotation matrices already in compute dtype on GPU (no .to() per call)
      - Sign flips already in compute dtype on GPU (no .to() per call)
      - quantizer.free() removed (buffers reused across calls)
    """
    x_dtype = x.dtype

    if getattr(self, '_unrot_fused', False) and self.quantizer.bits < 16:

        if self.rotation is not None:
            init_shape = x.shape
            x_flat = x.reshape(-1, init_shape[-1])
            x_rot = (x_flat @ self.rotation).reshape(init_shape)
            self.quantizer.find_params(x_rot)
            x = self.quantize_activation_for_linear(x_rot).to(x_dtype)

        elif self.rotation_per_head is not None:
            B_T = x.shape[:-1]
            H, d = self.num_heads, self.head_dim
            hlen = self.quantizer.high_bits_length  # fp16/bf16 channels per head 
            d_q  = d - hlen                         # 4-bit channels per head 
            x_heads = x.reshape(*B_T, H, d)
            x_rot_heads = torch.einsum('...hd,hde->...he', x_heads, self.rotation_per_head)
            if hlen > 0:
                # Rearrange so bf16/fp16 channels are contiguous at end: [*B_T, H*d_q | H*hlen]
                x_rearranged = torch.cat([
                    x_rot_heads[..., :d_q].reshape(*B_T, H * d_q),
                    x_rot_heads[..., d_q:].reshape(*B_T, H * hlen),
                ], dim=-1)
                saved_hlen = self.quantizer.high_bits_length
                saved_gs   = self.quantizer.groupsize
                self.quantizer.high_bits_length = H * hlen
                self.quantizer.groupsize        = d_q if d_q > 0 else -1
                self.quantizer.find_params(x_rearranged)
                x_quant = self.quantize_activation_for_linear(x_rearranged)
                self.quantizer.high_bits_length = saved_hlen
                self.quantizer.groupsize        = saved_gs
                # Re-interleave: [*B_T, H*d_q | H*hlen] -> [*B_T, H*d]
                x_q_heads = x_quant[..., :H * d_q].reshape(*B_T, H, d_q)
                x_h_heads = x_quant[..., H * d_q:].reshape(*B_T, H, hlen)
                x = torch.cat([x_q_heads, x_h_heads], dim=-1).reshape(*B_T, H * d).to(x_dtype)
            else:
                x_rot_flat = x_rot_heads.reshape(*B_T, H * d)
                self.quantizer.find_params(x_rot_flat)
                x = self.quantize_activation_for_linear(x_rot_flat).to(x_dtype)

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
            x = self.quantize_activation_for_linear(x_rot).to(x_dtype)

        elif self.perm_idx is not None:
            # PCA-only permutation: O(D) gather, weight already pre-permuted.
            x = x[..., self.perm_idx]
            self.quantizer.find_params(x)
            x = self.quantize_activation_for_linear(x).to(x_dtype)

    else:
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantize_activation_for_linear(x).to(x_dtype)

    return self.module(x).to(x_dtype)


def patch_forward_fast():
    """Monkey-patch ActQuantWrapper.forward with the fast fused version."""
    ActQuantWrapper.forward = _fused_forward_fast
    print("Patched ActQuantWrapper.forward with fast fused version (fp16/bf16 rotations, no free).")
