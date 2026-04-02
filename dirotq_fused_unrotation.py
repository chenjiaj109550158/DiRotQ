"""
dirotq_fused_unrotation.py

Optimized DiRotQ generation: pre-absorb the unrotation matrices into linear
layer weights, eliminating the per-forward-pass unrotation matmul.

Math:
  Original: x_rot = x @ U → quantize → x_unrot = x_quant @ U.T → y = x_unrot @ W.T
  Fused:    x_rot = x @ U → quantize →                          → y = x_quant @ W_fused.T

  where W_fused = W @ U  (precomputed once, same result since x_quant @ U.T @ W.T
  = x_quant @ (W @ U).T)

This removes 1 matmul per wrapped layer per timestep (the unrotation step).

The forward rotation (x @ U) is still computed at each forward pass because
activation quantization calibration (find_params) needs the rotated activations.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.quant_utils import ActQuantWrapper
from utils.hadamard_utils import fast_hadamard_transform

# ---------------------------------------------------------------------------
# Weight fusion
# ---------------------------------------------------------------------------

def fuse_unrotation_into_weights(transformer):
    """
    Pre-absorb the unrotation step into each linear layer's weight matrix.

    For standard rotation (hidden-dim):
        W_fused = W @ U   [out, in] = [out, in] @ [in, in]
        Forward:  x_quant @ W_fused.T  (skip x_quant @ U.T)

    For per-head rotation (attention out_proj):
        W_fused = W @ BlockDiag(U_ph)
        Computed as: einsum('ohd,hde->ohe', W.reshape(out,H,d), U_ph).reshape(out,H*d)
        Forward:  x_quant_flat @ W_fused.T  (skip per-head unrotation einsum)

    After fusion the mod.rotation / mod.rotation_per_head are kept (still
    needed for the FORWARD rotation x @ U).  A flag _unrot_fused is set so
    the patched forward knows to skip the unrotation.
    """
    n = 0
    for name, mod in transformer.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue

        W = mod.module.weight.data          # [out_features, in_features]

        if mod.rotation is not None and mod.quantizer.bits < 16:
            # W_fused = W @ U
            # Proof: x_quant @ U.T @ W.T = x_quant @ (W @ U).T = x_quant @ W_fused.T
            U = mod.rotation.to(device=W.device, dtype=torch.float32)  # [D, D]
            W_fused = W.float() @ U                   # [out, D] @ [D, D] → [out, D]
            mod.module.weight.data = W_fused.to(W.dtype)
            mod._unrot_fused = True
            n += 1

        elif mod.rotation_per_head is not None and mod.quantizer.bits < 16:
            # W_fused = W @ BlockDiag(U_ph)
            # BlockDiag(U_ph) = block-diagonal matrix with U_ph[h] on diagonal h
            # Efficient: reshape W to [out, H, d], einsum with U_ph [H, d, d]
            U_ph = mod.rotation_per_head.to(device=W.device, dtype=torch.float32)  # [H, d, d]
            H, d = mod.num_heads, mod.head_dim
            out_f = W.shape[0]
            W_3d = W.float().reshape(out_f, H, d)                 # [out, H, d]
            W_fused_3d = torch.einsum('ohd,hde->ohe', W_3d, U_ph) # [out, H, d]
            mod.module.weight.data = W_fused_3d.reshape(out_f, H * d).to(W.dtype)
            mod._unrot_fused = True
            n += 1

        elif getattr(mod, 'use_hadamard', False) and mod.quantizer.bits < 16:
            # Hadamard fusion: build dense inverse-Hadamard matrix for the low_dim part,
            # then fuse into weights.  Forward rotation uses fast O(D log D) transform;
            # only this one-time fusion uses the dense matrix.
            #
            # Inverse Hadamard = diag(sign_flips) @ H_dense  (H is self-inverse)
            # W_fused = W @ BlockDiag(inv_H_low, I_high)
            low_dim = mod.hadamard_low_dim
            D = W.shape[1]
            high_dim = D - low_dim

            # Build dense normalized Hadamard matrix for low_dim (power of 2)
            H_dense = torch.eye(low_dim, dtype=torch.float32, device=W.device)
            H_dense = fast_hadamard_transform(H_dense)  # each row is H @ e_i

            # Apply sign flips: inv_H = diag(s) @ H
            if mod.hadamard_sign_flips is not None:
                sf = mod.hadamard_sign_flips.to(W.device, dtype=torch.float32)
                H_dense = sf.unsqueeze(0) * H_dense  # broadcast: [low, low]

            # Fuse: W_low_fused = W[:, :low_dim] @ H_dense.T
            W_f = W.float()
            W_low = W_f[:, :low_dim] @ H_dense.T
            if high_dim > 0:
                W_fused = torch.cat([W_low, W_f[:, low_dim:]], dim=1)
            else:
                W_fused = W_low
            mod.module.weight.data = W_fused.to(W.dtype)
            mod._unrot_fused = True
            n += 1

    print(f"Fused unrotation into weights for {n} layers.")
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
            U = self.rotation.to(x.device, dtype=torch.float32)
            x_fp32 = x.float().reshape(-1, init_shape[-1])
            x_rot = (x_fp32 @ U).reshape(init_shape)          # fp32
            self.quantizer.find_params(x_rot)
            x = self.quantizer(x_rot).to(x_dtype)              # single fp16 cast
            self.quantizer.free()
            # x_quant @ U.T step removed — W already contains U

        elif self.rotation_per_head is not None:
            # Per-head forward rotation → quantize → (unrotation SKIPPED)
            B_T = x.shape[:-1]
            H, d = self.num_heads, self.head_dim
            U_ph = self.rotation_per_head.to(x.device, dtype=torch.float32)
            x_heads = x.float().reshape(*B_T, H, d)
            x_rot_flat = torch.einsum('...hd,hde->...he', x_heads, U_ph).reshape(*B_T, H * d)
            self.quantizer.find_params(x_rot_flat)
            x = self.quantizer(x_rot_flat).to(x_dtype)         # single fp16 cast
            self.quantizer.free()
            # per-head einsum unrotation removed — W already contains BlockDiag(U_ph)

        elif getattr(self, 'use_hadamard', False):
            # Hadamard forward rotation → quantize → (inverse SKIPPED, absorbed in W)
            # O(D log D) via fast Walsh-Hadamard on the low_dim part
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
                sf = self.hadamard_sign_flips.to(x_low.device, dtype=x_low.dtype)
                x_low = x_low * sf
            x_rot_low = fast_hadamard_transform(x_low)

            if x_high is not None:
                x_rot = torch.cat([x_rot_low, x_high], dim=-1)
            else:
                x_rot = x_rot_low

            self.quantizer.find_params(x_rot)
            x = self.quantizer(x_rot).to(x_dtype)
            self.quantizer.free()
            # Inverse Hadamard removed — W already contains it

    else:
        # No fusion (bits=16, or no rotation): standard activation quantization
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

    # Linear forward (uses fused weight when _unrot_fused=True)
    x = self.module(x).to(x_dtype)

    # Output quantization (unused in current DiRotQ setup, kept for completeness)
    if self.out_quantizer.bits < 16:
        self.out_quantizer.find_params(x)
        x = self.out_quantizer(x).to(x_dtype)
        self.out_quantizer.free()

    return x


def patch_forward():
    """Monkey-patch ActQuantWrapper.forward with the fused version."""
    ActQuantWrapper.forward = _fused_forward
    print("Patched ActQuantWrapper.forward to use fused unrotation.")
