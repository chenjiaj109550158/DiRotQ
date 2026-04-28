"""
mixed_linear.py

Real-kernel forward pass for DiRotQ ActQuantWrapper layers.

The fake-quantization forward in `dirotq_fused_unrotation_fast.py` does:
    x_rot   = x @ U                           (fp16/bf16 rotation)
    find_params + STE quantize x_rot          (fp16 in/out — fake)
    y       = self.module(x)                  (fp16 GEMM with fused W)

For the real-kernel path, we replace the fake quantize+matmul step with:

    Layer split (layer-time, once):
        W_low_q = self.module.weight[:, :D-h]    # int4 grid in fp/bf
        W_tail  = self.module.weight[:, D-h:]    # fp passthrough
        Repack W_low_q to int4 + group scales (torch._weight_int4pack_mm)
                                       OR int4 packed bytes (Triton W4A4).
    Forward:
        x_rot   = x @ U
        x_low   = x_rot[..., :D-h]
        x_high  = x_rot[..., D-h:]
        y_low   = real_int4_gemm(x_low, W_low_packed)     # W4A16 or W4A4
        y_high  = x_high @ W_tail.T                       # fp passthrough
        y       = y_low + y_high + bias

This module exposes a single entry point `patch_forward_real(transformer,
backend, device)` that walks the transformer, prepares packed weights
for every eligible ActQuantWrapper, and rebinds `forward` to a real-kernel
implementation. Layers that can't be handled (e.g. per-head rotation_per_head
or Hadamard) fall back to the existing fake-quant path.
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

_DIROTQ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIROTQ_ROOT not in sys.path:
    sys.path.insert(0, _DIROTQ_ROOT)

from utils.quant_utils import ActQuantWrapper  # noqa: E402

from .kernels import torch_int4 as ti4  # noqa: E402
from .kernels import triton_w4a4 as tt4  # noqa: E402


# ---------------------------------------------------------------------------
# Per-layer setup: extract low/tail, repack, attach to module
# ---------------------------------------------------------------------------

def _eligible_for_real_kernel(mod: ActQuantWrapper, *, backend: str,
                              w_groupsize: int) -> tuple[bool, str]:
    """Return (eligible, reason). Per-head and Hadamard layers stay fake."""
    if mod.quantizer.bits >= 16:
        return False, "no quantization (bits=16)"
    if not getattr(mod, "_unrot_fused", False):
        return False, "not fused (no rotation)"
    if mod.rotation_per_head is not None:
        return False, "per-head rotation (interleaved layout)"
    if getattr(mod, "use_hadamard", False):
        return False, "Hadamard layer"
    if mod.rotation is None:
        return False, "no rotation matrix"

    in_f = mod.module.weight.shape[1]
    hlen = int(mod.quantizer.high_bits_length)
    n_low = in_f - hlen
    if n_low <= 0:
        return False, "no low region"
    if n_low % w_groupsize != 0:
        return False, f"low width {n_low} not multiple of group_size {w_groupsize}"
    if backend == "triton":
        # The W4A4 kernel needs even K (two int4 per byte) and a multiple of
        # the activation group size (we reuse w_groupsize for activations).
        if n_low % 2 != 0:
            return False, f"low width {n_low} not even"
    return True, ""


def _prepare_layer_real(mod: ActQuantWrapper, *, backend: str, w_groupsize: int,
                        device: str) -> bool:
    """Prepare packed weights + tail for one layer. Returns True if real."""
    eligible, reason = _eligible_for_real_kernel(
        mod, backend=backend, w_groupsize=w_groupsize)
    if not eligible:
        mod._real_kernel = False
        mod._real_reason = reason
        return False

    W = mod.module.weight.data
    bias = mod.module.bias.data if mod.module.bias is not None else None
    in_f = W.shape[1]
    hlen = int(mod.quantizer.high_bits_length)
    n_low = in_f - hlen

    W_low_fp = W[:, :n_low].to(device).contiguous()
    W_tail = W[:, n_low:].to(device).contiguous() if hlen > 0 else None

    if backend == "torch":
        packed = ti4.pack_weight_int4(W_low_fp, w_groupsize)
        mod._real_backend = "torch"
        mod._real_packed = packed
        mod._real_w_tail = W_tail
        mod._real_bias = bias.to(device) if bias is not None else None
        mod._real_n_low = n_low
        mod._real_group_size = w_groupsize
    elif backend == "triton":
        if not tt4.is_supported():
            mod._real_kernel = False
            mod._real_reason = "triton not installed"
            return False
        w_packed, w_scales = tt4.pack_w4(W_low_fp, w_groupsize)
        mod._real_backend = "triton"
        mod._real_w_packed = w_packed.contiguous()
        mod._real_w_scales = w_scales.contiguous()
        mod._real_w_tail = W_tail
        mod._real_bias = bias.to(device) if bias is not None else None
        mod._real_n_low = n_low
        mod._real_out_features = W.shape[0]
        mod._real_group_size = w_groupsize
        # Pre-compute tail dtype for output cast
        mod._real_out_dtype = W.dtype
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    # Free the fp16 fused weight to reclaim memory — we only need W_tail and
    # the packed int4 representation from here on.
    mod.module.weight.data = torch.empty(0, dtype=W.dtype, device=device)
    if mod.module.bias is not None:
        mod.module.bias.data = torch.empty(0, dtype=bias.dtype, device=device)

    mod._real_kernel = True
    return True


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _real_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in forward for ActQuantWrapper that uses a real low-bit kernel.

    Falls back to the fake-quant fused forward for layers we couldn't pack
    (per-head, Hadamard, etc.).
    """
    if not getattr(self, "_real_kernel", False):
        return _fake_fused_forward(self, x)

    x_dtype = x.dtype

    # ---- Forward rotation (always required so the layer sees rotated x) ----
    init_shape = x.shape
    x_flat = x.reshape(-1, init_shape[-1])
    x_rot = (x_flat @ self.rotation).reshape(init_shape)

    n_low = self._real_n_low
    x_low = x_rot[..., :n_low].contiguous()
    x_high = x_rot[..., n_low:].contiguous() if x_rot.shape[-1] > n_low else None

    if self._real_backend == "torch":
        # W4A16: activations stay fp16/bf16
        y_low = ti4.int4_gemm(x_low, self._real_packed)
    elif self._real_backend == "triton":
        # W4A4: quantize activation low region to int4, run kernel
        gs = self._real_group_size
        leading = x_low.shape[:-1]
        K = x_low.shape[-1]
        M = 1
        for d in leading:
            M *= d
        a_packed, a_scales = tt4.quantize_act_int4(x_low, gs)
        y_low_2d = tt4.triton_w4a4_gemm(
            a_packed, a_scales,
            self._real_w_packed, self._real_w_scales,
            M=M, N=self._real_out_features, K=K,
            group_size=gs,
            out_dtype=self._real_out_dtype,
        )
        y_low = y_low_2d.reshape(*leading, self._real_out_features)
    else:
        raise RuntimeError(f"Unknown backend {self._real_backend}")

    if x_high is not None and self._real_w_tail is not None and x_high.shape[-1] > 0:
        y_high = torch.matmul(x_high.to(x_dtype), self._real_w_tail.t().to(x_dtype))
        y = y_low.to(x_dtype) + y_high
    else:
        y = y_low.to(x_dtype)

    if self._real_bias is not None:
        y = y + self._real_bias.to(x_dtype)

    return y


# Fake-quant fallback — copy of the fast path for layers we can't pack.
def _fake_fused_forward(self, x: torch.Tensor) -> torch.Tensor:
    from utils.hadamard_utils import fast_hadamard_transform

    x_dtype = x.dtype

    if getattr(self, "_unrot_fused", False) and self.quantizer.bits < 16:

        if self.rotation is not None:
            init_shape = x.shape
            x_flat = x.reshape(-1, init_shape[-1])
            x_rot = (x_flat @ self.rotation).reshape(init_shape)
            self.quantizer.find_params(x_rot)
            x = self.quantizer(x_rot).to(x_dtype)

        elif self.rotation_per_head is not None:
            B_T = x.shape[:-1]
            H, d = self.num_heads, self.head_dim
            hlen = self.quantizer.high_bits_length
            d_q = d - hlen
            x_heads = x.reshape(*B_T, H, d)
            x_rot_heads = torch.einsum(
                '...hd,hde->...he', x_heads, self.rotation_per_head)
            if hlen > 0:
                x_rearranged = torch.cat([
                    x_rot_heads[..., :d_q].reshape(*B_T, H * d_q),
                    x_rot_heads[..., d_q:].reshape(*B_T, H * hlen),
                ], dim=-1)
                saved_hlen = self.quantizer.high_bits_length
                saved_gs = self.quantizer.groupsize
                self.quantizer.high_bits_length = H * hlen
                self.quantizer.groupsize = d_q if d_q > 0 else -1
                self.quantizer.find_params(x_rearranged)
                x_quant = self.quantizer(x_rearranged)
                self.quantizer.high_bits_length = saved_hlen
                self.quantizer.groupsize = saved_gs
                x_q_heads = x_quant[..., :H * d_q].reshape(*B_T, H, d_q)
                x_h_heads = x_quant[..., H * d_q:].reshape(*B_T, H, hlen)
                x = torch.cat([x_q_heads, x_h_heads], dim=-1).reshape(
                    *B_T, H * d).to(x_dtype)
            else:
                x_rot_flat = x_rot_heads.reshape(*B_T, H * d)
                self.quantizer.find_params(x_rot_flat)
                x = self.quantizer(x_rot_flat).to(x_dtype)

        elif getattr(self, "use_hadamard", False):
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

    return self.module(x).to(x_dtype)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def patch_forward_real(transformer: nn.Module, *, backend: str = "torch",
                       device: str = "cuda") -> dict:
    """Walk transformer, repack eligible layers, swap forward to the real path.

    Returns a stats dict: {"real": int, "fake": int, "skipped_reasons": dict}.
    Also pre-converts rotation matrices to compute dtype on `device`.
    """
    assert backend in ("torch", "triton")

    # Pre-convert rotations (fp16/bf16) to device — same trick as the fast path.
    compute_dtype = next(transformer.parameters()).dtype
    n_rot = 0
    for _, mod in transformer.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue
        if mod.rotation is not None:
            mod.rotation = mod.rotation.to(device=device, dtype=compute_dtype)
            n_rot += 1
        elif mod.rotation_per_head is not None:
            mod.rotation_per_head = mod.rotation_per_head.to(
                device=device, dtype=compute_dtype)
            n_rot += 1
        if getattr(mod, "hadamard_sign_flips", None) is not None:
            mod.hadamard_sign_flips = mod.hadamard_sign_flips.to(
                device=device, dtype=compute_dtype)
    print(f"Preconverted {n_rot} rotation matrices to {compute_dtype} on {device}.")

    # Pull w_groupsize from any quantizer (they share it).
    w_groupsize = 64  # DiRotQ INT4 default
    for _, mod in transformer.named_modules():
        if isinstance(mod, ActQuantWrapper) and mod.quantizer.bits < 16:
            w_groupsize = max(w_groupsize, int(mod.quantizer.groupsize)
                              if mod.quantizer.groupsize > 0 else w_groupsize)
            break

    n_real = 0
    n_fake = 0
    reasons: dict[str, int] = {}
    for name, mod in transformer.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue
        ok = _prepare_layer_real(
            mod, backend=backend, w_groupsize=w_groupsize, device=device)
        if ok:
            n_real += 1
        else:
            n_fake += 1
            reason = getattr(mod, "_real_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1

    ActQuantWrapper.forward = _real_forward
    print(f"Patched ActQuantWrapper.forward with {backend!r} real-kernel path.")
    print(f"  Real-kernel layers: {n_real}")
    print(f"  Fake-quant fallback: {n_fake} (reasons: {reasons})")

    return {"real": n_real, "fake": n_fake, "skipped_reasons": reasons,
            "backend": backend}
