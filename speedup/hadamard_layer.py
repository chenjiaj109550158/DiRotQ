"""
speedup/hadamard_layer.py

Hadamard-rotation building blocks for the speedup benchmarks.

What this gives us:
  - `hadamard_dims(K)` → (low_dim, hlen): the largest power-of-2 ≤ K is the
        FWHT-rotated low region; the K-low_dim residual is the fp16 tail.
  - `prepare_hadamard_layer(W, hlen=None, group_size=64, ...)` → dict of
        {sign_flips, W_low_packed, W_low_scales, W_tail, ...} ready for
        forward.  (Hadamard is folded into W at calibration: W_low =
        W[:, :low_dim] @ (diag(sf) @ FWHT_dense)^T, then int4-packed.)
  - `forward_hadamard_w4a4(x, packed)` → output tensor — the runtime path:
        x_low = sf ⊙ x[:, :low_dim]; x_low_rot = FWHT(x_low); int4-quantize;
        triton W4A4 GEMM; fp16 tail GEMM; sum.

Imports `fast_hadamard_transform` and `generate_sign_flips` from DiRotQ's
`utils/hadamard_utils.py` *read-only* — no edits to the main codebase.

Why a separate file:
  The existing DiRotQ code only attaches Hadamard to layer suffixes that are
  explicitly listed via `--hadamard-layers` (and only at the top of `apply_dirotq.py`'s
  pipeline). For our speedup study we want to use Hadamard for *all* rotated
  layers without touching the main flow, so we build a parallel layer
  primitive here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_DIROTQ_ROOT = Path(__file__).resolve().parent.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))

from utils.hadamard_utils import (  # noqa: E402  (read-only import)
    fast_hadamard_transform, generate_sign_flips,
)

from speedup.kernels import torch_int4 as ti4  # noqa: E402
from speedup.kernels import triton_w4a4 as tt4  # noqa: E402


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def hadamard_dims(K: int) -> tuple[int, int]:
    """Pick (low_dim, hlen) for K.

    low_dim = largest power of 2 ≤ K   (FWHT operates on the first low_dim chans)
    hlen    = K - low_dim              (kept fp16 — pure passthrough)

    For pixart-sigma:
        K=1152  → low_dim=1024, hlen=128
        K=4608  → low_dim=4096, hlen=512
    For flux-dev:
        K=3072  → low_dim=2048, hlen=1024     (24% kept fp16)
        K=12288 → low_dim=8192, hlen=4096     (33% kept fp16)
    Note for flux these tails are bigger than DiRotQ's PCA-derived 12.5%.
    If that's too lossy you can pad K up to a higher power of 2 by appending
    zero-channels (handled at caller); otherwise use block-Hadamard via
    `utils.hadamard_utils.hadamard_transform` (cost is the same).
    """
    if K <= 0:
        return 0, 0
    low_dim = 1 << (K.bit_length() - 1)
    if low_dim > K:
        low_dim //= 2
    return low_dim, K - low_dim


# ---------------------------------------------------------------------------
# Calibration: pre-rotate W and pack to int4
# ---------------------------------------------------------------------------

def prepare_hadamard_layer(
    W: torch.Tensor,
    *,
    sign_flips: torch.Tensor | None = None,
    group_size: int = 64,
    seed: int = 42,
    use_dense_matmul: bool = True,
) -> dict:
    """Pre-rotate W by Hadamard and quantize the low region to int4.

    W: [N, K] in fp16/bf16.
    sign_flips: optional ±1 tensor [low_dim] for randomized Hadamard. If None,
                generated deterministically from `seed`.
    use_dense_matmul: if True, runtime rotation uses a dense low_dim×low_dim
                Hadamard matrix via cuBLAS GEMM (cheap & well-tuned). If
                False, runtime uses PyTorch's torch.stack-based FWHT (slow
                — only useful for correctness checks since it allocates).

    Returns a dict with everything `forward_hadamard_w4a4` needs.
    """
    assert W.dim() == 2
    N, K = W.shape
    low_dim, hlen = hadamard_dims(K)
    if low_dim == 0:
        raise ValueError(f"K={K} too small for Hadamard")
    if low_dim % group_size != 0:
        raise ValueError(f"low_dim={low_dim} not divisible by group_size={group_size}")
    if low_dim % 2 != 0:
        raise ValueError(f"low_dim={low_dim} not even")

    device = W.device
    dtype = W.dtype

    if sign_flips is None:
        sign_flips = generate_sign_flips(low_dim, seed=seed).to(device=device, dtype=dtype)
    else:
        sign_flips = sign_flips.to(device=device, dtype=dtype)
        assert sign_flips.shape == (low_dim,)

    # Fold Hadamard into W on the low region.
    # The runtime applies sign_flips then FWHT to x_low. To get the same
    # output via a pre-rotated weight, we need:
    #     y = (x_low ⊙ sf) @ FWHT @ W_low_rot.T
    # So W_low_rot = (FWHT applied along K-axis to W_low) ⊙ sf along K-axis.
    # Build by applying FWHT to each row of W[:, :low_dim] then sign-flip.
    W_f32 = W.float()
    W_low_rotated = fast_hadamard_transform(W_f32[:, :low_dim].contiguous())
    W_low_rotated = W_low_rotated * sign_flips.float()  # diag(sf) along K
    W_low_rotated = W_low_rotated.to(dtype).contiguous()

    # int4 pack for the low region.
    w_packed, w_scales = tt4.pack_w4(W_low_rotated, group_size)

    # Pre-pack the int4-packed weight (also for torch._weight_int4pack_mm
    # — used in W4A16 variant).
    w_int4pack = ti4.pack_weight_int4(W_low_rotated, group_size)

    if hlen > 0:
        W_tail = W[:, low_dim:].contiguous()
    else:
        W_tail = None

    # Build the dense low_dim×low_dim Hadamard matrix once (with sign-flips on
    # the input side baked in). At runtime we'll do `x_low @ H_dense` via
    # cuBLAS — much faster than PyTorch's torch.stack-based FWHT.
    H_dense = None
    if use_dense_matmul:
        eye = torch.eye(low_dim, dtype=torch.float32, device=device)
        H_dense = fast_hadamard_transform(eye)         # rows = FWHT basis
        H_dense = H_dense * sign_flips.float().unsqueeze(0)  # diag(sf) @ FWHT, applied to the input
        # Transpose so x_low @ H_dense produces same result as
        # fast_hadamard_transform(sf ⊙ x_low) row-wise.
        H_dense = H_dense.t().contiguous().to(dtype)

    return {
        "sign_flips":     sign_flips,
        "H_dense":        H_dense,
        "W_low_w4a4":     {"packed": w_packed, "scales": w_scales},
        "W_low_w4a16":    w_int4pack,
        "W_tail":         W_tail,
        "low_dim":        low_dim,
        "hlen":           hlen,
        "group_size":     group_size,
        "N":              N,
        "K":              K,
        "dtype":          dtype,
        "use_dense_matmul": use_dense_matmul,
    }


def _apply_hadamard_rotation(x_low: torch.Tensor, packed: dict) -> torch.Tensor:
    """Compute `(sf ⊙ x_low) FWHT_normalized` either via cuBLAS dense matmul
    (with precomputed H_dense, fast) or PyTorch's torch.stack FWHT (slow,
    used only when `use_dense_matmul=False`)."""
    if packed.get("use_dense_matmul", True) and packed.get("H_dense") is not None:
        return x_low @ packed["H_dense"]
    # Fallback: explicit PyTorch FWHT (allocations dominate runtime)
    sf = packed["sign_flips"]
    return fast_hadamard_transform((x_low * sf).float()).to(x_low.dtype)


# ---------------------------------------------------------------------------
# Runtime forward — three variants
# ---------------------------------------------------------------------------

def forward_hadamard_w4a4(x: torch.Tensor, packed: dict, bias: torch.Tensor | None = None
                           ) -> torch.Tensor:
    """Hadamard rotation (low region only, FWHT) + W4A4 main + fp16 tail.

    x: [M, K] in compute dtype.
    """
    low_dim = packed["low_dim"]
    hlen    = packed["hlen"]
    gs      = packed["group_size"]
    N       = packed["N"]
    M       = x.shape[0]
    sf      = packed["sign_flips"]

    x_low = x[:, :low_dim].contiguous()
    x_low_rot = _apply_hadamard_rotation(x_low, packed)

    a_packed, a_scales = tt4.quantize_act_int4(x_low_rot.contiguous(), gs)
    y_low = tt4.triton_w4a4_gemm(
        a_packed, a_scales,
        packed["W_low_w4a4"]["packed"], packed["W_low_w4a4"]["scales"],
        M=M, N=N, K=low_dim, group_size=gs, out_dtype=x.dtype,
    )
    if hlen > 0:
        x_high = x[:, low_dim:].contiguous()
        y = y_low + (x_high @ packed["W_tail"].t())
    else:
        y = y_low
    if bias is not None:
        y = y + bias
    return y


def forward_hadamard_w4a16(x: torch.Tensor, packed: dict, bias: torch.Tensor | None = None
                            ) -> torch.Tensor:
    """Hadamard rotation + W4A16 (torch int4pack) main + fp16 tail."""
    low_dim = packed["low_dim"]
    hlen    = packed["hlen"]

    x_low = x[:, :low_dim].contiguous()
    x_low_rot = _apply_hadamard_rotation(x_low, packed)

    y_low = ti4.int4_gemm(x_low_rot.contiguous(), packed["W_low_w4a16"])
    if hlen > 0:
        x_high = x[:, low_dim:].contiguous()
        y = y_low + (x_high @ packed["W_tail"].t())
    else:
        y = y_low
    if bias is not None:
        y = y + bias
    return y


def forward_hadamard_fakequant(x: torch.Tensor, packed: dict, bias: torch.Tensor | None = None
                                ) -> torch.Tensor:
    """Hadamard rotation + dequant-then-fp16 GEMM (no real low-bit kernel).

    Useful as the *correctness* reference for the SNR comparison: it shows what
    the rotation+quant scheme would produce if the kernel were perfectly
    accurate. Any int4-kernel error is below the bf16 noise floor at the
    matmul output, so this is effectively the 'design accuracy' of the scheme.
    """
    low_dim = packed["low_dim"]
    hlen    = packed["hlen"]
    sf      = packed["sign_flips"]
    gs      = packed["group_size"]

    x_low_unrot = (x[:, :low_dim] * sf).float()
    x_low_rot = fast_hadamard_transform(x_low_unrot)

    # Re-dequantize the int4-packed weight to fp32, fp16-multiply.
    from speedup.convert_to_int4_ckpt import unpack_int4_sym
    W_low_recon = unpack_int4_sym(
        packed["W_low_w4a4"]["packed"],
        packed["W_low_w4a4"]["scales"],
        gs, low_dim,
    ).float()

    # int4-quantize-dequantize the activation (same recipe as the kernel uses).
    Xg = x_low_rot.reshape(x_low_rot.shape[0], low_dim // gs, gs)
    a_scale = Xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
    a_codes = (Xg / a_scale).round().clamp(-8, 7)
    x_low_qd = (a_codes * a_scale).reshape(x_low_rot.shape)

    y_low = (x_low_qd @ W_low_recon.t()).to(x.dtype)

    if hlen > 0:
        x_high = x[:, low_dim:].contiguous()
        y = y_low + (x_high @ packed["W_tail"].t())
    else:
        y = y_low
    if bias is not None:
        y = y + bias
    return y
