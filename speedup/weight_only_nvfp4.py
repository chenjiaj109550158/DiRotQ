"""
speedup/weight_only_nvfp4.py

Weight-only NVFP4 linear: weights stored in 4-bit (E2M1 + FP8 group-16 scale),
activations kept bf16 (A16W4). This is the "weight-only" point of comparison
to the full W4A4 NVFP4 path — it isolates the memory win from the compute win.

There is no Blackwell mixed bf16xFP4 tensor-core GEMM, so the forward
dequantizes the 4-bit weight to bf16 (one fused Triton pass) and runs a bf16
cuBLAS matmul. On a compute-bound workload (flux, M=4608) that is the honest
behaviour of weight-only quant: the matmul FLOPs are unchanged (still bf16),
so there is ~no compute speedup — only a ~2x weight-memory reduction (and
bandwidth savings that matter at small M / decode, not here).

The dequant is a single bandwidth-bound Triton kernel (read 4-bit codes +
scales once, write bf16 once), so it adds only the unavoidable extra weight
traffic and the forward lands at ~fp16 speed. (An earlier version dequantized
with eager-torch ops — LUT gather + repeat_interleave + stack — which launched
~6 kernels and several extra copies per layer and made the forward ~2x slower;
the Triton kernel removes that.)

Self-contained: a simple group-16 layout (not nunchaku's MMA layout, which is
only needed by the W4A4 kernel). E2M1 codebook matches nunchaku_pack.fp_quantize.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# E2M1 codebook (matches speedup/nunchaku_pack.py fp_quantize): index -> value.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_NVFP4_MAX = 6.0     # top E2M1 magnitude
_E4M3_MAX = 448.0    # max FP8 e4m3 (caps the per-group micro-scale range)
_GROUP = 16


def _quantize_nvfp4(W: torch.Tensor):
    """W [N, K] (K multiple of 16) -> (codes [N, K//2] uint8,
                                       micro [N, K//16] e4m3,
                                       gscale [N] bf16).

    Two-level NVFP4 (per real hardware): per-output-channel global scale +
    per-group(16) e4m3 micro-scale. The global scale normalizes the group
    scales into the e4m3 range so small-magnitude groups don't underflow.

        group_scale_g = amax_g / 6
        gscale[n]     = max_g(group_scale_g) / 448
        micro_g       = e4m3(group_scale_g / gscale[n])    in [0, 448]
        W ~= e2m1(code) * micro_g * gscale[n]
    """
    N, K = W.shape
    assert K % _GROUP == 0, K
    dev = W.device
    cb = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    Wg = W.float().reshape(N, K // _GROUP, _GROUP)
    gscale_full = (Wg.abs().amax(-1) / _NVFP4_MAX).clamp(min=1e-12)        # [N, K//16] fp32
    gscale = (gscale_full.amax(-1) / _E4M3_MAX).clamp(min=1e-12)           # [N] fp32
    micro = (gscale_full / gscale[:, None]).to(torch.float8_e4m3fn)        # [N, K//16] e4m3
    eff = micro.float() * gscale[:, None]                                  # effective group scale
    idx = (Wg / eff.unsqueeze(-1)).unsqueeze(-1).sub(cb).abs().argmin(-1)  # [N, K//16, 16]
    idx = idx.reshape(N, K).to(torch.uint8)
    # Planar packing: byte j holds column j (low nibble) and column j+K/2 (high
    # nibble). Dequant then writes the two column-halves contiguously (coalesced),
    # unlike an even/odd interleave which forces strided stores.
    K2 = K // 2
    codes = (idx[:, :K2] | (idx[:, K2:] << 4)).contiguous()               # [N, K//2] uint8
    return codes, micro.contiguous(), gscale.to(torch.bfloat16).contiguous()


@triton.jit
def _dequant_kernel(
    CODES, MICRO, GSCALE, CB, W,
    N, K,
    s_cn, s_ck, s_mn, s_mg, s_wn, s_wk,
    BN: tl.constexpr, BK: tl.constexpr,
):
    """Fused NVFP4 weight dequant: 4-bit codes + scales -> bf16 W, one pass.

    One program writes a contiguous [BN rows x BK cols] tile of W. Planar
    packing (see _quantize_nvfp4): output column c reads byte (c mod K/2),
    low nibble if c < K/2 else high nibble. Output columns are contiguous so
    the store is coalesced. Reads codes once (per nibble), writes W once."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_n * BN + tl.arange(0, BN)          # [BN]
    cols = pid_k * BK + tl.arange(0, BK)          # [BK]
    K2 = K // 2
    rmask = rows < N
    cmask = cols < K
    mask2 = rmask[:, None] & cmask[None, :]

    is_hi = cols >= K2
    bcol = cols - is_hi.to(tl.int32) * K2          # byte column [BK]
    shift = is_hi.to(tl.int32) * 4

    gsc = tl.load(GSCALE + rows, mask=rmask, other=0.0).to(tl.float32)        # [BN]
    grp = cols // 16
    micro = tl.load(MICRO + rows[:, None] * s_mn + grp[None, :] * s_mg,
                    mask=mask2, other=0.0).to(tl.float32)                     # [BN,BK]
    scale = micro * gsc[:, None]

    byte = tl.load(CODES + rows[:, None] * s_cn + bcol[None, :] * s_ck,
                   mask=mask2, other=0).to(tl.int32)                          # [BN,BK]
    idx = (byte >> shift[None, :]) & 0xF
    val = tl.load(CB + idx) * scale                                          # [BN,BK]
    tl.store(W + rows[:, None] * s_wn + cols[None, :] * s_wk,
             val.to(W.dtype.element_ty), mask=mask2)


def _dequant_triton(codes, micro, gscale, cb, dtype):
    N, Kh = codes.shape
    K = Kh * 2
    W = torch.empty(N, K, dtype=dtype, device=codes.device)
    BN, BK = 16, 256
    grid = (triton.cdiv(N, BN), triton.cdiv(K, BK))
    _dequant_kernel[grid](
        codes, micro, gscale, cb, W, N, K,
        codes.stride(0), codes.stride(1), micro.stride(0), micro.stride(1),
        W.stride(0), W.stride(1), BN=BN, BK=BK, num_warps=4)
    return W


class WeightOnlyNVFP4Linear(nn.Module):
    """A16W4 linear: bf16 activations, NVFP4 (4-bit) weights."""

    def __init__(self, in_features, out_features, bias=True,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.register_buffer("codes", torch.zeros(out_features, in_features // 2,
                                                   dtype=torch.uint8, device=device))
        self.register_buffer("micro", torch.zeros(out_features, in_features // _GROUP,
                                                   dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("gscale", torch.ones(out_features, dtype=torch.bfloat16, device=device))
        self.register_buffer("_cb", torch.tensor(_E2M1, dtype=dtype, device=device))
        self.register_buffer("_cb_f32", torch.tensor(_E2M1, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear):
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                dtype=lin.weight.dtype, device=lin.weight.device)
        codes, micro, gscale = _quantize_nvfp4(lin.weight.data)
        m.codes.copy_(codes)
        m.micro.copy_(micro)
        m.gscale.copy_(gscale)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data.to(m.bias.dtype))
        return m

    def dequant(self) -> torch.Tensor:
        """Reconstruct the bf16 weight [out, in] via the fused Triton kernel."""
        return _dequant_triton(self.codes, self.micro, self.gscale, self._cb_f32, self.dtype)

    def dequant_torch(self) -> torch.Tensor:
        """Eager-torch reference dequant (slow; used to validate the kernel)."""
        N, Kh = self.codes.shape
        K = Kh * 2
        lo = (self.codes & 0xF).to(torch.long)                 # cols [0, K/2)
        hi = (self.codes >> 4).to(torch.long)                  # cols [K/2, K)
        idx = torch.cat([lo, hi], dim=1)                       # planar -> [N, K]
        vals = self._cb[idx]                                   # [N, K] e2m1 values
        scale = (self.micro.to(self.dtype) * self.gscale[:, None]).repeat_interleave(_GROUP, dim=1)
        return vals * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequant(), self.bias)

    def weight_bytes(self) -> int:
        # uint8 codes + e4m3 micro-scales (1 byte each) + bf16 per-channel global
        return self.codes.numel() + self.micro.numel() + self.gscale.numel() * 2
