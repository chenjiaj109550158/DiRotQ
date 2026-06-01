"""
speedup/kernels/fp8_rotation.py

FP8 (e4m3) cuBLAS rotation `y = x @ U` — the fastest of the three rotation
kernels on Blackwell (sm_120).

Why this exists
---------------
The rotation `x @ U` is a dense K=3072 GEMM injected before every quantized
K=3072 op. After moving off the int8-Triton kernel (slow on sm_120) to bf16
cuBLAS (kernels/bf16_rotation.py), the rotation was still ~25% of the NVFP4
forward. The bf16 GEMM is already near cuBLAS's bf16 ceiling, so the only
lever left is precision: Blackwell FP8 tensor cores run ~1.7x the bf16
throughput, and the rotation tolerates FP8 because its output is FP4-quantized
by the very next kernel anyway.

Naive FP8 loses to bf16 because the per-call activation quantization
(amax + divide + cast over [M, K]) costs as much as the GEMM saves. The trick
here is a single fused Triton pass that reads each row once, computes its
amax, and writes the e4m3 codes + per-row scale — ~0.02 ms — after which
`torch._scaled_mm` runs the FP8 GEMM with rowwise scaling.

Microbench, K=3072 @ M=4608 (RTX PRO 6000 Blackwell):

    bf16 cuBLAS rotation        : ~0.305 ms
    pure FP8 GEMM (no quant)    : ~0.182 ms   (1.68x)
    this kernel (quant + GEMM)  : ~0.175 ms   (1.74x)   rel err ~0.038

Precision note
--------------
The rotation is computed in FP8 here (per-row activation scale, per-column
weight scale), rel err ~0.04 vs the bf16 rotation. That output is then
FP4-quantized by the nunchaku kernel, which is far coarser, so the FP8
rotation is not the accuracy-limiting step. If you need the rotation at full
bf16 precision (e.g. to match a bf16-rotation accuracy baseline exactly), use
kernels/bf16_rotation.py instead — both share the same call signature.

API (drop-in for bf16_rotation / int8_rotation):
    prepare_U_fp8(U)                 -> (U_fp8_colmajor, scale_U[1,N])
    fp8_rotation_forward(x, U, sU)   -> y = x @ U  (bf16)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_E4M3 = torch.float8_e4m3fn
_FP8_MAX = 448.0


@triton.jit
def _quant_rows_fp8_kernel(
    X_ptr,          # [M, K] bf16
    XQ_ptr,         # [M, K] e4m3 (out)
    SX_ptr,         # [M, 1] fp32 (out, per-row scale)
    M, K,
    sx_m, sx_k,
    BK: tl.constexpr,
):
    """One program per row: read row once → amax → write e4m3 codes + scale.

    Single fused pass keeps the activation-quant cost ~0.02 ms so it doesn't
    eat the FP8 GEMM win. A K=3072 row is 6 KB, comfortably resident while we
    do the two sweeps (amax, then quantize)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BK)

    amax = 0.0
    for k in range(0, tl.cdiv(K, BK)):
        c = k * BK + offs
        m = c < K
        x = tl.load(X_ptr + row * sx_m + c * sx_k, mask=m, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))

    sx = tl.maximum(amax / 448.0, 1e-9)  # 448 = e4m3 max
    tl.store(SX_ptr + row, sx)

    for k in range(0, tl.cdiv(K, BK)):
        c = k * BK + offs
        m = c < K
        x = tl.load(X_ptr + row * sx_m + c * sx_k, mask=m, other=0.0).to(tl.float32)
        q = x / sx
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(XQ_ptr + row * K + c, q.to(XQ_ptr.dtype.element_ty), mask=m)


def _quantize_rows_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] bf16 → (x_e4m3 [M, K] contiguous, scale [M, 1] fp32)."""
    M, K = x.shape
    xq = torch.empty(M, K, device=x.device, dtype=_E4M3)
    sx = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    BK = 1024 if K % 1024 == 0 else 512
    _quant_rows_fp8_kernel[(M,)](
        x, xq, sx, M, K, x.stride(0), x.stride(1), BK=BK, num_warps=4)
    return xq, sx


def prepare_U_fp8(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize the rotation matrix once.

    Returns (B, scale_U) where B is U in e4m3, laid out column-major (as a
    ``.t()`` view of a contiguous [N, K] buffer) so it is a valid second
    operand for ``torch._scaled_mm``, and scale_U is the per-column scale of
    shape [1, N] (rowwise scaling).
    """
    assert U.dim() == 2
    K, N = U.shape
    Uf = U.float()
    sUc = (Uf.abs().amax(dim=0, keepdim=True) / _FP8_MAX).clamp(min=1e-9)  # [1, N]
    Ucm = (Uf.t().contiguous() / sUc.t()).clamp(-_FP8_MAX, _FP8_MAX).to(_E4M3)  # [N, K]
    B = Ucm.t()  # [K, N] column-major view
    return B, sUc.contiguous()


def fp8_rotation_forward(
    x: torch.Tensor,
    U: torch.Tensor,
    scale_U: torch.Tensor,
    *,
    scale_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``y = x @ U`` in FP8 (e4m3) via ``torch._scaled_mm``.

    Args:
        x:       [M, K] bf16
        U:       e4m3 column-major [K, N] from :func:`prepare_U_fp8`
        scale_U: [1, N] fp32 per-column scale from :func:`prepare_U_fp8`
        scale_x: ignored (per-row scale is computed in the fused quant pass);
                 kept for signature compatibility with the other rotations.

    Returns y [M, N] bf16.
    """
    assert x.dim() == 2 and U.dim() == 2
    assert x.shape[1] == U.shape[0], (x.shape, U.shape)
    if not x.is_contiguous():
        x = x.contiguous()
    xq, sx = _quantize_rows_fp8(x)
    return torch._scaled_mm(xq, U, scale_a=sx, scale_b=scale_U,
                            out_dtype=torch.bfloat16)
