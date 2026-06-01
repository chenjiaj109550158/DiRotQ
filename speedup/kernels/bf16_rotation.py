"""
speedup/kernels/bf16_rotation.py

Plain bf16 cuBLAS rotation `y = x @ U` — the Blackwell-friendly sibling of
kernels/int8_rotation.py.

Why this exists
---------------
The int8-Triton rotation kernel was a win on Ada (sm_89): K=3072 ran ~2x
faster than cuBLAS bf16. On Blackwell (sm_120) that same Triton kernel runs
at a small fraction of the hardware's int8 throughput and ends up *slower*
than bf16 cuBLAS for the rotation shapes — it became the bottleneck of the
NVFP4 forward (the FP4 GEMM itself is fine). Microbench, K=3072 @ M=4608:

    bf16 GEMM (cuBLAS)        : ~0.31 ms
    nvfp4 quant GEMM          : ~0.13 ms
    int8-Triton rotation      : ~0.64 ms   <- dominates
    bf16 rotation (this file) : ~0.31 ms   <- 2x faster than int8 here

So on Blackwell the rotation is just a bf16 matmul against cuBLAS, which is
already well-tuned for sm_120. No quantization of U or x, no Triton.

The INT4 path keeps using kernels/int8_rotation.py unchanged.

API mirrors int8_rotation just enough to be a drop-in for the report script:
    prepare_U_bf16(U)               -> (U_contig, None)   (analog of quantize_U_int8)
    bf16_rotation_forward(x, U, _)  -> y = x @ U  (bf16)
"""

from __future__ import annotations

import torch


def prepare_U_bf16(U: torch.Tensor) -> tuple[torch.Tensor, None]:
    """Prepare a rotation matrix for the bf16 path.

    Unlike :func:`quantize_U_int8`, there is nothing to quantize — we just
    return a contiguous copy plus a ``None`` placeholder so call sites that
    unpack ``(U, scale)`` (kept identical to the int8 path) still work.
    """
    assert U.dim() == 2
    return U.contiguous(), None


def bf16_rotation_forward(
    x: torch.Tensor,
    U: torch.Tensor,
    scale_U: torch.Tensor | None = None,
    *,
    scale_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``y = x @ U`` in bf16 via cuBLAS.

    Args:
        x:       [M, K] bf16
        U:       [K, N] bf16 (orthogonal rotation matrix)
        scale_U: ignored — accepted only so this is a drop-in replacement for
                 :func:`int8_rotation_forward(x, U_int8, scale_U)`.
        scale_x: ignored, same reason.

    Returns y [M, N] in x's dtype (bf16).
    """
    assert x.dim() == 2 and U.dim() == 2
    assert x.shape[1] == U.shape[0], (x.shape, U.shape)
    return torch.matmul(x, U.to(x.dtype))
