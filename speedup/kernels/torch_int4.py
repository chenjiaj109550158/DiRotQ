"""
W4A16 backend using torch._weight_int4pack_mm.

This is the simpler of the two real-kernel paths in DiRotQ/speedup:
  - Weight in the low region is repacked to int4 with per-group asymmetric scales.
  - Activation in the low region stays at fp16/bf16 (no activation quantization).
  - The matmul uses PyTorch's built-in `_weight_int4pack_mm` (group-quantized
    W4A16 GEMM, originally exposed for AWQ-style mobile inference).

This gives a memory-bandwidth speedup (4x lower weight traffic) but does NOT
reduce activation traffic or use int4 tensor cores. The dirotq-triton backend
in triton_w4a4.py covers that case.

API:
    pack_weight_int4(W_low_fp, group_size) -> packed_dict
        W_low_fp: [out, in_low] in fp16/bf16, values already on the int4 grid
                  (i.e. read straight out of DiRotQ's quantized cache).
        Returns dict with keys {"weight_int4pack", "scales_and_zeros",
                                 "group_size", "in_features", "out_features"}.

    int4_gemm(act_low_fp, packed_dict) -> [M, out] fp tensor

Notes on torch's API (PyTorch 2.1+):
  torch._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    x:                    [M, K] bf16/fp16
    weight_int4pack:      packed via torch._convert_weight_to_int4pack
    groupsize:            int (32, 64, 128, ...)
    scales_and_zeros:     [K // groupsize, N, 2] bf16
    -> returns:           [M, N] bf16/fp16
"""

from __future__ import annotations

import torch


def _supports_int4pack_mm() -> bool:
    return hasattr(torch.ops.aten, "_weight_int4pack_mm") and \
        hasattr(torch.ops.aten, "_convert_weight_to_int4pack")


def pack_weight_int4(W_low_fp: torch.Tensor, group_size: int,
                     inner_k_tiles: int = 8) -> dict:
    """Repack fake-quantized fp16/bf16 weights into torch's int4pack format.

    W_low_fp lives on the int4 grid (DiRotQ's RTN/GPTQ cache), so reverse-
    engineering codes + scales is a clean per-group fit.

    Convention used by `_weight_int4pack_mm` (verified empirically on torch
    2.6 cu124, see speedup/README.md notes):

        codes : uint4 in [0, 15], packed two per byte with the FIRST code in
                the HIGH nibble and the SECOND in the LOW nibble.
                byte = (code[2k] << 4) | (code[2k+1] & 0xF)
        dequant: W_dequant = (code - 8) * scale + zero_offset
                  where zero_offset = wmin + 8 * scale.
        scales_and_zeros[g, n, 0] = scale
        scales_and_zeros[g, n, 1] = zero_offset   (both in bf16/fp16)

    Constraints (from the CUDA kernel):
        in_features must be divisible by inner_k_tiles * 16  (default 128)
        out_features must be divisible by 8

    Returns a dict with the packed int4 weight, scale/zero tensor, and
    metadata for the forward call.
    """
    if not _supports_int4pack_mm():
        raise RuntimeError(
            "torch._weight_int4pack_mm is not available. Update PyTorch >= 2.1.")

    assert W_low_fp.dim() == 2
    out_f, in_f = W_low_fp.shape
    assert in_f % group_size == 0, \
        f"in_features={in_f} not divisible by group_size={group_size}"
    div = inner_k_tiles * 16
    if in_f % div != 0:
        # Fall back to a smaller inner_k_tiles if possible.
        for ikt in (4, 2):
            if in_f % (ikt * 16) == 0:
                inner_k_tiles = ikt
                div = ikt * 16
                break
        else:
            raise ValueError(
                f"in_features={in_f} not divisible by inner_k_tiles*16 "
                f"(tried 8/4/2). Pick a different group_size or layer.")
    if out_f % 8 != 0:
        raise ValueError(f"out_features={out_f} must be divisible by 8")

    device = W_low_fp.device
    # _weight_int4pack_mm on CUDA only supports bf16 (as of torch 2.6). The
    # caller may have fp16 weights — we route through bf16 internally and
    # remember the original dtype so int4_gemm can cast back.
    orig_dtype = W_low_fp.dtype
    target_dtype = torch.bfloat16

    # Per-group asymmetric quantization with min-max recipe
    Wg = W_low_fp.float().reshape(out_f, in_f // group_size, group_size)
    wmax = Wg.amax(dim=-1)            # [out_f, n_groups]
    wmin = Wg.amin(dim=-1)
    scale = ((wmax - wmin) / 15).clamp(min=1e-6)
    # codes = round((W - wmin) / scale)  in [0, 15]
    codes = ((Wg - wmin.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(0, 15).to(torch.int32)
    codes_flat = codes.reshape(out_f, in_f).contiguous()

    # Pack two codes per byte, FIRST code in high nibble.
    # codes_flat[:, 0::2] -> high nibble, codes_flat[:, 1::2] -> low nibble.
    high = codes_flat[..., 0::2] & 0xF
    low = codes_flat[..., 1::2] & 0xF
    packed_u8 = ((high << 4) | low).to(torch.uint8).contiguous()
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
        packed_u8.to(device), inner_k_tiles
    )

    # zero_offset = wmin + 8 * scale  (matches kernel's (code-8)*scale + zero)
    zero_offset = wmin + 8 * scale
    scales_and_zeros = torch.stack(
        [scale.to(target_dtype), zero_offset.to(target_dtype)], dim=-1
    ).transpose(0, 1).contiguous()  # [n_groups, out_f, 2]

    return {
        "weight_int4pack": weight_int4pack,
        "scales_and_zeros": scales_and_zeros.to(device),
        "group_size": group_size,
        "in_features": in_f,
        "out_features": out_f,
        "inner_k_tiles": inner_k_tiles,
        "compute_dtype": target_dtype,
        "orig_dtype": orig_dtype,
    }


def int4_gemm(act_low_fp: torch.Tensor, packed: dict) -> torch.Tensor:
    """W4A16 GEMM. act_low_fp: [..., in_low] -> output: [..., out].

    Reshapes leading dims into M, runs _weight_int4pack_mm, reshapes back.
    Routes through bf16 internally (the only dtype the kernel supports on
    CUDA), and casts the output back to the original dtype.
    """
    in_f = packed["in_features"]
    out_f = packed["out_features"]
    gs = packed["group_size"]
    target_dtype = packed["compute_dtype"]
    out_dtype = packed.get("orig_dtype", target_dtype)

    leading = act_low_fp.shape[:-1]
    M = 1
    for d in leading:
        M *= d
    x2d = act_low_fp.reshape(M, in_f).to(target_dtype).contiguous()

    y = torch.ops.aten._weight_int4pack_mm(
        x2d, packed["weight_int4pack"], gs, packed["scales_and_zeros"]
    )
    return y.reshape(*leading, out_f).to(out_dtype)
