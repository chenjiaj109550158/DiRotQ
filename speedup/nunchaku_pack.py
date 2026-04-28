"""
speedup/nunchaku_pack.py

Self-contained port of deepcompressor's nunchaku weight-packer.

This is the function chain that converts a quantized weight + scale into the
exact byte/fragment layout that nunchaku's `gemm_w4a4` CUDA kernel expects
(matching the s4 MMA fragment layout for `mma.m16n8k64.s4.s4.s32`).

Source: https://github.com/nunchaku-tech/deepcompressor
        deepcompressor/backend/nunchaku/utils.py
        deepcompressor/backend/utils.py

I copied only the bits we need. Not modified beyond minor cleanup.

Public API:
    convert_to_nunchaku_w4x4y16(weight, scale, bias=None, smooth=None,
                                  lora=None) -> (qweight, scale, bias, smooth, lora)

    NunchakuWeightPacker(bits=4, warp_n=128)  — exposes pack_weight, pack_scale, etc.
"""

from __future__ import annotations

import typing as tp

import torch


# ===========================================================================
# Utilities (from deepcompressor/backend/utils.py)
# ===========================================================================

def ceil_divide(x: int, divisor: int) -> int:
    return (x + divisor - 1) // divisor


def pad(tensor: tp.Optional[torch.Tensor],
        divisor: int | tp.Sequence[int],
        dim: int | tp.Sequence[int],
        fill_value: float | int = 0) -> torch.Tensor:
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result


def fp_quantize(x: torch.Tensor, codebook: torch.Tensor | None = None) -> torch.Tensor:
    if codebook is None:
        codebook = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=x.dtype, device=x.device,
        )
    return (x.unsqueeze(-1) - codebook.unsqueeze(0)).abs().argmin(dim=-1)


# ===========================================================================
# MMA fragment-aware weight packer (deepcompressor/backend/utils.py)
# ===========================================================================

class MmaWeightPackerBase:
    def __init__(self, bits: int, warp_n: int,
                 comp_n: int | None = None, comp_k: int | None = None):
        self.bits = bits
        assert self.bits in (1, 4, 8, 16, 32)

        self.comp_n = comp_n if comp_n is not None else 16
        self.comp_k = comp_k if comp_k is not None else 256 // self.bits
        self.insn_n = 8
        self.insn_k = self.comp_k
        assert self.insn_k * self.bits in (128, 256)
        assert self.comp_n % self.insn_n == 0

        self.num_lanes = 32
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        assert warp_n >= self.comp_n and warp_n % self.comp_n == 0
        self.warp_n = warp_n

        self.reg_k = 32 // self.bits
        self.reg_n = 1
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        self.pack_size = self.k_pack_size * self.n_pack_size
        assert 1 <= self.pack_size <= 4
        assert self.k_pack_size * self.num_k_lanes * self.reg_k == self.comp_k
        assert self.n_pack_size * self.num_n_lanes * self.reg_n == self.comp_n
        self.mem_k = self.comp_k
        self.mem_n = warp_n
        self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)


class NunchakuWeightPacker(MmaWeightPackerBase):
    """Packs int4/int8 weight + scale tensors into nunchaku's gemm_w4a4 layout."""

    def __init__(self, bits: int, warp_n: int = 128):
        super().__init__(bits=bits, warp_n=warp_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.dtype == torch.int32
        n, k = weight.shape
        assert n % self.mem_n == 0
        assert k % (self.mem_k * self.num_k_unrolls) == 0

        n_tiles, k_tiles = n // self.mem_n, k // self.mem_k
        weight = weight.reshape(
            n_tiles, self.num_n_packs, self.n_pack_size, self.num_n_lanes, self.reg_n,
            k_tiles, self.num_k_packs, self.k_pack_size, self.num_k_lanes, self.reg_k,
        )
        weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        assert weight.shape[4:-2] == (8, 4, 2, 2)
        if self.bits == 4:
            weight = weight.bitwise_and_(0xF)
            shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        elif self.bits == 8:
            weight = weight.bitwise_and_(0xFF)
            shift = torch.arange(0, 32, 8, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        else:
            raise NotImplementedError(f"weight bits {self.bits} is not supported.")
        return weight.view(dtype=torch.int8).view(n, -1)

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if self.check_if_micro_scale(group_size=group_size):
            return self.pack_micro_scale(scale, group_size=group_size)
        assert scale.dtype in (torch.float16, torch.bfloat16)
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n
        scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4,
                               s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1) if group_size == -1 else scale.view(-1, n)

    def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        assert scale.dtype in (torch.float16, torch.bfloat16)
        assert scale.max() <= 448 and scale.min() >= -448
        assert group_size == 16 and self.insn_k == 64 and self.warp_n >= 32
        scale = scale.to(dtype=torch.float8_e4m3fn)
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 4 * 8
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n
        scale = scale.view(n // warp_s, num_s_packs, s_pack_size, 4, 8, -1,
                            self.insn_k // group_size)
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, n)

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        assert weight.dtype in (torch.float16, torch.bfloat16)
        reg_n, reg_k = 1, 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        weight = pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
        if down:
            r, c = weight.shape
            r_packs, c_packs = r // pack_n, c // pack_k
            weight = weight.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
        else:
            c, r = weight.shape
            c_packs, r_packs = c // pack_n, r // pack_k
            weight = weight.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
        weight = weight.reshape(
            c_packs, r_packs, self.n_pack_size, self.num_n_lanes, reg_n,
            self.k_pack_size, self.num_k_lanes, reg_k,
        )
        weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
        return weight.view(c, r) if down else weight.view(c, r)

    def check_if_micro_scale(self, group_size: int) -> bool:
        return self.insn_k == group_size * 4

    def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.ndim == 2
        return pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1))

    def pad_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if group_size > 0 and scale.numel() > scale.shape[0]:
            scale = scale.view(scale.shape[0], 1, -1, 1)
            if self.check_if_micro_scale(group_size=group_size):
                scale = pad(scale, divisor=(self.warp_n, self.insn_k // group_size),
                            dim=(0, 2), fill_value=1)
            else:
                scale = pad(scale, divisor=(self.warp_n, self.num_k_unrolls),
                            dim=(0, 2), fill_value=1)
        else:
            scale = pad(scale, divisor=self.warp_n, dim=0, fill_value=1)
        return scale

    def pad_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        assert weight.ndim == 2
        return pad(weight, divisor=self.warp_n, dim=1 if down else 0)


# ===========================================================================
# High-level conversion: fp weight + scale → nunchaku-format packed tensors
# ===========================================================================

def convert_to_nunchaku_w4x4y16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
    float_point: bool = False,
):
    """
    Quantize + pack a (out_features, in_features) weight into nunchaku's W4A4 layout.

    Args:
        weight: [oc, ic] fp16/bf16 (will be int4-quantized)
        scale:  [oc, 1, ng, 1] fp16/bf16 group scales (where ng = ic // group_size).
                Or [oc] for per-channel scale (will be reshaped).
        bias:   [oc] fp16/bf16, optional.
        smooth: [ic] fp16/bf16, optional. If None, treated as ones (no smoothing).
        lora:   (lora_down [R, ic], lora_up [oc, R]) fp16/bf16, optional.
                Note pre-pack convention follows PyTorch nn.Linear LoRA:
                lora_A.weight = lora_down [R, ic],  lora_B.weight = lora_up [oc, R].
                After packing both have shape[0] == ic / oc respectively
                so the kernel asserts `lora_down.shape[0]==K` and
                `lora_up.shape[0]==N` succeed.
        float_point: True for nvfp4, False for int4. Default False.

    Returns:
        (qweight, scale_packed, bias_packed, smooth_packed, lora_packed)
        - qweight: [oc, ic // 2] int8 (packed int4)
        - scale_packed: [ic // group_size, oc] bf16/fp16
        - bias_packed: [oc] bf16/fp16
        - smooth_packed: [ic] bf16/fp16
        - lora_packed: (lora_down_packed, lora_up_packed) or None
    """
    assert weight.ndim == 2
    device, dtype = weight.device, weight.dtype
    assert dtype in (torch.float16, torch.bfloat16)
    assert scale is not None

    oc, ic = weight.shape
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc).reshape(oc, 1, 1, 1)
        per_tensor_scale = True
    else:
        per_tensor_scale = False
        if scale.ndim == 1:
            scale = scale.reshape(oc, 1, 1, 1)
        elif scale.ndim == 2:
            # [oc, ng] → [oc, 1, ng, 1]
            scale = scale.reshape(oc, 1, scale.shape[1], 1)
        elif scale.ndim == 3:
            # [oc, ng, 1] (common output of amax(keepdim=True)) → [oc, 1, ng, 1]
            scale = scale.reshape(oc, 1, scale.shape[1], scale.shape[2])
    assert scale.ndim == 4
    assert scale.shape[1] == scale.shape[3] == 1
    assert scale.shape[0] == oc
    ng, gs = scale.shape[2], ic // scale.shape[2]
    assert ic == gs * ng

    # Quantize weight to int codes ∈ [-8, 7]
    weight_q = (weight.to(dtype=torch.float32).view(oc, 1, ng, gs)
                / scale.to(dtype=torch.float32, device=device))
    weight_q = weight_q.view(oc, ic)
    if float_point:
        weight_q = fp_quantize(weight_q)
        assert weight_q.min() >= 0 and weight_q.max() <= 15
    else:
        weight_q = weight_q.round_()
        assert weight_q.min() >= -8 and weight_q.max() <= 7, \
            f"int4 codes outside [-8,7]: min={weight_q.min().item()}, max={weight_q.max().item()}"

    bias = torch.zeros([oc, 1], dtype=dtype, device=device) if bias is None else bias.view(-1, 1)
    smooth = torch.ones([ic, 1], dtype=dtype, device=device) if smooth is None else smooth.view(-1, 1)

    packer = NunchakuWeightPacker(bits=4)
    weight_q = packer.pad_weight(weight_q.to(dtype=torch.int32))
    scale = packer.pad_scale(scale.to(dtype=dtype), group_size=gs)
    bias = packer.pad_scale(bias.to(dtype=dtype), group_size=-1)
    smooth = packer.pad_scale(smooth.to(dtype=dtype), group_size=-1)

    weight_packed = packer.pack_weight(weight_q)
    scale_packed = packer.pack_scale(scale, group_size=gs if gs < ic else -1)
    bias_packed = packer.pack_scale(bias, group_size=-1)
    smooth_packed = packer.pack_scale(smooth, group_size=-1)

    lora_packed = None
    if lora is not None:
        lora_down = packer.pack_lowrank_weight(packer.pad_lowrank_weight(lora[0], down=True), down=True)
        lora_up = packer.pack_lowrank_weight(packer.pad_lowrank_weight(lora[1], down=False), down=False)
        lora_packed = (lora_down, lora_up)

    if per_tensor_scale:
        scale_packed = scale_packed.view(-1)[0].view([1])
    return weight_packed, scale_packed, bias_packed, smooth_packed, lora_packed


# ===========================================================================
# DiRotQ-side convenience: turn a low-region bf16 weight into nunchaku tensors
# ===========================================================================

def convert_dirotq_low_weight_to_nunchaku(
    W_low: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
    rank: int = 32,
    use_max_div_8: bool = False,
):
    """Convert a DiRotQ low-region weight into nunchaku gemm_w4a4 input tensors.

    DiRotQ post-rotation produces a weight slice `W[:, :n_low]` that is
    quantized in groups of 64 channels with scale = max(|.|) / 7 (RTN). This
    helper packs that slice into the byte/scale/smooth/lora layout that
    `svdq_gemm_w4a4_cuda` expects.

    Args:
        W_low:      [N, n_low] fp16/bf16. n_low must be a multiple of group_size.
        bias:       [N] fp16/bf16 or None.
        group_size: group size for per-group weight scales (64 for INT4).
        rank:       LoRA rank. We emit zero LoRA tensors of this rank — the
                    kernel computes lora_act_in / lora_up contributions but
                    they cancel to zero, so the output is pure W4A4 GEMM.
        use_max_div_8: if True, scale = max/8 (GPTQ-style codes [-8,7]).
                       Default False matches DiRotQ's existing recipe.

    Returns:
        dict with keys:
          'qweight'      : [N, n_low/2]   int8   — packed int4 codes
          'wscales'      : [n_low/gs, N]  bf16
          'bias'         : [N]            bf16
          'smooth'       : [n_low]        bf16   — ones (no smoothing)
          'lora_down'    : [n_low, R]     bf16   — zeros, packed
          'lora_up'      : [N, R]         bf16   — zeros, packed
    """
    assert W_low.ndim == 2
    assert W_low.dtype in (torch.float16, torch.bfloat16)
    N, n_low = W_low.shape
    assert n_low % group_size == 0, f"n_low={n_low} not multiple of {group_size}"

    device, dtype = W_low.device, W_low.dtype
    Wg = W_low.float().reshape(N, n_low // group_size, group_size)
    denom = 8.0 if use_max_div_8 else 7.0
    w_scale = (Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / denom).to(dtype)

    smooth = torch.ones(n_low, dtype=dtype, device=device)
    bias_ = bias if bias is not None else torch.zeros(N, dtype=dtype, device=device)

    lora_down_fp = torch.zeros(rank, n_low, dtype=dtype, device=device)
    lora_up_fp = torch.zeros(N, rank, dtype=dtype, device=device)

    qweight, wscales, bias_packed, smooth_packed, (lora_d, lora_u) = \
        convert_to_nunchaku_w4x4y16(
            weight=W_low, scale=w_scale, bias=bias_, smooth=smooth,
            lora=(lora_down_fp, lora_up_fp), float_point=False)

    return {
        'qweight': qweight,
        'wscales': wscales,
        'bias': bias_packed.view(-1),
        'smooth': smooth_packed,
        'lora_down': lora_d,
        'lora_up': lora_u,
        'rank': rank,
        'group_size': group_size,
    }
