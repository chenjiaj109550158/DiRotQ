"""Packed INT4 reference/runtime for the FLUX shared-PCA audit.

This module deliberately separates *storage* from arithmetic:

* weights persist as two signed INT4 values per byte plus one BF16 scale per
  output-row/K group;
* activations are dynamically quantized to unsigned INT4 with the exact
  asymmetric per-token/group rule used by :class:`ActQuantizer`;
* the low dot product is accumulated from integer codes and group scales -- a
  dense BF16 reconstruction of the low operand is never materialized;
* the protected PCA tail remains a native BF16 matrix multiplication.

CUDA execution uses a Triton kernel that reads the packed nibbles directly,
forms K64 INT32 partial dots and applies group scales during FP32 accumulation.
This is an accuracy/memory kernel, not a latency-tuned Nunchaku replacement.
It is nevertheless materially different from fake quantization: the
persistent low weight is genuinely 4-bit and no dense low operand is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_integer_codes(codes: torch.Tensor, low: int, high: int) -> None:
    if codes.is_floating_point():
        if not torch.equal(codes, torch.round(codes)):
            raise ValueError("INT4 codes must be integral")
    if codes.numel() and (int(codes.min()) < low or int(codes.max()) > high):
        raise ValueError(f"INT4 codes outside [{low}, {high}]")


def pack_signed_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed two's-complement INT4; earlier K is the low nibble."""
    _require_integer_codes(codes, -8, 7)
    values = codes.to(torch.int16)
    if values.shape[-1] % 2:
        values = F.pad(values, (0, 1))
    nibbles = values & 0xF
    return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)


def unpack_signed_int4(payload: torch.Tensor, logical_k: int | None = None) -> torch.Tensor:
    """Decode signed two's-complement INT4 to int8."""
    if payload.dtype != torch.uint8:
        raise TypeError("packed INT4 payload must be uint8")
    low = (payload & 0xF).to(torch.int8)
    high = ((payload >> 4) & 0xF).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.stack((low, high), dim=-1).flatten(-2)
    return out[..., :logical_k] if logical_k is not None else out


def pack_unsigned_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack unsigned INT4; earlier K is the low nibble."""
    _require_integer_codes(codes, 0, 15)
    values = codes.to(torch.uint8)
    if values.shape[-1] % 2:
        values = F.pad(values, (0, 1))
    return values[..., 0::2] | (values[..., 1::2] << 4)


def unpack_unsigned_int4(payload: torch.Tensor, logical_k: int | None = None) -> torch.Tensor:
    if payload.dtype != torch.uint8:
        raise TypeError("packed INT4 payload must be uint8")
    low = payload & 0xF
    high = (payload >> 4) & 0xF
    out = torch.stack((low, high), dim=-1).flatten(-2)
    return out[..., :logical_k] if logical_k is not None else out


@dataclass(frozen=True)
class ActivationInt4:
    payload: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    logical_k: int
    padded_k: int
    original_shape: tuple[int, ...]

    def codes(self) -> torch.Tensor:
        return unpack_unsigned_int4(self.payload, self.padded_k)

    def decode(self, dtype: torch.dtype) -> torch.Tensor:
        m = self.codes().shape[0]
        groups = self.scales.shape[1]
        group_size = self.padded_k // groups
        q = self.codes().reshape(m, groups, group_size).to(self.scales.dtype)
        out = self.scales * (q - self.zeros)
        out = out.reshape(m, self.padded_k)[:, : self.logical_k]
        return out.reshape(*self.original_shape[:-1], self.logical_k).to(dtype)


def quantize_activation_int4(
    x: torch.Tensor,
    group_size: int = 64,
    *,
    clip_ratio: float = 1.0,
) -> ActivationInt4:
    """Exact asymmetric INT4 rule used by the repository's INT W4A4 path."""
    if not x.is_floating_point():
        raise TypeError("activation must be floating point")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    original_shape = tuple(x.shape)
    logical_k = x.shape[-1]
    padded_k = ((logical_k + group_size - 1) // group_size) * group_size
    flat = x.reshape(-1, logical_k)
    if padded_k != logical_k:
        flat = F.pad(flat, (0, padded_k - logical_k))
    grouped = flat.reshape(flat.shape[0], -1, group_size)
    xmax = grouped.amax(dim=-1, keepdim=True) * clip_ratio
    xmin = grouped.amin(dim=-1, keepdim=True) * clip_ratio
    zero_group = (xmin == 0) & (xmax == 0)
    xmin = torch.where(zero_group, torch.full_like(xmin, -1), xmin)
    xmax = torch.where(zero_group, torch.full_like(xmax, 1), xmax)
    scales = ((xmax - xmin).clamp(min=1e-5) / 15).to(x.dtype)
    zeros = torch.round(-xmin / scales).to(x.dtype)
    codes = torch.clamp(torch.round(grouped / scales) + zeros, 0, 15).to(torch.uint8)
    payload = pack_unsigned_int4(codes.reshape(flat.shape[0], padded_k))
    return ActivationInt4(
        payload=payload,
        scales=scales,
        zeros=zeros,
        logical_k=logical_k,
        padded_k=padded_k,
        original_shape=original_shape,
    )


@dataclass(frozen=True)
class WeightPackingReport:
    elements: int
    exact_bf16_elements: int
    max_abs_decode_error: float
    mean_abs_decode_error: float
    payload_bytes: int
    scale_bytes: int

    @property
    def exact_fraction(self) -> float:
        return self.exact_bf16_elements / max(1, self.elements)


def fit_reconstructed_int4_weight(
    weight: torch.Tensor,
    group_size: int = 64,
    *,
    scale_dtype: torch.dtype = torch.bfloat16,
    chunk_groups: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, WeightPackingReport]:
    """Recover legal signed-INT4 payload/scales from a reconstructed cache.

    Historical GPTQ caches retain only the BF16 reconstructed matrix.  The
    original fixed scale is therefore not directly available.  For each group
    we enumerate the only plausible scales implied by the largest positive and
    negative reconstructed values (positive codes 1..7, negative codes 1..8),
    round those scales to the serialized dtype, and select the candidate with
    minimum BF16 reconstruction SSE.  This is fail-closed at the caller: the
    returned report exposes any mismatch instead of silently calling it the
    original GPTQ payload.
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a 2-D floating tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    n, logical_k = weight.shape
    padded_k = ((logical_k + group_size - 1) // group_size) * group_size
    work = weight.detach()
    if padded_k != logical_k:
        work = F.pad(work, (0, padded_k - logical_k))
    groups = work.reshape(-1, group_size)
    out_codes = torch.empty_like(groups, dtype=torch.int8)
    out_scales = torch.empty(groups.shape[0], dtype=scale_dtype, device=weight.device)

    pos_div = torch.arange(1, 8, dtype=torch.float32, device=weight.device)
    neg_div = torch.arange(1, 9, dtype=torch.float32, device=weight.device)
    for start in range(0, groups.shape[0], chunk_groups):
        end = min(start + chunk_groups, groups.shape[0])
        target_native = groups[start:end]
        target = target_native.float()
        positive = target.clamp_min(0).amax(dim=1)
        negative = (-target.clamp_max(0)).amax(dim=1)
        candidates = torch.cat(
            (positive[:, None] / pos_div, negative[:, None] / neg_div), dim=1
        )
        candidates = candidates.to(scale_dtype).float()
        candidates = torch.where(candidates > 0, candidates, torch.ones_like(candidates))
        q = torch.round(target[:, None, :] / candidates[:, :, None]).clamp(-8, 7)
        decoded_native = (q * candidates[:, :, None]).to(weight.dtype)
        error = (decoded_native.float() - target[:, None, :]).square().sum(dim=2)
        chosen = error.argmin(dim=1)
        row = torch.arange(end - start, device=weight.device)
        selected_scale = candidates[row, chosen].to(scale_dtype)
        selected_q = torch.round(target / selected_scale.float()[:, None]).clamp(-8, 7)
        out_scales[start:end] = selected_scale
        out_codes[start:end] = selected_q.to(torch.int8)

    codes = out_codes.reshape(n, padded_k)
    scales = out_scales.reshape(n, padded_k // group_size)
    payload = pack_signed_int4(codes)
    decoded = decode_weight_int4(payload, scales, logical_k, group_size, weight.dtype)
    diff = (decoded.float() - weight.float()).abs()
    report = WeightPackingReport(
        elements=weight.numel(),
        exact_bf16_elements=int((decoded == weight).sum()),
        max_abs_decode_error=float(diff.max()) if diff.numel() else 0.0,
        mean_abs_decode_error=float(diff.mean()) if diff.numel() else 0.0,
        payload_bytes=payload.numel() * payload.element_size(),
        scale_bytes=scales.numel() * scales.element_size(),
    )
    return payload, scales, report


def decode_weight_int4(
    payload: torch.Tensor,
    scales: torch.Tensor,
    logical_k: int,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    n = payload.shape[0]
    padded_k = payload.shape[1] * 2
    if padded_k % group_size or scales.shape != (n, padded_k // group_size):
        raise ValueError("payload/scales/group-size mismatch")
    q = unpack_signed_int4(payload, padded_k).reshape(n, -1, group_size)
    decoded = q.to(scales.dtype) * scales[..., None]
    return decoded.reshape(n, padded_k)[:, :logical_k].to(dtype)


def _integer_mm(a: torch.Tensor, b_t: torch.Tensor, require_cuda: bool) -> torch.Tensor:
    if a.dtype != torch.int8 or b_t.dtype != torch.int8:
        raise TypeError("integer GEMM operands must be int8")
    if a.device != b_t.device:
        raise RuntimeError("integer GEMM operands are on different devices")
    if a.device.type == "cuda":
        if not hasattr(torch, "_int_mm"):
            raise RuntimeError("this PyTorch build has no CUDA integer matmul")
        return torch._int_mm(a.contiguous(), b_t.contiguous())
    if require_cuda:
        raise RuntimeError("real INT4 runtime forbids silent CPU fallback")
    return a.to(torch.int32) @ b_t.to(torch.int32)


class PackedSplitInt4Linear(nn.Module):
    """Persistent packed W4 + optional native-dtype protected high branch."""

    def __init__(
        self,
        qweight: torch.Tensor,
        weight_scales: torch.Tensor,
        *,
        logical_low_k: int,
        group_size: int,
        high_weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        require_cuda: bool = True,
    ) -> None:
        super().__init__()
        if qweight.dtype != torch.uint8 or weight_scales.ndim != 2:
            raise TypeError("invalid packed weight payload/scales")
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("weight_scales", weight_scales.contiguous())
        self.register_buffer(
            "high_weight",
            None if high_weight is None else high_weight.detach().contiguous(),
        )
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())
        self.logical_low_k = int(logical_low_k)
        self.group_size = int(group_size)
        self.in_features = self.logical_low_k + (
            0 if high_weight is None else int(high_weight.shape[1])
        )
        self.out_features = int(qweight.shape[0])
        self.require_cuda = bool(require_cuda)

    @property
    def persistent_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.qweight, self.weight_scales, self.high_weight, self.bias)
            if tensor is not None
        )

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor | None = None) -> torch.Tensor:
        if x_low.device != self.qweight.device:
            raise RuntimeError("activation and packed weight are on different devices")
        if self.require_cuda and x_low.device.type != "cuda":
            raise RuntimeError("real INT4 runtime forbids silent CPU fallback")
        original_shape = x_low.shape[:-1]
        activation = quantize_activation_int4(x_low, self.group_size)
        padded_k = activation.padded_k
        groups = padded_k // self.group_size
        if x_low.device.type == "cuda":
            if self.group_size != 64:
                raise RuntimeError("CUDA packed INT4 kernel requires K64 groups")
            from .packed_int4_triton import packed_w4a4_gemm

            output = packed_w4a4_gemm(
                activation.payload,
                self.qweight,
                activation.scales,
                activation.zeros,
                self.weight_scales,
            )
        else:
            a_codes = activation.codes().reshape(-1, groups, self.group_size)
            w_codes = unpack_signed_int4(self.qweight, padded_k).reshape(
                self.out_features, groups, self.group_size
            )
            a_zero = activation.zeros.reshape(-1, groups, 1)
            a_scale = activation.scales.reshape(-1, groups)
            output = torch.zeros(
                a_codes.shape[0], self.out_features, dtype=torch.float32, device=x_low.device
            )
            for group in range(groups):
                centered = a_codes[:, group].to(torch.int16) - a_zero[:, group].to(torch.int16)
                if centered.numel() and (centered.min() < -128 or centered.max() > 127):
                    raise RuntimeError("activation zero-point produced values outside int8")
                product = _integer_mm(
                    centered.to(torch.int8),
                    w_codes[:, group].to(torch.int8).T,
                    self.require_cuda,
                )
                factor = (
                    a_scale[:, group, None].float()
                    * self.weight_scales[:, group][None].float()
                )
                output.add_(product.float() * factor)

        if self.high_weight is not None:
            if x_high is None or x_high.shape[-1] != self.high_weight.shape[1]:
                raise ValueError("protected high activation/weight mismatch")
            high = F.linear(x_high, self.high_weight, None)
            output.add_(high.reshape(-1, self.out_features).float())
        elif x_high is not None and x_high.shape[-1]:
            raise ValueError("unexpected protected high activation")
        if self.bias is not None:
            output.add_(self.bias.float())
        return output.to(x_low.dtype).reshape(*original_shape, self.out_features)


def packed_linear_state(module: PackedSplitInt4Linear) -> dict[str, Any]:
    return {
        "qweight": module.qweight.detach().cpu(),
        "weight_scales": module.weight_scales.detach().cpu(),
        "high_weight": None if module.high_weight is None else module.high_weight.detach().cpu(),
        "bias": None if module.bias is None else module.bias.detach().cpu(),
        "logical_low_k": module.logical_low_k,
        "group_size": module.group_size,
        "in_features": module.in_features,
        "out_features": module.out_features,
    }
