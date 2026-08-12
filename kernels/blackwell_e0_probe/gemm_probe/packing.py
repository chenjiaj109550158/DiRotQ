"""Canonical padded byte layouts for the static-format GEMM probe."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.blackwell_e0_probe.packing import (
    decode_e4m3_bytes,
    encode_e4m3_bytes,
    pack_nibbles,
    unpack_nibbles,
)


def ceil_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        raise ValueError("GEMM dimensions must be positive")
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("GEMM dimensions must be positive")

    @property
    def mp(self) -> int:
        return ceil_multiple(self.m, 16)

    @property
    def np(self) -> int:
        return ceil_multiple(self.n, 8)

    @property
    def kp(self) -> int:
        return ceil_multiple(self.k, 64)

    @property
    def k_blocks(self) -> int:
        return (self.k + 15) // 16


@dataclass
class CanonicalInputs:
    shape: GemmShape
    packed_a: torch.Tensor
    packed_b: torch.Tensor
    a_scales: torch.Tensor
    b_scales: torch.Tensor
    alpha_a: float
    alpha_b: float


def pack_canonical_a(codes: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if codes.dtype != torch.uint8 or tuple(codes.shape) != (shape.m, shape.k):
        raise ValueError("A codes must be uint8[M,K]")
    if bool((codes > 15).any()):
        raise ValueError("A code exceeds one nibble")
    padded = torch.zeros((shape.mp, shape.kp), dtype=torch.uint8)
    padded[:shape.m, :shape.k] = codes.cpu()
    return pack_nibbles(padded.reshape(-1)).reshape(shape.mp, shape.kp // 2)


def pack_canonical_b(codes: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if codes.dtype != torch.uint8 or tuple(codes.shape) != (shape.k, shape.n):
        raise ValueError("B codes must be uint8[K,N]")
    if bool((codes > 15).any()):
        raise ValueError("B code exceeds one nibble")
    # Canonical B is [N,Kp/2]: one contiguous column stream per row here.
    padded = torch.zeros((shape.np, shape.kp), dtype=torch.uint8)
    padded[:shape.n, :shape.k] = codes.cpu().T
    return pack_nibbles(padded.reshape(-1)).reshape(shape.np, shape.kp // 2)


def pack_canonical_a_scales(scales: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if not scales.is_floating_point() or tuple(scales.shape) != (shape.m, shape.k_blocks):
        raise ValueError("A scales must be floating [M,ceil(K/16)]")
    padded = torch.ones((shape.mp, shape.kp // 16), dtype=torch.float32)
    padded[:shape.m, :shape.k_blocks] = scales.float().cpu()
    return encode_e4m3_bytes(padded.contiguous()).reshape(padded.shape)


def pack_canonical_b_scales(scales: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if not scales.is_floating_point() or tuple(scales.shape) != (shape.n, shape.k_blocks):
        raise ValueError("B scales must be floating [N,ceil(K/16)]")
    padded = torch.ones((shape.np, shape.kp // 16), dtype=torch.float32)
    padded[:shape.n, :shape.k_blocks] = scales.float().cpu()
    return encode_e4m3_bytes(padded.contiguous()).reshape(padded.shape)


def unpack_canonical_a(packed: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (shape.mp, shape.kp // 2):
        raise ValueError("A payload does not match canonical padded shape")
    return unpack_nibbles(packed.reshape(-1), shape.mp * shape.kp).reshape(shape.mp, shape.kp)


def unpack_canonical_b(packed: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (shape.np, shape.kp // 2):
        raise ValueError("B payload does not match canonical padded shape")
    columns = unpack_nibbles(packed.reshape(-1), shape.np * shape.kp).reshape(shape.np, shape.kp)
    return columns.T.contiguous()


def unpack_canonical_a_scales(encoded: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if encoded.dtype != torch.uint8 or tuple(encoded.shape) != (shape.mp, shape.kp // 16):
        raise ValueError("A scales do not match canonical padded shape")
    return decode_e4m3_bytes(encoded.contiguous()).reshape(encoded.shape)


def unpack_canonical_b_scales(encoded: torch.Tensor, shape: GemmShape) -> torch.Tensor:
    if encoded.dtype != torch.uint8 or tuple(encoded.shape) != (shape.np, shape.kp // 16):
        raise ValueError("B scales do not match canonical padded shape")
    return decode_e4m3_bytes(encoded.contiguous()).reshape(encoded.shape)


def validate_canonical(inputs: CanonicalInputs) -> None:
    shape = inputs.shape
    a = unpack_canonical_a(inputs.packed_a, shape)
    b = unpack_canonical_b(inputs.packed_b, shape)
    a_scales = unpack_canonical_a_scales(inputs.a_scales, shape)
    b_scales = unpack_canonical_b_scales(inputs.b_scales, shape)
    if bool((a[shape.m:, :] != 0).any()) or bool((a[:shape.m, shape.k:] != 0).any()):
        raise ValueError("A padding payload must use positive-zero nibbles")
    if bool((b[:, shape.n:] != 0).any()) or bool((b[shape.k:, :shape.n] != 0).any()):
        raise ValueError("B padding payload must use positive-zero nibbles")
    if bool((a_scales[shape.m:, :] != 1).any()) or bool((a_scales[:shape.m, shape.k_blocks:] != 1).any()):
        raise ValueError("A padding scales must equal one")
    if bool((b_scales[shape.n:, :] != 1).any()) or bool((b_scales[:shape.n, shape.k_blocks:] != 1).any()):
        raise ValueError("B padding scales must equal one")


__all__ = [
    "CanonicalInputs", "GemmShape", "ceil_multiple", "pack_canonical_a",
    "pack_canonical_a_scales", "pack_canonical_b", "pack_canonical_b_scales",
    "unpack_canonical_a", "unpack_canonical_a_scales", "unpack_canonical_b",
    "unpack_canonical_b_scales", "validate_canonical",
]
