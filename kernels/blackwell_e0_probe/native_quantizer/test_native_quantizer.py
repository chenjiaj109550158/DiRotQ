from __future__ import annotations

import struct

import pytest
import torch

from kernels.blackwell_e0_probe.native_quantizer.reference import (
    UE4M3_ONE, canonical_scales_from_native, e4m3_encode_scalar,
    native_scale_offset, native_scale_size, native_scales_from_canonical,
    scalar_quantize, vectorized_quantize,
)


@pytest.mark.parametrize("fmt", ["e2m1", "e0m3"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scalar_vectorized_byte_contract(fmt: str, dtype: torch.dtype) -> None:
    source = torch.tensor([[2688.0, -12.0, -0.0, 0.0] + [float(i - 5) / 2 for i in range(61)]], dtype=dtype)
    scalar, vector = scalar_quantize(source, fmt), vectorized_quantize(source, fmt)
    assert struct.pack("<f", scalar.alpha) == struct.pack("<f", vector.alpha)
    assert torch.equal(scalar.payload, vector.payload)
    assert torch.equal(scalar.native_scales, vector.native_scales)


def test_e4m3_rne_matches_pytorch_all_intervals() -> None:
    values = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    probes = [0.0, 2.0**-10, 2.0**-9, 1.0, 448.0]
    probes += [float((values[index] + values[index + 1]) / 2) for index in range(0x7E)]
    for value in probes:
        expected = int(torch.tensor([value], dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)[0])
        assert e4m3_encode_scalar(value) == expected


def test_midpoint_ties_choose_lower_fp4_magnitude() -> None:
    # alpha=1 because a distinct block contains the full-range 2688 value.
    e0 = torch.zeros((1, 64), dtype=torch.bfloat16)
    e0[0, 0] = 2688
    e0[0, 16:24] = torch.tensor([.5,1.5,2.5,3.5,4.5,5.5,6.5,7], dtype=torch.bfloat16)
    result = vectorized_quantize(e0, "e0m3")
    codes = torch.empty((result.mp,result.kp),dtype=torch.uint8)
    codes[:,0::2],codes[:,1::2]=result.payload&15,result.payload>>4
    assert codes[0,16:24].tolist() == list(range(8))
    e2 = torch.zeros((1,64),dtype=torch.bfloat16)
    e2[0,0]=2688
    e2[0,16:24]=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5,6],dtype=torch.bfloat16)
    result=vectorized_quantize(e2,"e2m1")
    codes[:,0::2],codes[:,1::2]=result.payload&15,result.payload>>4
    assert codes[0,16:24].tolist() == list(range(8))


def test_zero_and_padding_are_canonical() -> None:
    source=torch.tensor([[-0.0]],dtype=torch.bfloat16)
    result=vectorized_quantize(source,"e0m3")
    assert result.alpha == 1.0
    assert not bool(result.payload.any())
    canonical=result.canonical_scales
    assert bool((canonical == UE4M3_ONE).all())


def test_native_scale_layout_roundtrip_and_offsets() -> None:
    mp,kp=144,128
    canonical=torch.arange(mp*(kp//16),dtype=torch.int32).remainder(0x7F).to(torch.uint8).reshape(mp,kp//16)
    native=native_scales_from_canonical(canonical,mp,kp)
    assert native.numel()==native_scale_size(mp,kp)
    assert torch.equal(canonical_scales_from_native(native,mp,kp),canonical)
    assert native[native_scale_offset(0,0,mp,kp)]==canonical[0,0]
    assert native[native_scale_offset(127,7,mp,kp)]==canonical[127,7]


def test_noncontiguous_and_nonfinite_rejected() -> None:
    source=torch.zeros((4,8),dtype=torch.bfloat16).T
    assert not source.is_contiguous()
    with pytest.raises(ValueError,match="contiguous"):
        vectorized_quantize(source,"e0m3")
    for value in (float("nan"),float("inf"),-float("inf")):
        with pytest.raises(ValueError,match="NaN or Inf"):
            scalar_quantize(torch.tensor([[value]],dtype=torch.float16),"e2m1")


def test_e4m3_subnormal_normal_max_and_saturation() -> None:
    assert e4m3_encode_scalar(0.0)==0x00
    assert e4m3_encode_scalar(2.0**-9)==0x01
    assert e4m3_encode_scalar(2.0**-6)==0x08
    assert e4m3_encode_scalar(448.0)==0x7E
    assert e4m3_encode_scalar(1000.0)==0x7E
