import hashlib
import json

import pytest
import torch

from utils.hardware_weight_fp4 import (
    E0_VALUES,
    E2_VALUES,
    decode_packing_record,
    frozen_block_scales,
    gptq_quantize_hardware_fixed,
    hardware_global_scale,
    make_packing_record,
    pack_nibbles,
    quantize_with_frozen_scales,
    tensor_sha256,
    unpack_nibbles,
    expected_metadata,
    metadata_path,
    packing_path,
    validate_metadata,
)


EMPTY_HASH = hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_global_scale_low_only_and_maximum_scales(dtype):
    low = torch.zeros(3, 19, dtype=dtype)
    low[1, 7] = 2688
    alpha = hardware_global_scale(low)
    assert alpha.dtype == torch.float32
    assert alpha.item() == 1.0
    e2, _ = frozen_block_scales(low, "hardware-fixed-e2", alpha)
    e0, _ = frozen_block_scales(low, "hardware-fixed-e0", alpha)
    assert e2.max().item() == 448.0
    assert e0.max().item() == 384.0

    # A hypothetical high tail is deliberately not passed to the low-only scale.
    high = torch.full((3, 5), 65504, dtype=dtype)
    assert hardware_global_scale(low).item() == 1.0
    assert hardware_global_scale(torch.cat([low, high], 1)).item() > 1.0


def test_zero_and_e4m3_underflow_are_safe():
    zero = torch.zeros(2, 17)
    alpha = hardware_global_scale(zero)
    assert alpha.item() == 1.0
    for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
        scales, _ = frozen_block_scales(zero, fmt, alpha)
        assert torch.all(scales == 1)
        out = quantize_with_frozen_scales(zero, fmt, alpha, scales)
        assert torch.equal(out, zero)
        assert torch.isfinite(out).all()

    source = torch.zeros(1, 32)
    source[0, 0] = 1.0
    source[0, 16] = 1e-10
    alpha = hardware_global_scale(source)
    scales, raw = frozen_block_scales(source, "hardware-fixed-e2", alpha)
    assert raw[0, 1] > 0
    assert scales[0, 1].item() == 2.0 ** -9
    out = quantize_with_frozen_scales(source, "hardware-fixed-e2", alpha, scales)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "fmt,values",
    [("hardware-fixed-e2", E2_VALUES), ("hardware-fixed-e0", E0_VALUES)],
)
def test_codebooks_and_packing_roundtrip(fmt, values):
    contiguous = torch.tensor(values, dtype=torch.float32).repeat(3, 2)
    source = torch.stack([contiguous, contiguous], dim=-1)[..., 0][:, :-3]
    assert not source.is_contiguous()
    alpha = hardware_global_scale(source)
    scales, raw = frozen_block_scales(source, fmt, alpha)
    q = quantize_with_frozen_scales(source, fmt, alpha, scales)
    record = make_packing_record(
        q, fmt,
        {"global_scale": alpha, "block_scales": scales.to(torch.float8_e4m3fn),
         "raw_block_scales": raw},
        high_branch_hash=EMPTY_HASH,
    )
    decoded = decode_packing_record(record)
    assert decoded.shape == source.shape
    assert torch.equal(decoded, q)
    assert record["group_size"] == 16
    assert record["logical_shape"] == [source.shape[1], source.shape[0]]
    assert record["reconstructed_low_hash_fp32"] == tensor_sha256(decoded)


def test_nibble_pack_roundtrip_with_odd_count():
    indices = torch.tensor([[0, 1, 14, 7, 3]], dtype=torch.uint8)
    assert torch.equal(unpack_nibbles(pack_nibbles(indices), 5), indices)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reconstructed_native_dtype_roundtrip(dtype):
    torch.manual_seed(4)
    source = torch.randn(5, 35, dtype=dtype)
    for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
        alpha = hardware_global_scale(source)
        scales, raw = frozen_block_scales(source, fmt, alpha)
        q = quantize_with_frozen_scales(source, fmt, alpha, scales)
        record = make_packing_record(
            q, fmt,
            {"global_scale": alpha, "block_scales": scales.to(torch.float8_e4m3fn),
             "raw_block_scales": raw},
            high_branch_hash=EMPTY_HASH,
        )
        decoded = decode_packing_record(record, dtype=dtype)
        assert decoded.dtype == dtype
        assert torch.equal(decoded, q.to(dtype))
        assert torch.isfinite(decoded).all()


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        hardware_global_scale(torch.tensor([[float("inf")]]))


def test_hardware_e2_e0_cache_metadata_isolated(tmp_path):
    basis = tmp_path / "basis.pt"
    rotation = tmp_path / "rotation.pt"
    hessian = tmp_path / "hessian.pt"
    basis.write_bytes(b"basis")
    rotation.write_bytes(b"rotation")
    hessian.write_bytes(b"hessian")
    common = dict(
        model="sana-1.6b", calibration_count=8, damp_pct=.01,
        basis_path=basis, rotation_path=rotation, hessian_cache=hessian,
        skip_layers=["attn2.to_k"],
    )
    e2 = expected_metadata(fmt="hardware-fixed-e2", **common)
    e0 = expected_metadata(fmt="hardware-fixed-e0", **common)
    assert e2["weight_scale_semantics"] == e0["weight_scale_semantics"]
    assert e2["hessian_cache_sha256"] == e0["hessian_cache_sha256"]
    assert e2["weight_format"] != e0["weight_format"]

    cache = tmp_path / "hardware-fixed-e2.pt"
    cache.write_bytes(b"cache")
    packing_path(cache).write_bytes(b"packing")
    saved = {
        **e2, "cache_sha256": hashlib.sha256(b"cache").hexdigest(),
        "packing_sha256": hashlib.sha256(b"packing").hexdigest(),
    }
    metadata_path(cache).write_text(json.dumps(saved))
    assert validate_metadata(cache, e2)["weight_format"] == "hardware-fixed-e2"
    with pytest.raises(RuntimeError, match="weight_format"):
        validate_metadata(cache, e0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", ["hardware-fixed-e2", "hardware-fixed-e0"])
def test_gpu_gptq_uses_frozen_hardware_scales_without_cpu_fallback(fmt):
    torch.manual_seed(17)
    source = torch.randn(8, 64, device="cuda")
    x = torch.randn(96, 64, device="cuda")
    hessian = 2.0 / x.shape[0] * x.T @ x
    q, stats, frozen = gptq_quantize_hardware_fixed(source, hessian, fmt)
    assert q is not None
    assert q.device.type == "cuda"
    assert stats["gptq_status"] == "gptq"
    assert stats["failure"] is None
    assert frozen["global_scale"].dtype == torch.float32
    assert frozen["block_scales"].dtype == torch.float8_e4m3fn
    decoded = decode_packing_record(
        make_packing_record(q, fmt, frozen, high_branch_hash=EMPTY_HASH),
        device="cuda",
    )
    assert torch.equal(decoded, q)


def test_gpu_gptq_refuses_silent_cpu_fallback():
    source = torch.randn(3, 16)
    with pytest.raises(RuntimeError, match="silent CPU fallback"):
        gptq_quantize_hardware_fixed(source, torch.eye(16), "hardware-fixed-e2")
