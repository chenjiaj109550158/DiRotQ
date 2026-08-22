import pytest
import torch
import torch.nn.functional as F

from utils.flux_w4a16_modulators import (
    GROUP_SIZE,
    PackedW4A16Linear,
    quantize_w4a16_weight,
    w4a16_provenance,
)
from utils.packed_int4_runtime import decode_weight_int4, unpack_signed_int4
from metrics.run_flux_shared_pca_w4a16_memory import completed_image


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_w4a16_pack_contract_and_cpu_reference(dtype):
    torch.manual_seed(7)
    weight = torch.randn(9, 70, dtype=dtype)
    payload, scales, report = quantize_w4a16_weight(weight)
    assert payload.dtype == torch.uint8
    assert scales.dtype == torch.bfloat16
    assert payload.shape == (9, 64)
    assert scales.shape == (9, 2)
    assert report["logical_shape"] == [9, 70]
    codes = unpack_signed_int4(payload, 128)
    assert int(codes.min()) >= -8 and int(codes.max()) <= 7

    module = PackedW4A16Linear(
        payload,
        scales,
        logical_k=70,
        bias=torch.randn(9, dtype=dtype),
        require_cuda=False,
    )
    x = torch.randn(2, 3, 70, dtype=dtype)
    decoded = decode_weight_int4(payload, scales, 70, GROUP_SIZE, dtype)
    expected = (
        F.linear(x, decoded, None).float() + module.bias.float()
    ).to(dtype)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert module.persistent_bytes == (
        payload.numel() + scales.numel() * 2 + module.bias.numel() * 2
    )


def test_w4a16_zero_groups_use_positive_safe_scale():
    weight = torch.zeros(4, 64, dtype=torch.bfloat16)
    payload, scales, report = quantize_w4a16_weight(weight)
    assert torch.count_nonzero(payload) == 0
    assert torch.equal(scales, torch.ones_like(scales))
    assert report["zero_groups"] == 4


def test_w4a16_provenance_distinguishes_flux_variants():
    assert w4a16_provenance("black-forest-labs/FLUX.1-dev")["model"] == "flux-dev"
    assert (
        w4a16_provenance("/cache/models--black-forest-labs--FLUX.1-schnell/snapshots/rev")[
            "model"
        ]
        == "flux-schnell"
    )


def test_w4a16_cuda_kernel_matches_decoded_bf16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(11)
    weight = torch.randn(37, 130, dtype=torch.bfloat16)
    payload, scales, _ = quantize_w4a16_weight(weight)
    module = PackedW4A16Linear(
        payload.cuda(),
        scales.cuda(),
        logical_k=130,
        bias=torch.randn(37, dtype=torch.bfloat16, device="cuda"),
        require_cuda=True,
    )
    x = torch.randn(3, 130, dtype=torch.bfloat16, device="cuda")
    decoded = decode_weight_int4(
        payload.cuda(), scales.cuda(), 130, GROUP_SIZE, torch.bfloat16
    )
    expected = F.linear(x, decoded, module.bias)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=0.25)
    assert actual.device.type == "cuda"


def test_w4a16_cuda_requirement_fails_closed():
    weight = torch.randn(2, 64, dtype=torch.bfloat16)
    payload, scales, _ = quantize_w4a16_weight(weight)
    module = PackedW4A16Linear(
        payload, scales, logical_k=64, bias=None, require_cuda=True
    )
    with pytest.raises(RuntimeError, match="forbids silent CPU fallback"):
        module(torch.randn(1, 64, dtype=torch.bfloat16))


def test_flux_nested_image_resume_requires_completed_single_png(tmp_path):
    log = tmp_path / "run.log"
    image_dir = tmp_path / "images"
    nested = image_dir / "category" / "sample.png"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"png sentinel")
    log.write_text("Inference-only peak CUDA memory: allocated=1 bytes, reserved=2 bytes\n")
    assert completed_image(log, image_dir) is None
    log.write_text(log.read_text() + "All done.\n")
    assert completed_image(log, image_dir) == nested
    extra = image_dir / "extra.png"
    extra.write_bytes(b"extra")
    assert completed_image(log, image_dir) is None
