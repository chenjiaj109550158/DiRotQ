import gc
import json
import os
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from utils.quant_utils import ActQuantizer
from utils.real_tile_export import (
    CapturedCase,
    CaptureSpec,
    RealTileExportController,
    UE4M3_ONE_BYTE,
    canonical_a,
    canonical_b,
    encode_sign_magnitude,
    sidecar_case_arrays,
    transcode_sidecar_indices_to_sign_magnitude,
    verify_with_receiver,
)
from utils.tilemixfp4_utils import E0M3_MAGNITUDES, fake_quantize_e0m3


def _unpack_rows(payload, k):
    packed = torch.from_numpy(payload)
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8)
    out[:, 0::2] = packed & 0xF
    out[:, 1::2] = packed >> 4
    return out[:, :k]


def _pack_sidecar_indices(indices):
    if indices.shape[1] % 2:
        indices = torch.nn.functional.pad(indices, (0, 1))
    return indices[:, 0::2] | (indices[:, 1::2] << 4)


def test_capture_uses_complete_low_call_alpha_and_excludes_tail():
    quantizer = ActQuantizer()
    quantizer.configure(
        bits=4, groupsize=16, sym=True, high_bits_length=3, quant_dtype="e0m3"
    )
    events = []
    quantizer.real_tile_capture = events.append
    x = torch.zeros(20, 20, dtype=torch.bfloat16)
    x[:16, :17] = 1
    x[19, 16] = 2688
    x[:, 17:] = 32768  # high-precision tail cannot enter alpha
    out = quantizer(x)
    assert len(events) == 1
    event = events[0]
    assert event["low_k"] == 17
    assert float(event["global_scale"]) == 1.0
    selected_only_alpha = float(x[:16, :17].float().abs().max() / 2688)
    assert selected_only_alpha != float(event["global_scale"])
    assert torch.equal(out[..., 17:], x[..., 17:])


def test_capture_payload_scales_are_the_runtime_candidate_and_output_unchanged():
    torch.manual_seed(51)
    x = torch.randn(3, 29, dtype=torch.bfloat16)
    baseline = fake_quantize_e0m3(x)
    events = []
    captured = fake_quantize_e0m3(x, capture_hook=events.append)
    assert torch.equal(captured, baseline)
    event = events[0]
    codes = event["logical_codes"]
    scale = event["block_scales"].unsqueeze(-1)
    reconstructed = (codes * scale * event["global_scale"]).reshape(3, -1)[:, :29]
    assert torch.equal(reconstructed.to(x.dtype), captured)
    payload = encode_sign_magnitude(codes, E0M3_MAGNITUDES)
    assert set(torch.unique(payload).tolist()).issubset(set(range(16)))


def test_capture_callback_does_not_extend_full_activation_lifetime():
    retained = []

    def callback(event):
        retained.append(event["logical_codes"][0, 0].clone())

    x = torch.randn(19, 67)
    reference = weakref.ref(x)
    fake_quantize_e0m3(x, capture_hook=callback)
    del x
    gc.collect()
    assert reference() is None
    assert retained[0].numel() == 16


@pytest.mark.parametrize("m,n,k", [(16, 8, 64), (17, 9, 65), (3, 2, 17)])
def test_canonical_padding_nibble_order_and_ue4m3_one(m, n, k):
    blocks = (k + 15) // 16
    a_codes = torch.arange(m * k, dtype=torch.int64).reshape(m, k).remainder(16).byte()
    b_codes = torch.arange(n * k, dtype=torch.int64).reshape(n, k).remainder(16).byte()
    a_scales = torch.full((m, blocks), 0x40, dtype=torch.uint8)
    b_scales = torch.full((n, blocks), 0x48, dtype=torch.uint8)
    ap, ass = canonical_a(a_codes, a_scales, m, k)
    bp, bss = canonical_b(b_codes, b_scales, n, k)
    assert torch.equal(_unpack_rows(ap, ((k + 63) // 64) * 64)[:m, :k], a_codes)
    assert torch.equal(_unpack_rows(bp, ((k + 63) // 64) * 64)[:n, :k], b_codes)
    assert (_unpack_rows(ap, ((k + 63) // 64) * 64)[m:] == 0).all()
    assert (_unpack_rows(bp, ((k + 63) // 64) * 64)[n:] == 0).all()
    assert (ass[:m, blocks:] == UE4M3_ONE_BYTE).all()
    assert (bss[:n, blocks:] == UE4M3_ONE_BYTE).all()


def test_weight_sidecar_is_transcoded_from_payload_not_requantized_bf16():
    # Sorted sidecar indices: -7,-1,0,+1,+7 -> receiver f,9,0,1,7.
    raw = torch.tensor([[0, 6, 7, 8, 14] + [7] * 27], dtype=torch.uint8)
    expected = torch.tensor([[15, 9, 0, 1, 7] + [0] * 27], dtype=torch.uint8)
    assert torch.equal(transcode_sidecar_indices_to_sign_magnitude(raw), expected)
    record = {
        "stored_shape": [1, 17],
        "group_size": 16,
        "packed_payload": _pack_sidecar_indices(raw),
        "block_scales": torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn),
        "global_scale": torch.tensor(0.25, dtype=torch.float32),
    }
    payload, scales, alpha = sidecar_case_arrays(
        record, column_start=0, n=1, k=17
    )
    decoded_codes = _unpack_rows(payload, 64)[0, :17]
    assert torch.equal(decoded_codes, expected[0, :17])
    assert scales[0, :2].tolist() == [0x38, 0x40]
    assert alpha == 0.25


def test_runtime_bf16_expected_is_full_n_then_slice_semantics():
    torch.manual_seed(52)
    a = torch.randn(17, 65, dtype=torch.bfloat16)
    w = torch.randn(13, 65, dtype=torch.bfloat16)
    full = torch.matmul(a, w.T).to(torch.bfloat16)
    expected = full[:, 2:11].float().numpy()
    assert expected.dtype == np.float32
    assert np.array_equal(expected, torch.matmul(a, w.T)[:, 2:11].to(torch.bfloat16).float().numpy())


def _receiver_root():
    candidate = Path(os.environ.get(
        "DIROTQ_BLACKWELL_RECEIVER", "/tmp/dirotq-blackwell-receiver-5dedc043"
    ))
    if not (candidate / "kernels/blackwell_e0_probe/real_tile_handoff/verify_package.py").is_file():
        pytest.skip("pinned Blackwell receiver worktree is unavailable")
    return candidate


@pytest.mark.parametrize("pairing,fmt", [("e0xe2", "hardware-fixed-e2"), ("e0xe0", "hardware-fixed-e0")])
def test_receiver_verifier_interoperability_and_manifest_hashes(tmp_path, pairing, fmt):
    spec = CaptureSpec(
        "tail", "transformer_blocks.0.attn1.to_q", 0, 0, 0, 0, 17, 9
    )
    k = 65
    a_values = torch.arange(spec.m * k).reshape(spec.m, k).remainder(15).float() - 7
    # E0 codebook is integer 0..7 with sign.
    a_values = a_values.clamp(-7, 7)
    a_codes = encode_sign_magnitude(a_values, E0M3_MAGNITUDES)
    a_scales = torch.full((spec.m, (k + 15) // 16), 0x38, dtype=torch.uint8)
    sidecar_indices = torch.arange(spec.n * k).reshape(spec.n, k).remainder(15).byte()
    sidecar_padded = torch.full((spec.n, 80), 7, dtype=torch.uint8)
    sidecar_padded[:, :k] = sidecar_indices
    record = {
        "format": fmt,
        "stored_shape": [spec.n, k], "logical_shape": [k, spec.n],
        "group_size": 16,
        "packed_payload": _pack_sidecar_indices(sidecar_padded),
        "block_scales": torch.ones(spec.n, (k + 15) // 16, dtype=torch.float8_e4m3fn),
        "global_scale": torch.tensor(0.5),
        "high_branch_hash": "x",
    }
    controller = RealTileExportController.__new__(RealTileExportController)
    controller.output_root = tmp_path
    controller.receiver_root = _receiver_root()
    controller.specs = (spec,)
    controller.prompt_id = "000438f99177213e07a9b9c875248eea17b8c8c6"
    controller.model_name = "unit-test"
    controller.model_revision = "unit-test-revision"
    controller.producer_commit = "0" * 40
    controller.basis_sha256 = "1" * 64
    controller.rotation_sha256 = "2" * 64
    controller.cache_hashes = {pairing: "3" * 64}
    controller.sidecar_hashes = {pairing: "4" * 64}
    controller.records = {pairing: {spec.layer_name: record}}
    controller.captured = {
        spec.case_id: CapturedCase(
            spec, 999.0, 0, 31, k, 0.25, 672.0,
            a_codes, a_scales, torch.zeros(spec.m, k, dtype=torch.bfloat16),
            {pairing: np.zeros((spec.m, spec.n), dtype=np.float32)},
        )
    }
    package, _ = controller._write_package(pairing, f"package-{pairing}", "2026-08-13T00:00:00+08:00")
    report = verify_with_receiver(controller.receiver_root, package)
    assert report["passed"] and report["case_count"] == 1
    manifest = json.loads((package / "manifest.json").read_text())
    for file_record in manifest["cases"][0]["files"].values():
        assert len(file_record["sha256"]) == 64


def test_four_case_paired_package_smoke_reuses_identical_a(tmp_path):
    receiver = _receiver_root()
    specs = tuple(
        CaptureSpec(
            f"case_{index}", f"synthetic.layer.{index}", index, 0,
            0, 0, 16 if index % 2 == 0 else 17, 8 if index % 2 == 0 else 9,
        )
        for index in range(4)
    )
    controller = RealTileExportController.__new__(RealTileExportController)
    controller.output_root = tmp_path
    controller.receiver_root = receiver
    controller.specs = specs
    controller.prompt_id = "000438f99177213e07a9b9c875248eea17b8c8c6"
    controller.model_name = "four-case-smoke"
    controller.model_revision = "receiver-interoperability"
    controller.producer_commit = "0" * 40
    controller.basis_sha256 = "1" * 64
    controller.rotation_sha256 = "2" * 64
    controller.cache_hashes = {"e0xe2": "3" * 64, "e0xe0": "4" * 64}
    controller.sidecar_hashes = {"e0xe2": "5" * 64, "e0xe0": "6" * 64}
    controller.records = {"e0xe2": {}, "e0xe0": {}}
    controller.captured = {}
    k = 65
    for index, spec in enumerate(specs):
        for pairing, fmt in (("e0xe2", "hardware-fixed-e2"),
                             ("e0xe0", "hardware-fixed-e0")):
            sidecar = torch.arange(spec.n * k).reshape(spec.n, k).remainder(15).byte()
            padded = torch.full((spec.n, 80), 7, dtype=torch.uint8)
            padded[:, :k] = sidecar
            controller.records[pairing][spec.layer_name] = {
                "format": fmt, "stored_shape": [spec.n, k],
                "logical_shape": [k, spec.n], "group_size": 16,
                "packed_payload": _pack_sidecar_indices(padded),
                "block_scales": torch.ones(
                    spec.n, (k + 15) // 16, dtype=torch.float8_e4m3fn
                ),
                "global_scale": torch.tensor(0.5), "high_branch_hash": "x",
            }
        logical = (
            torch.arange(spec.m * k).reshape(spec.m, k).remainder(15).float() - 7
        ).clamp(-7, 7)
        controller.captured[spec.case_id] = CapturedCase(
            spec, float(1000 - index), index, 31, k, 0.25, 672.0,
            encode_sign_magnitude(logical, E0M3_MAGNITUDES),
            torch.full((spec.m, (k + 15) // 16), 0x38, dtype=torch.uint8),
            torch.zeros(spec.m, k, dtype=torch.bfloat16),
            {
                "e0xe2": np.zeros((spec.m, spec.n), dtype=np.float32),
                "e0xe0": np.zeros((spec.m, spec.n), dtype=np.float32),
            },
        )
    e2, _ = controller._write_package(
        "e0xe2", "four-case-e0xe2", "2026-08-13T00:00:00+08:00"
    )
    e0, _ = controller._write_package(
        "e0xe0", "four-case-e0xe0", "2026-08-13T00:00:00+08:00"
    )
    assert verify_with_receiver(receiver, e2)["case_count"] == 4
    assert verify_with_receiver(receiver, e0)["case_count"] == 4
    for spec in specs:
        for filename in ("a_payload.npy", "a_scales.npy"):
            left = (e2 / "cases" / spec.case_id / filename).read_bytes()
            right = (e0 / "cases" / spec.case_id / filename).read_bytes()
            assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_capture_matches_cpu_and_has_no_cpu_fallback():
    torch.manual_seed(53)
    x = torch.randn(17, 65, device="cuda", dtype=torch.bfloat16)
    events = []
    result = fake_quantize_e0m3(x, capture_hook=events.append)
    assert result.device.type == "cuda"
    assert events[0]["device"].type == "cuda"
    assert torch.equal(result.cpu(), fake_quantize_e0m3(x.cpu()))
