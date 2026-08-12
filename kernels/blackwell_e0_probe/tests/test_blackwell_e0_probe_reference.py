from pathlib import Path

import pytest
import torch

from kernels.blackwell_e0_probe.generate_golden import CASE_NAMES, PAIRINGS, generate_documents
from kernels.blackwell_e0_probe.packing import (
    A_SCALE_SHAPE, A_SHAPE, B_SCALE_SHAPE, B_SHAPE, decode_e4m3_bytes,
    decode_nibbles, encode_e4m3_bytes, encode_values, pack_a, pack_a_scales,
    pack_b, pack_b_scales, pack_nibbles, unpack_a, unpack_a_scales, unpack_b,
    unpack_b_scales, unpack_nibbles,
)
from kernels.blackwell_e0_probe.verify_golden import verify_directory


GOLDEN = Path(__file__).resolve().parents[1] / "golden"


@pytest.mark.parametrize("fmt,levels", [
    ("e2m1", (0, .5, 1, 1.5, 2, 3, 4, 6)),
    ("e0m3", (0, 1, 2, 3, 4, 5, 6, 7)),
])
def test_nibble_round_trip_codebook_and_signed_zero(fmt, levels):
    codes = torch.arange(16, dtype=torch.uint8)
    values = decode_nibbles(codes, fmt)
    assert values[:8].tolist() == list(levels)
    assert values[8:].tolist() == [-x for x in levels]
    assert not torch.signbit(values[0]) and torch.signbit(values[8])
    assert torch.equal(encode_values(values, fmt), codes)
    assert torch.equal(unpack_nibbles(pack_nibbles(codes), 16), codes)


def test_e4m3_scale_byte_round_trip_and_known_encoding():
    values = torch.tensor([0.0, -0.0, 2**-9, 2**-6, 1.0, 384.0, 448.0])
    encoded = encode_e4m3_bytes(values)
    assert encoded.tolist() == [0x00, 0x00, 0x01, 0x08, 0x38, 0x7C, 0x7E]
    decoded = decode_e4m3_bytes(encoded)
    assert torch.equal(decoded, values.abs())
    assert not torch.signbit(decoded).any()
    with pytest.raises(ValueError, match="nonnegative"):
        decode_e4m3_bytes(torch.tensor([0x80], dtype=torch.uint8))
    with pytest.raises(ValueError, match="NaN"):
        decode_e4m3_bytes(torch.tensor([0x7F], dtype=torch.uint8))


def test_a_b_pack_round_trip_and_row_column_nibble_order():
    a = (torch.arange(A_SHAPE[0]*A_SHAPE[1]).reshape(A_SHAPE) % 16).to(torch.uint8)
    b = ((3*torch.arange(B_SHAPE[0]).reshape(-1, 1) + torch.arange(B_SHAPE[1])) % 16).to(torch.uint8)
    packed_a, packed_b = pack_a(a), pack_b(b)
    assert packed_a[0].item() == a[0, 0].item() | (a[0, 1].item() << 4)
    assert packed_b[0].item() == b[0, 0].item() | (b[1, 0].item() << 4)
    assert torch.equal(unpack_a(packed_a), a)
    assert torch.equal(unpack_b(packed_b), b)


def test_scale_index_mapping_and_round_trip():
    a = torch.tensor([.5, 1, 2, 4.]).repeat(A_SCALE_SHAPE[0], 1)
    b = torch.tensor([8., 16., 32., 64.]).repeat(B_SCALE_SHAPE[0], 1)
    assert torch.equal(unpack_a_scales(pack_a_scales(a)), a)
    assert torch.equal(unpack_b_scales(pack_b_scales(b)), b)
    assert pack_a_scales(a)[2].item() == encode_e4m3_bytes(torch.tensor([2.]))[0].item()
    assert pack_b_scales(b)[5].item() == encode_e4m3_bytes(torch.tensor([16.]))[0].item()


def test_golden_generation_is_deterministic_and_committed_files_are_current():
    first, second = generate_documents(), generate_documents()
    assert first == second
    assert set(first) == {f"{a}_x_{b}.json" for a, b in PAIRINGS}
    for name, content in first.items():
        assert (GOLDEN / name).read_text() == content


def test_all_golden_pairings_and_cases_reload_on_cpu():
    assert verify_directory(GOLDEN, device="cpu") == len(PAIRINGS) * len(CASE_NAMES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_golden_pairings_and_cases_reload_on_cuda():
    assert verify_directory(GOLDEN, device="cuda") == len(PAIRINGS) * len(CASE_NAMES)
