#!/usr/bin/env python3
"""Generate compact synthetic golden vectors without model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernels.blackwell_e0_probe.packing import (  # noqa: E402
    A_SCALE_SHAPE, A_SHAPE, B_SCALE_SHAPE, B_SHAPE, E0M3_MAGNITUDES,
    E2M1_MAGNITUDES, HARDWARE_ENCODING_TO_VERIFY, K, M, N,
    decode_nibbles, pack_a, pack_a_scales, pack_b, pack_b_scales,
)
from kernels.blackwell_e0_probe.reference import scalar_mma, vectorized_mma  # noqa: E402


CONTRACT_VERSION = "blackwell-e0-probe-logical-v1"
PAIRINGS = (("e2m1", "e2m1"), ("e0m3", "e2m1"), ("e2m1", "e0m3"), ("e0m3", "e0m3"))
CASE_NAMES = (
    "all_zero", "positive_negative_codebooks", "alternating_signs",
    "per_k_block_scales", "scale_extremes", "deterministic_random",
    "layout_sensitive", "accumulation_cancellation",
)


def _codes(name: str, operand: str, seed: int) -> torch.Tensor:
    shape = A_SHAPE if operand == "a" else B_SHAPE
    count = shape[0] * shape[1]
    linear = torch.arange(count).reshape(shape)
    if name == "all_zero":
        return torch.zeros(shape, dtype=torch.uint8)
    if name == "positive_negative_codebooks":
        return (linear % 16).to(torch.uint8)
    if name == "alternating_signs":
        return ((linear % 8) | ((linear % 2) << 3)).to(torch.uint8)
    if name in {"per_k_block_scales", "scale_extremes"}:
        k_index = torch.arange(K).reshape(1, K) if operand == "a" else torch.arange(K).reshape(K, 1)
        base = (1 + k_index % 7).expand(shape)
        sign = ((linear // 7) % 2) << 3
        return (base | sign).to(torch.uint8)
    if name == "deterministic_random":
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(0, 16, shape, generator=generator, dtype=torch.uint8)
    if name == "layout_sensitive":
        if operand == "a":
            m = torch.arange(M).reshape(M, 1)
            k = torch.arange(K).reshape(1, K)
            return (((3*m + 5*k) % 8) | (((m + 2*k) % 3 == 0).to(torch.int64) << 3)).to(torch.uint8)
        k = torch.arange(K).reshape(K, 1)
        n = torch.arange(N).reshape(1, N)
        return (((7*k + 3*n) % 8) | (((2*k + n) % 5 == 0).to(torch.int64) << 3)).to(torch.uint8)
    if name == "accumulation_cancellation":
        if operand == "a":
            k = torch.arange(K).reshape(1, K)
            return ((torch.full(shape, 6) | ((k % 2) << 3)).to(torch.uint8))
        return torch.full(shape, 6, dtype=torch.uint8)
    raise ValueError(name)


def _scales(name: str, operand: str) -> torch.Tensor:
    shape = A_SCALE_SHAPE if operand == "a" else B_SCALE_SHAPE
    if name == "scale_extremes":
        base = torch.tensor([2.0**-9, 2.0**-6, 384.0, 448.0])
    elif name == "per_k_block_scales":
        base = torch.tensor([0.5, 1.0, 2.0, 4.0])
    elif name == "deterministic_random":
        base = torch.tensor([0.125, 0.75, 3.0, 16.0])
    else:
        base = torch.ones(4)
    return base.repeat(shape[0], 1)


def _globals(name: str) -> tuple[float, float]:
    return {
        "all_zero": (1.0, 1.0),
        "positive_negative_codebooks": (0.5, 2.0),
        "alternating_signs": (0.25, 1.5),
        "per_k_block_scales": (0.125, 0.75),
        "scale_extremes": (2.0**-8, 2.0**-4),
        "deterministic_random": (0.03125, 3.25),
        "layout_sensitive": (1.25, 0.375),
        "accumulation_cancellation": (2.0, 0.5),
    }[name]


def make_case(name: str, a_format: str, b_format: str, pairing_index: int) -> dict:
    a_codes = _codes(name, "a", 20260812 + 10*pairing_index)
    b_codes = _codes(name, "b", 20260813 + 10*pairing_index)
    a_scale_values, b_scale_values = _scales(name, "a"), _scales(name, "b")
    packed_a, packed_b = pack_a(a_codes), pack_b(b_codes)
    a_scale_bytes, b_scale_bytes = pack_a_scales(a_scale_values), pack_b_scales(b_scale_values)
    ga, gb = _globals(name)
    raw, scaled = scalar_mma(
        packed_a, packed_b, a_scale_bytes, b_scale_bytes,
        a_format, b_format, ga, gb,
    )
    vector_raw, vector_scaled = vectorized_mma(
        packed_a, packed_b, a_scale_bytes, b_scale_bytes,
        a_format, b_format, ga, gb,
    )
    torch.testing.assert_close(vector_raw, raw, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(vector_scaled, scaled, rtol=2e-6, atol=1e-5)
    return {
        "name": name,
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "shape": {"m": M, "n": N, "k": K},
            "a_format": a_format, "b_format": b_format,
            "a_layout": "logical row-major; packed stream index=m*64+k",
            "b_layout": "logical column-major; packed stream index=n*64+k",
            "nibble_order": "first stream element=low nibble; second=high nibble",
            "a_scale_index": "m*4+floor(k/16)",
            "b_scale_index": "n*4+floor(k/16)",
            "accumulator": "sequential IEEE FP32 scalar reference",
            "global_scale_application": "after block-scaled raw FP32 MMA",
            "e0_sass_encoding": HARDWARE_ENCODING_TO_VERIFY,
        },
        "logical_decoded_a": decode_nibbles(a_codes, a_format).tolist(),
        "logical_decoded_b": decode_nibbles(b_codes, b_format).tolist(),
        "packed_a_bytes": packed_a.tolist(), "packed_b_bytes": packed_b.tolist(),
        "a_block_scale_bytes": a_scale_bytes.tolist(),
        "b_block_scale_bytes": b_scale_bytes.tolist(),
        "a_global_fp32_scale": ga, "b_global_fp32_scale": gb,
        "expected_raw_fp32_mma": raw.tolist(),
        "expected_fully_scaled_fp32_output": scaled.tolist(),
    }


def generate_documents() -> dict[str, str]:
    documents = {}
    for pairing_index, (a_format, b_format) in enumerate(PAIRINGS):
        payload = {
            "contract_version": CONTRACT_VERSION,
            "pairing": f"{a_format}_x_{b_format}",
            "codebooks": {"e2m1_magnitudes": E2M1_MAGNITUDES, "e0m3_magnitudes": E0M3_MAGNITUDES},
            "cases": [make_case(name, a_format, b_format, pairing_index) for name in CASE_NAMES],
        }
        documents[f"{a_format}_x_{b_format}.json"] = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("golden"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    documents = generate_documents()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in documents.items():
        path = args.output_dir / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise RuntimeError(f"golden file is stale or missing: {path}")
        else:
            path.write_text(content)
    print(f"{'Verified' if args.check else 'Wrote'} {len(documents)} golden files in {args.output_dir}")


if __name__ == "__main__":
    main()
