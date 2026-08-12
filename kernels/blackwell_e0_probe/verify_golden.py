#!/usr/bin/env python3
"""Reload and validate every committed Blackwell FP4 golden vector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernels.blackwell_e0_probe.generate_golden import PAIRINGS  # noqa: E402
from kernels.blackwell_e0_probe.packing import (  # noqa: E402
    decode_nibbles, unpack_a, unpack_b,
)
from kernels.blackwell_e0_probe.reference import scalar_mma, vectorized_mma  # noqa: E402


def _u8(values, device="cpu"):
    return torch.tensor(values, dtype=torch.uint8, device=device)


def verify_case(case: dict, *, device: str = "cpu") -> None:
    metadata = case["metadata"]
    af, bf = metadata["a_format"], metadata["b_format"]
    pa, pb = _u8(case["packed_a_bytes"], device), _u8(case["packed_b_bytes"], device)
    sa = _u8(case["a_block_scale_bytes"], device)
    sb = _u8(case["b_block_scale_bytes"], device)
    expected_a = torch.tensor(case["logical_decoded_a"], device=device)
    expected_b = torch.tensor(case["logical_decoded_b"], device=device)
    actual_a, actual_b = decode_nibbles(unpack_a(pa), af), decode_nibbles(unpack_b(pb), bf)
    torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
    torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)
    raw, scaled = scalar_mma(pa, pb, sa, sb, af, bf, case["a_global_fp32_scale"], case["b_global_fp32_scale"])
    expected_raw = torch.tensor(case["expected_raw_fp32_mma"])
    expected_scaled = torch.tensor(case["expected_fully_scaled_fp32_output"])
    torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
    torch.testing.assert_close(scaled, expected_scaled, rtol=0, atol=0)
    vector_raw, vector_scaled = vectorized_mma(
        pa, pb, sa, sb, af, bf, case["a_global_fp32_scale"], case["b_global_fp32_scale"]
    )
    torch.testing.assert_close(vector_raw.cpu(), expected_raw, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(vector_scaled.cpu(), expected_scaled, rtol=2e-6, atol=1e-5)


def verify_directory(directory: Path, *, device: str = "cpu") -> int:
    count = 0
    for af, bf in PAIRINGS:
        path = directory / f"{af}_x_{bf}.json"
        document = json.loads(path.read_text())
        if document["pairing"] != f"{af}_x_{bf}":
            raise RuntimeError(f"pairing metadata mismatch in {path}")
        for case in document["cases"]:
            verify_case(case, device=device)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", type=Path, default=Path(__file__).with_name("golden"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requested but unavailable")
    count = verify_directory(args.golden_dir, device=args.device)
    print(f"Verified {count} golden cases on {args.device}")


if __name__ == "__main__":
    main()
