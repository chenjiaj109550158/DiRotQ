#!/usr/bin/env python3
"""All-slot operand-format patcher for one pinned BF16-epilogue CUBIN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernels.blackwell_e0_probe.e0_probe.patch_operand_format import parse_elf


BASELINE_SHA256 = "758138eff83d057a769f2ca246dad80926bb08f6756ca2513c7d9ab40dc637d8"
KERNEL_SYMBOL = "optimized_fp4_gemm_bf16_kernel"
TEXT_SIZE = 0x6780
OMMA_OFFSETS = (
    0x37B0, 0x3820, 0x3850, 0x3880, 0x38B0, 0x38F0, 0x3910, 0x3930,
    0x3950, 0x3970, 0x3990, 0x39B0, 0x39D0, 0x39F0, 0x3A10, 0x3A30,
    0x3C90, 0x3D40, 0x3D70, 0x3DA0, 0x3DD0, 0x3E00, 0x3E20, 0x3E40,
    0x3E60, 0x3E80, 0x3EA0, 0x3EC0, 0x3EE0, 0x3F00, 0x3F20, 0x3F40,
    0x3FA0, 0x4000, 0x4020, 0x4040, 0x4060, 0x4080, 0x40A0, 0x40C0,
    0x40F0, 0x4110, 0x4130, 0x4150, 0x4170, 0x4190, 0x41B0, 0x41D0,
    0x42D0, 0x4380, 0x43A0, 0x43C0, 0x43E0, 0x4400, 0x4420, 0x4440,
    0x4460, 0x4480, 0x44A0, 0x44C0, 0x44E0, 0x4500, 0x4520, 0x4540,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch(source: Path, destination: Path, variant: str) -> dict:
    if variant not in {"00", "01", "11"}:
        raise ValueError("BF16 prototype permits variants 00, 01, and 11 only")
    original = source.read_bytes()
    if digest(original) != BASELINE_SHA256:
        raise ValueError(f"unknown BF16 baseline SHA-256: {digest(original)}")
    elf = parse_elf(original)
    if elf["function"]["name"] != KERNEL_SYMBOL:
        raise ValueError("unexpected BF16 kernel symbol")
    text = elf["text"]
    if text["name"] != f".text.{KERNEL_SYMBOL}" or text["size"] != TEXT_SIZE:
        raise ValueError("BF16 kernel text contract changed")
    if len(OMMA_OFFSETS) != 64 or len(set(OMMA_OFFSETS)) != 64:
        raise ValueError("partial or duplicate BF16 OMMA slot table")
    bits = set()
    if variant[1] == "1":
        bits.add(78)
    if variant[0] == "1":
        bits.add(79)
    result = bytearray(original)
    absolute_offsets = []
    for relative in OMMA_OFFSETS:
        absolute = text["offset"] + relative
        word = int.from_bytes(original[absolute:absolute + 16], "little")
        if (word >> 78) & 3:
            raise ValueError("public BF16 baseline has nonzero operand-format bits")
        for bit in bits:
            word |= 1 << bit
        result[absolute:absolute + 16] = word.to_bytes(16, "little")
        absolute_offsets.append(absolute)
    expected_diff_bits = {offset * 8 + bit for offset in absolute_offsets for bit in bits}
    actual_diff_bits = set()
    for index, (before, after) in enumerate(zip(original, result)):
        difference = before ^ after
        actual_diff_bits.update(index * 8 + bit for bit in range(8) if difference & (1 << bit))
    if actual_diff_bits != expected_diff_bits:
        raise ValueError("whole-CUBIN bit diff is not the complete allowlisted target set")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(result)
    report = {
        "variant": variant, "kernel_symbol": KERNEL_SYMBOL,
        "baseline_sha256": BASELINE_SHA256, "output_sha256": digest(result),
        "cubin_size": len(original), "text_size": TEXT_SIZE,
        "omma_slots": len(OMMA_OFFSETS), "patched_bits": sorted(bits),
        "changed_bytes": sum(a != b for a, b in zip(original, result)),
        "changed_bits": len(actual_diff_bits), "size_unchanged": len(result) == len(original),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variant", required=True, choices=("00", "01", "11"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = patch(args.source, args.destination, args.variant)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
