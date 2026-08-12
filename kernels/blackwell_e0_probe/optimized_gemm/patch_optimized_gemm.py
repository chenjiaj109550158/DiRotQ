#!/usr/bin/env python3
"""Baseline-pinned all-slot patcher for the optimized SM120 NVFP4 GEMM.

This is deliberately not a general CUBIN rewriter.  It accepts one CUBIN from
one compiler/CUTLASS/kernel configuration and changes only operand-format bits
78/79 of all 64 pre-audited OMMA instructions in its sole kernel text section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from kernels.blackwell_e0_probe.e0_probe.patch_operand_format import parse_elf


BASELINE_SHA256 = "597f7d6fb18fbcd2c39d033126098b39854128b4463940405d6df88a2667174d"
KERNEL_SYMBOL = "optimized_fp4_gemm_kernel"
TEXT_SIZE = 26112
CANDIDATE_BITS = (78, 79)
OMMA_OFFSETS_IN_TEXT = (
    0x3790, 0x3800, 0x3830, 0x3860, 0x3890, 0x38D0, 0x38F0, 0x3910,
    0x3930, 0x3950, 0x3970, 0x3990, 0x39B0, 0x39D0, 0x39F0, 0x3A10,
    0x3C70, 0x3D10, 0x3D40, 0x3D70, 0x3DA0, 0x3DD0, 0x3DF0, 0x3E10,
    0x3E30, 0x3E50, 0x3E70, 0x3E90, 0x3EB0, 0x3ED0, 0x3EF0, 0x3F10,
    0x3F70, 0x3FD0, 0x3FF0, 0x4010, 0x4030, 0x4050, 0x4070, 0x4090,
    0x40C0, 0x40E0, 0x4100, 0x4120, 0x4140, 0x4160, 0x4180, 0x41A0,
    0x42A0, 0x4350, 0x4370, 0x4390, 0x43B0, 0x43D0, 0x43F0, 0x4410,
    0x4430, 0x4450, 0x4470, 0x4490, 0x44B0, 0x44D0, 0x44F0, 0x4510,
)
EXPECTED_OMMA_COUNT = 64


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def variant_bits(variant: str) -> set[int]:
    if variant not in {"00", "01", "11"}:
        raise ValueError("optimized prototype only permits variants 00, 01, and 11")
    bits: set[int] = set()
    if variant[1] == "1":
        bits.add(78)
    if variant[0] == "1":
        bits.add(79)
    return bits


def analyze_baseline(data: bytes) -> dict:
    digest = sha256(data)
    if digest != BASELINE_SHA256:
        raise ValueError(f"unknown optimized baseline SHA-256: {digest}")
    elf = parse_elf(data)
    if elf["function"]["name"] != KERNEL_SYMBOL:
        raise ValueError("unexpected optimized kernel symbol")
    text = elf["text"]
    if text["name"] != f".text.{KERNEL_SYMBOL}" or text["size"] != TEXT_SIZE:
        raise ValueError("optimized kernel text contract changed")
    if len(OMMA_OFFSETS_IN_TEXT) != EXPECTED_OMMA_COUNT or len(set(OMMA_OFFSETS_IN_TEXT)) != EXPECTED_OMMA_COUNT:
        raise ValueError("invalid pinned OMMA slot table")
    absolute_offsets = []
    before = []
    for relative in OMMA_OFFSETS_IN_TEXT:
        if relative < 0 or relative + 16 > text["size"] or relative % 16:
            raise ValueError("pinned OMMA slot lies outside aligned kernel text")
        absolute = text["offset"] + relative
        instruction = data[absolute:absolute + 16]
        word = int.from_bytes(instruction, "little")
        if any((word >> bit) & 1 for bit in CANDIDATE_BITS):
            raise ValueError("baseline OMMA candidate bit is already set")
        absolute_offsets.append(absolute)
        before.append(instruction.hex())
    return {
        "elf": elf,
        "instruction_file_offsets": absolute_offsets,
        "instruction_before_hex": before,
    }


def validate_variant(original: bytes, patched: bytes, variant: str,
                     offsets: list[int]) -> dict:
    if len(offsets) != EXPECTED_OMMA_COUNT or len(set(offsets)) != EXPECTED_OMMA_COUNT:
        raise ValueError("partial or duplicate OMMA target list rejected")
    if len(original) != len(patched):
        raise ValueError("CUBIN size changed")
    bits = variant_bits(variant)
    expected = bytearray(original)
    for offset in offsets:
        word = int.from_bytes(original[offset:offset + 16], "little")
        for bit in bits:
            word |= 1 << bit
        expected[offset:offset + 16] = word.to_bytes(16, "little")
    if bytes(expected) != patched:
        unexpected = [index for index, pair in enumerate(zip(expected, patched))
                      if pair[0] != pair[1]]
        raise ValueError(f"binary differs outside complete OMMA patch: {unexpected[:8]}")

    byte_diffs = [index for index, pair in enumerate(zip(original, patched))
                  if pair[0] != pair[1]]
    file_bit_diffs = []
    for index in byte_diffs:
        difference = original[index] ^ patched[index]
        file_bit_diffs.extend(index * 8 + bit for bit in range(8)
                              if difference & (1 << bit))
    allowed = {offset * 8 + bit for offset in offsets for bit in CANDIDATE_BITS}
    if not set(file_bit_diffs).issubset(allowed):
        raise ValueError("non-allowlisted instruction bit changed")
    expected_diff_count = EXPECTED_OMMA_COUNT * len(bits)
    if len(file_bit_diffs) != expected_diff_count:
        raise ValueError("not all target OMMA slots received all requested bits")
    return {"byte_offsets": byte_diffs, "file_bit_offsets": file_bit_diffs}


def patch_cubin(source: Path, destination: Path, variant: str) -> dict:
    original = source.read_bytes()
    analysis = analyze_baseline(original)
    offsets = analysis["instruction_file_offsets"]
    patched = bytearray(original)
    bits = variant_bits(variant)
    for offset in offsets:
        word = int.from_bytes(original[offset:offset + 16], "little")
        for bit in bits:
            word |= 1 << bit
        patched[offset:offset + 16] = word.to_bytes(16, "little")
    patched_bytes = bytes(patched)
    diff = validate_variant(original, patched_bytes, variant, offsets)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,
                                         prefix=f".{destination.name}.",
                                         delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(patched_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o444)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    text = analysis["elf"]["text"]
    after = [patched_bytes[offset:offset + 16].hex() for offset in offsets]
    return {
        "variant": variant,
        "variant_spelling": "bit79_bit78",
        "baseline_sha256": sha256(original),
        "patched_sha256": sha256(patched_bytes),
        "kernel_symbol": KERNEL_SYMBOL,
        "target": "sm_120a",
        "text_section": text["name"],
        "text_file_offset": text["offset"],
        "text_size": text["size"],
        "omma_count": len(offsets),
        "instruction_offsets_in_text": list(OMMA_OFFSETS_IN_TEXT),
        "instruction_file_offsets": offsets,
        "instruction_before_hex": analysis["instruction_before_hex"],
        "instruction_after_hex": after,
        "instruction_bits_set_per_slot": sorted(bits),
        "whole_cubin_byte_diff": diff["byte_offsets"],
        "whole_cubin_file_bit_diff": diff["file_bit_offsets"],
        "size_unchanged": len(original) == len(patched_bytes),
        "all_slots_patched": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variant", required=True, choices=("00", "01", "11"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = patch_cubin(args.source, args.destination, args.variant)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
