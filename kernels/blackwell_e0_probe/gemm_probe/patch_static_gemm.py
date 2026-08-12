#!/usr/bin/env python3
"""Strict baseline-pinned operand-format patcher for static_fp4_gemm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from kernels.blackwell_e0_probe.e0_probe.patch_operand_format import parse_elf


BASELINE_SHA256 = "a4d0d3e1be4fb47365e8659fb99025e61b5b471ca08adbc707906157df0e91e3"
EXPECTED_INSTRUCTION = bytes.fromhex("7f740c10202db0720c3e040000e20f00")
CANDIDATE_BITS = (78, 79)
EXPECTED_OMMA_COUNT = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _variant_bits(variant: str) -> set[int]:
    if variant not in {"00", "01", "10", "11"}:
        raise ValueError("variant must be 00, 01, 10, or 11")
    result = set()
    if variant[1] == "1":
        result.add(78)
    if variant[0] == "1":
        result.add(79)
    return result


def variant_instruction(variant: str) -> bytes:
    result = bytearray(EXPECTED_INSTRUCTION)
    for bit in _variant_bits(variant):
        result[bit // 8] |= 1 << (bit % 8)
    return bytes(result)


def _clear_candidate_bits(instruction: bytes) -> bytes:
    result = bytearray(instruction)
    for bit in CANDIDATE_BITS:
        result[bit // 8] &= ~(1 << (bit % 8))
    return bytes(result)


def analyze_baseline(data: bytes) -> dict:
    digest = _sha256(data)
    if digest != BASELINE_SHA256:
        raise ValueError(f"unknown GEMM baseline SHA-256: {digest}")
    elf = parse_elf(data)
    if elf["function"]["name"] != "static_fp4_gemm":
        raise ValueError("unexpected target kernel symbol")
    text = elf["text"]
    offsets = []
    for relative in range(0, text["size"], 16):
        absolute = text["offset"] + relative
        instruction = data[absolute:absolute + 16]
        if _clear_candidate_bits(instruction) == EXPECTED_INSTRUCTION:
            if instruction != EXPECTED_INSTRUCTION:
                raise ValueError("baseline target OMMA already has candidate bits set")
            offsets.append(absolute)
    if len(offsets) != EXPECTED_OMMA_COUNT:
        raise ValueError(
            f"expected {EXPECTED_OMMA_COUNT} target OMMA slot, found {len(offsets)}"
        )
    return {"elf": elf, "instruction_file_offsets": offsets}


def validate_variant(original: bytes, patched: bytes, variant: str,
                     offsets: list[int]) -> dict:
    if len(offsets) != EXPECTED_OMMA_COUNT:
        raise ValueError("partial OMMA target list rejected")
    if len(original) != len(patched):
        raise ValueError("CUBIN size changed")
    expected = bytearray(original)
    target = variant_instruction(variant)
    for offset in offsets:
        expected[offset:offset + 16] = target
    if bytes(expected) != patched:
        unexpected = [i for i, pair in enumerate(zip(expected, patched))
                      if pair[0] != pair[1]]
        raise ValueError(f"binary differs outside complete OMMA patch: {unexpected[:8]}")

    byte_diffs = [i for i, pair in enumerate(zip(original, patched))
                  if pair[0] != pair[1]]
    file_bits = []
    for byte_offset in byte_diffs:
        xor = original[byte_offset] ^ patched[byte_offset]
        file_bits.extend(byte_offset * 8 + bit for bit in range(8)
                         if xor & (1 << bit))
    allowed = {offset * 8 + bit for offset in offsets for bit in CANDIDATE_BITS}
    if not set(file_bits).issubset(allowed):
        raise ValueError("non-candidate bit changed")

    for offset in offsets:
        if patched[offset:offset + 16] != target:
            raise ValueError("only part of the target OMMA slots was patched")
    return {"byte_offsets": byte_diffs, "file_bit_offsets": file_bits}


def patch_cubin(source: Path, destination: Path, variant: str) -> dict:
    original = source.read_bytes()
    analysis = analyze_baseline(original)
    offsets = analysis["instruction_file_offsets"]
    patched = bytearray(original)
    for offset in offsets:
        patched[offset:offset + 16] = variant_instruction(variant)
    patched_bytes = bytes(patched)
    diff = validate_variant(original, patched_bytes, variant, offsets)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
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
    return {
        "variant": variant,
        "variant_spelling": "bit79_bit78",
        "baseline_sha256": _sha256(original),
        "patched_sha256": _sha256(patched_bytes),
        "kernel": analysis["elf"]["function"]["name"],
        "elf_endianness": "little",
        "text_section": text["name"],
        "text_file_offset": text["offset"],
        "text_size": text["size"],
        "omma_count": len(offsets),
        "instruction_file_offsets": offsets,
        "instruction_offsets_in_text": [offset - text["offset"] for offset in offsets],
        "instruction_before_hex": EXPECTED_INSTRUCTION.hex(),
        "instruction_after_hex": variant_instruction(variant).hex(),
        "instruction_bits_set": sorted(_variant_bits(variant)),
        "whole_cubin_byte_diff": diff["byte_offsets"],
        "whole_cubin_file_bit_diff": diff["file_bit_offsets"],
        "size_unchanged": len(original) == len(patched_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variant", required=True, choices=("00", "01", "10", "11"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = patch_cubin(args.source, args.destination, args.variant)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
