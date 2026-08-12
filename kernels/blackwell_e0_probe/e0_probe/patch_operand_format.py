#!/usr/bin/env python3
"""Strictly patch the two candidate SM120 OMMA operand-format bits.

This module recognizes exactly one canonical CUBIN produced by the public
E2M1 probe.  It is deliberately not a general-purpose CUBIN patcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile


BASELINE_SHA256 = "a1280f145c02ed2118900a86020347e0dc80293346e187a21cd7c887bdedcf74"
EXPECTED_INSTRUCTION = bytes.fromhex("7f740808060e3072ff3e040000de0f00")
CANDIDATE_BITS = (78, 79)
INSTRUCTION_SIZE = 16
ELF_MACHINE_CUDA = 190
SHF_EXECINSTR = 0x4
SHT_PROGBITS = 1
SHT_SYMTAB = 2
STT_FUNC = 2


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cstring(table: bytes, offset: int) -> str:
    end = table.find(b"\0", offset)
    if end < 0:
        raise ValueError("unterminated ELF string")
    return table[offset:end].decode("utf-8")


def parse_elf(data: bytes) -> dict:
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ValueError("input is not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("only ELF64 little-endian CUBINs are accepted")
    e_machine = struct.unpack_from("<H", data, 18)[0]
    if e_machine != ELF_MACHINE_CUDA:
        raise ValueError(f"unexpected ELF machine {e_machine}")
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 58)
    if e_shentsize != 64 or not e_shnum or e_shstrndx >= e_shnum:
        raise ValueError("invalid ELF section table")
    if e_shoff + e_shentsize * e_shnum > len(data):
        raise ValueError("truncated ELF section table")

    sections = []
    for index in range(e_shnum):
        values = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + 64 * index)
        sections.append({
            "index": index,
            "name_offset": values[0],
            "type": values[1],
            "flags": values[2],
            "offset": values[4],
            "size": values[5],
            "link": values[6],
            "info": values[7],
            "align": values[8],
            "entsize": values[9],
        })
    shstr = sections[e_shstrndx]
    shstr_data = data[shstr["offset"]:shstr["offset"] + shstr["size"]]
    for section in sections:
        section["name"] = _cstring(shstr_data, section["name_offset"])
        if section["type"] != 8 and section["offset"] + section["size"] > len(data):
            raise ValueError(f"truncated section {section['name']}")

    text_sections = [
        section for section in sections
        if section["type"] == SHT_PROGBITS
        and section["flags"] & SHF_EXECINSTR
        and section["name"].startswith(".text.")
    ]
    if len(text_sections) != 1:
        raise ValueError(f"expected one executable kernel .text section, found {len(text_sections)}")
    text = text_sections[0]
    if text["offset"] % INSTRUCTION_SIZE or text["size"] % INSTRUCTION_SIZE:
        raise ValueError("kernel .text is not aligned to 128-bit instructions")

    functions = []
    for section in sections:
        if section["type"] != SHT_SYMTAB:
            continue
        if section["entsize"] != 24 or section["link"] >= len(sections):
            raise ValueError("invalid ELF symbol table")
        string_section = sections[section["link"]]
        strings = data[string_section["offset"]:string_section["offset"] + string_section["size"]]
        symbols = data[section["offset"]:section["offset"] + section["size"]]
        for offset in range(0, len(symbols), 24):
            st_name, st_info, _st_other, st_shndx, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", symbols, offset
            )
            if (st_info & 0x0F) == STT_FUNC and st_shndx == text["index"]:
                functions.append({
                    "name": _cstring(strings, st_name),
                    "value": st_value,
                    "size": st_size,
                })
    if len(functions) != 1:
        raise ValueError(f"expected one function in kernel .text, found {len(functions)}")
    if functions[0]["value"] != 0 or functions[0]["size"] != text["size"]:
        raise ValueError("kernel function does not span the unique .text section")
    return {"sections": sections, "text": text, "function": functions[0]}


def _clear_candidate_bits(instruction: bytes) -> bytes:
    result = bytearray(instruction)
    for bit in CANDIDATE_BITS:
        result[bit // 8] &= ~(1 << (bit % 8))
    return bytes(result)


def analyze_baseline(data: bytes) -> dict:
    digest = sha256(data)
    if digest != BASELINE_SHA256:
        raise ValueError(f"unknown baseline SHA-256: {digest}")
    elf = parse_elf(data)
    text = elf["text"]
    matches = []
    for relative in range(0, text["size"], INSTRUCTION_SIZE):
        absolute = text["offset"] + relative
        instruction = data[absolute:absolute + INSTRUCTION_SIZE]
        if _clear_candidate_bits(instruction) == EXPECTED_INSTRUCTION:
            matches.append((relative, absolute, instruction))
    if len(matches) != 1:
        raise ValueError(f"expected one confirmed OMMA instruction, found {len(matches)}")
    relative, absolute, instruction = matches[0]
    if instruction != EXPECTED_INSTRUCTION:
        raise ValueError("canonical baseline already has candidate bits set")
    return {
        "elf": elf,
        "instruction_offset_in_section": relative,
        "instruction_file_offset": absolute,
        "instruction": instruction,
    }


def variant_mask(variant: str) -> int:
    if variant not in {"00", "01", "10", "11"}:
        raise ValueError("variant must be one of 00, 01, 10, 11")
    # Conventional two-bit spelling is bit79 then bit78.
    return ((int(variant[1]) << 78) | (int(variant[0]) << 79))


def expected_variant_instruction(variant: str) -> bytes:
    instruction = bytearray(EXPECTED_INSTRUCTION)
    mask = variant_mask(variant)
    for bit in CANDIDATE_BITS:
        if mask & (1 << bit):
            instruction[bit // 8] |= 1 << (bit % 8)
    return bytes(instruction)


def validate_patched_binary(original: bytes, patched: bytes, instruction_offset: int,
                            variant: str) -> dict:
    if len(original) != len(patched):
        raise ValueError("CUBIN size changed")
    expected_instruction = expected_variant_instruction(variant)
    expected = bytearray(original)
    expected[instruction_offset:instruction_offset + INSTRUCTION_SIZE] = expected_instruction
    byte_diffs = [index for index, pair in enumerate(zip(original, patched)) if pair[0] != pair[1]]
    if bytes(expected) != patched:
        unexpected = [index for index, pair in enumerate(zip(expected, patched)) if pair[0] != pair[1]]
        raise ValueError(f"binary differs outside the exact candidate-bit patch: {unexpected[:8]}")
    bit_diffs = []
    for byte_index in byte_diffs:
        xor = original[byte_index] ^ patched[byte_index]
        for bit_in_byte in range(8):
            if xor & (1 << bit_in_byte):
                bit_diffs.append(byte_index * 8 + bit_in_byte)
    allowed_file_bits = {instruction_offset * 8 + bit for bit in CANDIDATE_BITS}
    if not set(bit_diffs).issubset(allowed_file_bits):
        raise ValueError("a non-candidate instruction bit changed")
    return {"byte_offsets": byte_diffs, "file_bit_offsets": bit_diffs}


def patch_cubin(source: Path, destination: Path, variant: str) -> dict:
    original = source.read_bytes()
    analysis = analyze_baseline(original)
    absolute = analysis["instruction_file_offset"]
    patched = bytearray(original)
    patched[absolute:absolute + INSTRUCTION_SIZE] = expected_variant_instruction(variant)
    patched_bytes = bytes(patched)
    diff = validate_patched_binary(original, patched_bytes, absolute, variant)

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
        "baseline_sha256": sha256(original),
        "patched_sha256": sha256(patched_bytes),
        "elf_endianness": "little",
        "kernel_symbol": analysis["elf"]["function"]["name"],
        "text_section": text["name"],
        "text_section_file_offset": text["offset"],
        "text_section_size": text["size"],
        "instruction_offset_in_section": analysis["instruction_offset_in_section"],
        "instruction_file_offset": absolute,
        "instruction_before_hex": analysis["instruction"].hex(),
        "instruction_after_hex": expected_variant_instruction(variant).hex(),
        "instruction_bits_set": [
            bit for bit in CANDIDATE_BITS if variant_mask(variant) & (1 << bit)
        ],
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
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
