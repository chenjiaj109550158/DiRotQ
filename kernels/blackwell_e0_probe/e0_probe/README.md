# Unsupported SM120 E0M3 SASS probe

> This experiment patches undocumented SASS operand-format bits.
> It is unsupported by NVIDIA and is not a portable or production API.

The probe starts from the canonical public E2M1 x E2M1 `m16n8k64` CUBIN and
tests candidate instruction bits 78 and 79.  Variant names use the spelling
`bit79_bit78`: `00`, `01`, `10`, and `11`.  This naming does not assume which
bit controls A or B; diagnostic numerical cases identify that mapping.

`patch_operand_format.py` accepts only the pinned canonical SHA-256, parses
the ELF section and symbol tables, requires exactly one kernel `.text`
section and function, and requires exactly one known OMMA instruction.  Every
variant is generated afresh from the read-only baseline.  It rejects any
change outside the two candidate bits and writes output atomically.

All CUBINs, logs, disassembly, manifests, and reports belong under the ignored
`kernels/blackwell_e0_probe/build/e0_probe/` directory.
