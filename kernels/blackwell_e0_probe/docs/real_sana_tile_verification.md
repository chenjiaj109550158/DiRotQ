# Real SANA FP4 tile verification

This document records correctness evidence only. It contains no model data,
payload, NPY file, CUBIN, disassembly, or performance claim.

## Revisions and provenance

- Receiver repository commit:
  `5dedc04376a8e68ae0f1b9103900e4c78db5478e`
- Static GEMM commit:
  `a41501af44877dcca8b9ef6a77bea8ce6986662c`
- Ada producer commit:
  `a5bf7dd28e21f75510ee0ba0f20b666a86b7cbe3`
- SANA model revision:
  `e2b3c0cbffebcd09d83805e88b9f5f106afc74ac`
- Archive SHA-256:
  `ecd669a3b0c8dc08c15ae5c2cc30d9fdf5999a5baf9b9b04efe93433acf43f9e`
- E0xE2 package tree SHA-256:
  `c6a42891ba36d19eb484d877acfe74d9788e3cbd5a3fd00d68983e53254f9c72`
- E0xE2 manifest SHA-256:
  `2275f8dfab30096905823fc6aca7c153d24c08a3ad760bbe0fb59a618c9f4fac`
- E0xE0 package tree SHA-256:
  `e8527ae5f99cb97fee103f8b99594d8080343c8fe38a0c01416dba09ca9e8431`
- E0xE0 manifest SHA-256:
  `1dbfbdde5a0cb3c568427622a3901da53ea7b5c13ccb3ff32086e5930ed1771e`

The package-tree hashes use the producer's pinned algorithm: sort regular
files by package-relative POSIX path, then hash each path length as an
eight-byte little-endian integer, the path bytes, and the file contents.

## Captured cases

The E0xE2 and E0xE0 packages use byte-identical A payload and A scale files,
identical activation global scales, prompt ID
`000438f99177213e07a9b9c875248eea17b8c8c6`, and K padding to 1984.

| Case | Layer | Timestep | MxNxK |
|---|---|---:|---:|
| `attn_input_early_aligned` | `transformer_blocks.0.attn1.to_q` | 999 | 16x8x1952 |
| `attn_output_early_tail` | `transformer_blocks.0.attn1.to_out.0` | 999 | 17x9x1960 |
| `attn_input_mid_tail` | `transformer_blocks.10.attn1.to_q` | 749 | 17x9x1952 |
| `attn_output_mid_aligned` | `transformer_blocks.10.attn1.to_out.0` | 749 | 16x8x1960 |

Both pairings passed strict CPU package verification before CUDA was touched.
On the RTX 5090, all four E0xE2 cases and all four E0xE0 cases matched their
packed FP32 references bitwise. Every gamma-K tolerance mismatch count was
zero, every bitwise mismatch count was zero, and every output canary remained
intact. The BF16 fake-quant runtime comparison was recorded separately as
informational numerical drift and was not used to relax packed correctness.

## Weight encoding transcode

The Ada sidecars store sorted signed-codebook indices 0 through 14. The
producer converts them to receiver sign-magnitude nibbles as follows:

```text
source index:    0 1 2 3 4 5 6 7 8 9 a b c d e
receiver nibble: f e d c b a 9 0 1 2 3 4 5 6 7
```

Reference decoding proved this is a bijective, codepoint-preserving encoding
change for both E2M1 and E0M3. It does not requantize the weight and does not
change block scales, the FP32 global scale, or reconstructed logical values.
Negative zero is canonicalized to positive zero, so nibble `8` is not emitted
by this 15-codepoint transcode.

## Binary routing

- E0xE2 uses static GEMM variant 01, SHA-256
  `ec4cc412692855b181cf96e39415ff9f6c089c19951e3acc212eb7439d86f0ae`.
- E0xE0 uses static GEMM variant 11, SHA-256
  `144c0aa4e7722c417ab624313ccce7066c17cc54a019dca0e48bcbff5d1d0d7a`.
- Public E2xE2 baseline SHA-256:
  `a4d0d3e1be4fb47365e8659fb99025e61b5b471ca08adbc707906157df0e91e3`.

Variant 01 differs only at the allowlisted operand-format bit 78; variant 11
differs only at bits 78 and 79. Each binary contains one target OMMA slot.
Wrong-pairing routing was rejected by the CUBIN SHA/pairing allowlist before
CUDA initialization.

## Support boundary

The E2M1xE2M1 PTX/CUTLASS path is public. The E0 operand-format selections are
undocumented SASS behavior verified experimentally on this RTX 5090 and CUDA
12.8.1 toolchain. They are unsupported by NVIDIA, are pinned to exact binary
hashes and instruction slots, and are not a portable or production API. This
evidence establishes tile/GEMM correctness only; it is not an end-to-end model
or performance result.

## Reproduction

After verifying and safely extracting the received archive, run:

```bash
conda run -n blackwell-e0-probe \
  python -m kernels.blackwell_e0_probe.real_tile_handoff.verify_package \
  /absolute/path/to/sana_real_e0xe2_v1

conda run -n blackwell-e0-probe \
  python -m kernels.blackwell_e0_probe.real_tile_handoff.verify_package \
  /absolute/path/to/sana_real_e0xe0_v1

CUDA_VISIBLE_DEVICES=0 conda run -n blackwell-e0-probe \
  python -m kernels.blackwell_e0_probe.real_tile_handoff.run_real_tile \
  /absolute/path/to/sana_real_e0xe2_v1 --report /tmp/sana_real_e0xe2_hardware.json

CUDA_VISIBLE_DEVICES=0 conda run -n blackwell-e0-probe \
  python -m kernels.blackwell_e0_probe.real_tile_handoff.run_real_tile \
  /absolute/path/to/sana_real_e0xe0_v1 --report /tmp/sana_real_e0xe0_hardware.json
```
