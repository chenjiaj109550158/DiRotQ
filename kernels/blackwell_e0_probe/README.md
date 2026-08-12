# Blackwell E0/E2 FP4 MMA golden handoff

This directory freezes a **numerical and logical-packing contract** for an
RTX 5090 hardware probe.  It contains only CPU/CUDA PyTorch references and
synthetic golden vectors.  It contains no CUDA kernel, SASS, CUBIN patch,
model tensor, cache, or performance claim.

## Hardware interface status

### `PUBLIC_PTX`

The documented warp MMA exposes **E2M1 × E2M1** with UE4M3 scales:

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
    .f32.e2m1.e2m1.f32.ue4m3
```

Equivalently, its relevant tokens are
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X` and
`.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`.  Public PTX and CUTLASS list
the operand pairing as E2M1/E2M1, shape `(16,8,64)`, and FP32 accumulator.

Primary sources:

- NVIDIA PTX ISA, warp-level matrix fragment for m16n8k64:
  https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16864
- NVIDIA CUTLASS Blackwell functionality:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- CUTLASS warp block-scaled MMA API:
  https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html

### `SASS_EVIDENCED_UNSUPPORTED`

Community reverse engineering reports that operand-format bits in
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` can select E0M3 independently for
the left and right operands, which would cover E0×E2, E2×E0, and E0×E0.
That report uses CUBIN post-processing.  It is not an NVIDIA-documented or
supported programming interface and is not implemented here.

- NVIDIA forum discussion:
  https://forums.developer.nvidia.com/t/does-blackwell-support-int4-native/326513

### `UNVERIFIED`

The following remain `HARDWARE_ENCODING_TO_VERIFY` on the RTX 5090:

- the actual E0M3 SASS operand-format bits;
- whether the reported bits independently enable E0×E2, E2×E0, E0×E0;
- the register-fragment/CUBIN encoding needed to exercise those paths;
- correspondence of the logical E0 nibble below to physical SASS payload;
- correctness and stability of any CUBIN patch across toolchains.

Do not describe community SASS evidence as NVIDIA support.

## Frozen numerical contract

```text
C[16,8] = A[16,64] @ B[64,8]
A logical layout: row-major
B logical layout: column-major
accumulator: FP32
K block: 16 elements
A scales: [16,4] = 64 E4M3 bytes
B scales: [8,4]  = 32 E4M3 bytes
```

The codebooks are sign-magnitude:

| Nibble | E2M1 | E0M3 logical probe contract |
|---:|---:|---:|
| `0x0..0x7` | `+{0,.5,1,1.5,2,3,4,6}` | `+{0,1,2,3,4,5,6,7}` |
| `0x8..0xF` | negative of `0x0..0x7` | negative of `0x0..0x7` |

Bit 3 is sign; bits 2:0 are the magnitude index. `0x0` is positive zero and
`0x8` is negative zero.  The E2M1 mapping is the public numerical format.
The E0M3 mapping is a **logical software/golden convention**; its physical
SASS encoding is `HARDWARE_ENCODING_TO_VERIFY`.

Packing:

- Each byte stores the earlier stream element in bits 3:0 and the next in
  bits 7:4.
- A stream index is `m*64+k`.
- B stream index is `n*64+k`; this is column-major for logical `[K,N]`.
- A scale index is `m*4+floor(k/16)`.
- B scale index is `n*4+floor(k/16)`.

Scale bytes use the nonnegative finite subset of literal
`torch.float8_e4m3fn`, whose bytes are the software contract for the UE4M3
scale operand. Examples: `0 -> 0x00`, minimum subnormal
`2^-9 -> 0x01`, minimum normal `2^-6 -> 0x08`, `1 -> 0x38`, `384 -> 0x7c`,
and `448 -> 0x7e`. Negative zero is canonicalized to `0x00`; bytes with bit 7
set and the NaN byte `0x7f` are invalid scale inputs to this contract.

For the raw result, decoded FP4 values are multiplied by their per-16 E4M3
scales and accumulated in FP32.  Tensor-global FP32 scales are outside the
MMA and are applied once afterward:

```text
C_raw  = (D_fp4(A) * S_A) @ (D_fp4(B) * S_B)
C_full = C_raw * alpha_A_fp32 * alpha_B_fp32
```

This is a logical contiguous handoff format.  It deliberately does not claim
to be the hardware warp-fragment/register layout.

## Golden vectors

`golden/` contains one transparent JSON file per pairing:

- `e2m1_x_e2m1.json`
- `e0m3_x_e2m1.json`
- `e2m1_x_e0m3.json`
- `e0m3_x_e0m3.json`

Every pairing contains zero, complete signed codebooks (including both zero
signs), alternating signs, distinct K-block scales, scale extremes,
deterministic random payloads, layout-sensitive payloads, and accumulation
cancellation.  Expected output is produced by the simple sequential scalar
FP32 implementation and cross-checked against vectorized PyTorch.

Regenerate and verify:

```bash
conda run -n dirotq python -m kernels.blackwell_e0_probe.generate_golden
conda run -n dirotq python -m kernels.blackwell_e0_probe.generate_golden --check
conda run -n dirotq python -m kernels.blackwell_e0_probe.verify_golden --device cpu
CUDA_VISIBLE_DEVICES=0 conda run -n dirotq \
  python -m kernels.blackwell_e0_probe.verify_golden --device cuda
```

On the 5090, a future probe should compare its unpacked/raw/full output with
these files.  Real model tiles must not be introduced until this synthetic
contract passes.
