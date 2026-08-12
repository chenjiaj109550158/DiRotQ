# Real DiRotQ FP4 tile handoff receiver

This directory defines package schema v1 and a correctness-only receiver for
pre-quantized real DiRotQ tiles.  It does not load SANA/PixArt, download a
model, quantize tensors, benchmark the kernel, or claim a production API.
Synthetic package generation is labeled `__synthetic_contract_only__`; its
outputs must never be represented as real model data.

## Package v1

```text
package/
  manifest.json
  cases/CASE_ID/
    a_payload.npy
    a_scales.npy
    b_payload.npy
    b_scales.npy
    expected_packed_fp32.npy
    expected_fakequant_runtime.npy
```

The schema identity is `dirotq.blackwell.real_fp4_tile`, version `1`.
`manifest.json` records the producer commit and optional hostname, model name
and revision, PCA and residual-rotation hashes, rotation mode, quantized-weight
cache hash, quantizer implementation/version, creation timestamp, complete
package byte size, and a hash/size/dtype/shape record for every NPY file.

Each case records the layer, prompt/image ID, scheduler timestep, wrapper call
index, original activation and reconstructed-weight dtypes, pairing, logical
and padded dimensions, formats, layouts, nibble order, block/global scales,
and both expected-output semantics.  Only these pairings are accepted:

- `e0xe2`: E0M3 A and E2M1 B, hardware variant `01`.
- `e0xe0`: E0M3 A and E0M3 B, hardware variant `11`.

## Canonical buffers

For positive logical `M,N,K`:

```text
Mp = ceil(M/16)*16
Np = ceil(N/8)*8
Kp = ceil(K/64)*64
A payload uint8 [Mp,Kp/2]
B payload uint8 [Np,Kp/2]
A scales  uint8 [Mp,Kp/16]
B scales  uint8 [Np,Kp/16]
expected_packed_fp32       float32 [M,N]
expected_fakequant_runtime float32 [M,N]
```

A is logical row-major `[M,K]`.  Each B storage row is one logical B column,
so B is logical column-major `[K,N]`; B must not be serialized row-major.
Within each K stream, the earlier element occupies the low nibble.  Payload
padding is the positive-zero nibble and scale padding is the literal UE4M3 one
byte.  A block scale covers 16 K elements.  Nonzero K padding is forbidden.

`alpha_A` and `alpha_B` are exact, finite, positive FP32 scalars.  A zero
tensor retains a safe positive global scale and uses zero payload or block
scales; `alpha=0` is not accepted.  The packed reference is

```text
FP32(alpha_A * alpha_B) *
sum_k((decode(A) * A_block_scale) * (decode(B) * B_block_scale))
```

with the global product applied once after FP32 K accumulation.

The fake-quant output is deliberately separate.  The producer applies each
global scale to its decoded operand, casts each operand to the declared
`bfloat16` or `float16` runtime dtype, executes `torch.matmul`, casts the
output to the runtime dtype, and stores the exactly materialized values in a
portable float32 NPY.  Its comparison is informational model-runtime drift;
it is not the packed-hardware acceptance condition.

## Strict verification boundary

`verify_package.py` completes before any CUDA initialization or module load.
It canonicalizes the package root; rejects root/content symlinks, path
traversal, non-regular entries, duplicate JSON keys and unlisted data; applies
a 256 MiB default total-size limit; and always loads NPY with
`allow_pickle=False`.  It verifies every hash, file byte size, dtype, shape,
layout declaration, pairing/format, padding value, scale encoding and global
scale.  It then reconstructs the packed FP32 result and checks the supplied
reference using the established K-dependent FP32 `gamma_K` model.

`run_real_tile.py` accepts only byte-identical variants `01` and `11` derived
from the currently pinned static GEMM baseline.  It verifies the whole-CUBIN
SHA, OMMA count, target slot and complete bit diff before CUDA load.  Each
hardware output retains the static GEMM's 64-byte prefix/suffix canaries and
CUDA API error checks.

## Receiver commands on the RTX 5090

```bash
conda run -n blackwell-e0-probe \
  python -m kernels.blackwell_e0_probe.real_tile_handoff.verify_package \
  /absolute/path/to/package --report /tmp/real_tile_verify.json

CUDA_VISIBLE_DEVICES=0 conda run -n blackwell-e0-probe \
  python -m kernels.blackwell_e0_probe.real_tile_handoff.run_real_tile \
  /absolute/path/to/package --report /tmp/real_tile_hardware.json
```

Contract-only synthetic fixtures are generated under an ignored build path:

```bash
conda run -n blackwell-e0-probe \
  python -m kernels.blackwell_e0_probe.real_tile_handoff.create_synthetic_package \
  kernels/blackwell_e0_probe/build/real_tile_handoff/synthetic
```

## Exact Ada producer handoff

The Ada producer must send one ordinary directory with `manifest.json` and,
for every selected real wrapper call, exactly the six NPY files above.  It
must provide:

1. the exact pre-quantized E0 activation payload and UE4M3 scales in canonical
   A layout, plus its positive FP32 global scale;
2. the exact cached E2 or E0 weight payload and UE4M3 scales in canonical B
   column layout, plus its positive FP32 global scale;
3. the valid `[M,N]` packed FP32 reference reconstructed from those same four
   buffers and scales;
4. the valid `[M,N]` original PyTorch fake-quant runtime output, materialized
   to float32 after its declared BF16/FP16 output cast;
5. real model/layer/call provenance and the exact PCA, residual rotation,
   quantized weight cache, quantizer, model revision and producer Git hashes.

The sender should not include models, caches, images, CUBINs, logs, pickle
files, opaque lane fragments, or padded output values.  After transfer, run
the strict verifier on the received directory before invoking the hardware
runner.
