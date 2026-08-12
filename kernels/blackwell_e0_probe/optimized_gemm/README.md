# Optimized Blackwell static FP4 GEMM prototype

This directory is a correctness-first, kernel-core feasibility prototype for a
single static operand-format pairing per launch.  It uses the official CUTLASS
SM120 NVFP4 GEMM path for the public E2M1 x E2M1 baseline, then applies the two
previously isolated operand-format bits to every target OMMA slot to study
E0M3 x E2M1 and E0M3 x E0M3.  Patched E0 binaries are undocumented,
unsupported SASS research artifacts.  They are not a production integration.

The pinned upstream is NVIDIA CUTLASS `v4.0.0`, commit
`b995f933179c22d3fe0d871c3a53d11e4681950f`.  The selected configuration is:

- SM120a `OpClassBlockScaledTensorOp`;
- `m16n8k64` NVFP4 OMMA atoms, E2M1 inputs, UE4M3 scales, vector size 16;
- FP32 accumulator and FP32 row-major output;
- 128 x 128 x 128 threadblock tile, 1 x 1 x 1 cluster, four resolved stages;
- TMA warp-specialized cooperative schedule, 384 threads, 92,160 dynamic
  shared-memory bytes for this build;
- no split-K, fused quantizer, selector, bias, projection, or model epilogue.

## External and native layouts

The receiver-facing input remains the verified canonical contract:

- A payload: `uint8 [Mp,Kp/2]`, row-major logical `[M,K]`;
- B payload: `uint8 [Np,Kp/2]`, each storage row is one logical B column;
- A scales: `uint8 [Mp,Kp/16]`;
- B scales: `uint8 [Np,Kp/16]`;
- the earlier K element occupies the low nibble;
- payload K padding is positive zero and scale padding is UE4M3 one (`0x38`).

CUTLASS's A row-major and B column-major subbyte payload strides are identical
to these canonical byte arrays.  No payload repack or swizzle is required, and
the benchmark reports the required A payload transform as zero.  Both scale
tensors use CUTLASS `Sm1xxBlockScaledConfig<16>`'s interleaved K-major atom:

```text
shape  = ((32,4),(16,4))
stride = ((16,4),(0,1))
```

`optimized_fp4_runner` performs canonical-to-native scale conversion on the
GPU.  Dynamic A scale conversion is measured per forward.  B scale conversion
is a one-time offline prepack and is excluded from GEMM-only/per-forward
latency.  Native scale storage can contain alignment padding; its exact bytes
and overhead are reported.  A future activation quantizer can eliminate the A
scale conversion only by emitting this native scale layout directly while
preserving the same UE4M3 values and block-to-K mapping.

Global scales are multiplied in FP32 on the host and supplied once as the
CUTLASS epilogue alpha.  They are not applied inside each K step.  FP32 output
is retained for the packed correctness gate.  The required FP32-to-BF16 output
cast is a separate GPU kernel and a separately reported cost.

## Reproducible build

CUTLASS remains under the ignored build tree; it is not vendored here.

```bash
cmake -S kernels/blackwell_e0_probe/optimized_gemm \
  -B kernels/blackwell_e0_probe/build/optimized_gemm/cmake-build \
  -GNinja \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCUTLASS_ROOT=kernels/blackwell_e0_probe/build/cutlass-v4.0.0
cmake --build kernels/blackwell_e0_probe/build/optimized_gemm/cmake-build -j4
```

The CMake build emits `baseline_00.cubin` and `optimized_fp4_runner`.  Copy or
reference those files from the ignored build directory, then create variants
independently from the same baseline:

```bash
python -m kernels.blackwell_e0_probe.optimized_gemm.patch_optimized_gemm \
  kernels/blackwell_e0_probe/build/optimized_gemm/baseline_00.cubin \
  kernels/blackwell_e0_probe/build/optimized_gemm/variant_01.cubin \
  --variant 01 --manifest kernels/blackwell_e0_probe/build/optimized_gemm/variant_01.patch.json
python -m kernels.blackwell_e0_probe.optimized_gemm.patch_optimized_gemm \
  kernels/blackwell_e0_probe/build/optimized_gemm/baseline_00.cubin \
  kernels/blackwell_e0_probe/build/optimized_gemm/variant_11.cubin \
  --variant 11 --manifest kernels/blackwell_e0_probe/build/optimized_gemm/variant_11.patch.json
```

The patcher accepts exactly one baseline SHA/toolchain/kernel/text-size/slot
table.  It requires 64 aligned target slots, modifies bit 78 for variant 01 and
bits 78 and 79 for variant 11 in all 64 slots, and rejects unknown baselines,
partial slot lists, size changes, and any non-allowlisted bit difference.
Variant routing is `00 = E2xE2`, `01 = E0xE2`, `11 = E0xE0`.

## Correctness and benchmark entry points

```bash
python -m kernels.blackwell_e0_probe.optimized_gemm.run_correctness \
  --real-e0xe2 /absolute/path/sana_real_e0xe2_v1 \
  --real-e0xe0 /absolute/path/sana_real_e0xe0_v1

python -m kernels.blackwell_e0_probe.optimized_gemm.benchmark \
  --warmup 100 --iterations 500 --rounds 5

pytest -q kernels/blackwell_e0_probe/optimized_gemm/test_optimized_gemm.py
```

Real packages always pass through the strict CUDA-free receiver verifier before
module load.  Hardware FP32 is compared to packed FP32 with the existing
gamma-K bound; BF16 output is compared separately to the Ada fake-quant runtime
reference as drift information.  Every correctness run checks output canaries
and CUDA errors.  `benchmark.py` uses CUDA events, preserves raw per-round
latencies, records GPU telemetry/processes, and reports GEMM, dynamic A scale
layout, static B scale prepack, and BF16 output cast separately.

Effective and padded TFLOP/s in the ignored report describe this kernel-core
experiment only.  They do not establish a DiRotQ layer or end-to-end model
speedup.  The prototype does not install or execute SANA/PixArt, optimize the
activation quantizer, fuse the high branch/projection, implement dynamic
TileMix, or explore alternative persistent/TMA kernel families.
