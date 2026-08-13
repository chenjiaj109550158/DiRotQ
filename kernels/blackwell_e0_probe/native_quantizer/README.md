# Native Blackwell FP4 activation quantizer

This directory is a correctness-first SM120a prototype. It converts contiguous
row-major BF16 or FP16 `X[M,K]` directly on the GPU to the payload and scale
factor buffers consumed by `optimized_gemm`. It does not run a model,
projection, high branch, format selector, or activation-quantizer fusion.

## Numerical contract

The first kernel reduces the finite full-tensor absolute maximum. A one-thread
finalizer writes the FP32 device scalars

```
alpha_A = 1                         when max(abs(X)) == 0
alpha_A = max(abs(X)) / 2688        otherwise
alpha_product = alpha_A * alpha_B
```

The quantize/pack kernel reads `alpha_A`, assigns one thread to each row/K16
block, converts `block_amax / 6` (E2M1) or `block_amax / 7` (E0M3) to UE4M3,
and writes both packed payload and native CUTLASS scale storage. CUDA 12.8
`__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)` provides
round-to-nearest-even E4M3 conversion. FP4 nearest-code ties retain the lower
magnitude index. Nibble bit 3 is sign, earlier K is the low nibble, and any
quantized zero is positive zero.

Zero blocks use UE4M3 one (`0x38`). M/K payload padding is zero and scale
padding is `0x38`. NaN/Inf, non-contiguous input, unsupported dtype/format, and
wrong static pairing are rejected rather than silently converted.

## Native layout

Payload is already the canonical/CUTLASS-native row-major A layout
`uint8[Mp,Kp/2]`. Scale factors are written directly to the
`Sm1xxBlockScaledConfig<16>` SFA layout. For each `(row, k_block)`:

```
atom   = floor(row/128) * (Kp/64) + floor(k_block/4)
within = (row mod 32)*16 + floor((row mod 128)/32)*4 + (k_block mod 4)
offset = atom*512 + within
```

The allocation covers complete 128-row by 64-K atoms. No canonical scale
tensor or scale-layout transform exists in the measured native path. The old
GPU canonical-to-native transform is retained only as a separately measured
comparison.

## BF16 epilogue and unsupported variants

`optimized_fp4_gemm_bf16_kernel` retains the pinned public CUTLASS E2M1×E2M1
mainloop, FP32 accumulator, 128×128×128 tile, cooperative TMA schedule, and a
BF16 D epilogue. The public baseline has 64 OMMA slots. Variants 01 and 11 are
generated independently from it by modifying only the already verified
operand-format bits 78 and 78/79 in all slots. Unknown hashes, partial slot
sets, and other binary changes are rejected.

E0M3 selection remains an undocumented and unsupported SASS research result.
The debug FP32 kernel remains the packed-mathematics authority; every BF16
epilogue variant is required to match a BF16 cast of its debug FP32 output
bit-for-bit.

## Build and run

```bash
cmake -S kernels/blackwell_e0_probe/native_quantizer \
  -B kernels/blackwell_e0_probe/build/native_quantizer/cmake-build -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCUTLASS_ROOT="$PWD/kernels/blackwell_e0_probe/build/cutlass-v4.0.0"
cmake --build kernels/blackwell_e0_probe/build/native_quantizer/cmake-build -j4

python -m kernels.blackwell_e0_probe.native_quantizer.run_correctness
python -m kernels.blackwell_e0_probe.native_quantizer.benchmark_pipeline \
  --warmup 100 --iterations 500 --rounds 5
pytest -q kernels/blackwell_e0_probe/native_quantizer/test_native_quantizer.py
```

Generated inputs, CUBINs, binaries, reports, sanitizer logs, and disassembly
belong under ignored `kernels/blackwell_e0_probe/build/native_quantizer/`.
Latency is a low-branch kernel-pipeline feasibility measurement, not a DiRotQ
layer or end-to-end model speedup.
