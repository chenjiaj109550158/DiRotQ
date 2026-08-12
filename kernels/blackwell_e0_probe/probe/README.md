# Public SM120 E2M1 MMA probe

This probe launches exactly one warp and issues exactly one public
`m16n8k64` E2M1 x E2M1 block-scaled MMA with UE4M3 scales and FP32
accumulators. It consumes the frozen logical handoff packing directly. It
does not contain an E0 path, a full GEMM, quantization, an epilogue, a binary
patch, or a benchmark.

Build and run from the repository root inside `blackwell-e0-probe`:

```bash
nvcc -std=c++17 -O2 -shared -Xcompiler -fPIC -arch=sm_120a \
  kernels/blackwell_e0_probe/probe/e2m1_mma_probe.cu \
  -o kernels/blackwell_e0_probe/build/libe2m1_mma_probe.so
python -m kernels.blackwell_e0_probe.probe.run_probe
pytest -q kernels/blackwell_e0_probe/probe/test_e2m1_mma_probe.py
```
