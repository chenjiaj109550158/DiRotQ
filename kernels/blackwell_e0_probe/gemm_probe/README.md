# Static-format SM120 FP4 GEMM correctness probe

This is a correctness-only, pre-quantized input probe.  One warp owns one
`16x8` output tile and executes one static public `m16n8k64` OMMA instruction
inside a runtime K loop.  There is no split-K, multi-warp tile, TMA, async
copy, pipeline, fused quantization, dynamic format selector, benchmark, or
performance claim.

External canonical storage is independent of lane fragments:

- `A_payload[Mp,Kp/2]` uint8: row-major A rows, K-contiguous, earlier nibble low.
- `B_payload[Np,Kp/2]` uint8: column-major B columns, K-contiguous, earlier nibble low.
- `A_scales[Mp,Kp/16]` uint8 UE4M3.
- `B_scales[Np,Kp/16]` uint8 UE4M3.
- `alpha_A`, `alpha_B`: FP32 scalar.
- Output: row-major FP32 `[M,N]` only.

`Mp=ceil(M/16)*16`, `Np=ceil(N/8)*8`, and `Kp=ceil(K/64)*64`.
Payload padding is positive-zero nibble and scale padding is UE4M3 one.
Zero-size dimensions are rejected.

Variants 01/10/11 patch undocumented SASS operand-format bits.  That path is
unsupported by NVIDIA and is neither portable nor a production API.
