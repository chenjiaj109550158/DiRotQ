# DiRotQ — Speedup Measurement

Measurement scripts for DiRotQ, modeled on
[nunchaku/app/.../latency.py](../../nunchaku/app/flux.1/t2i/latency.py).

DiRotQ's accuracy pipeline (in the parent directory) does **fake**
quantization — quantized values are dequantized back to fp16/bf16 before the
matmul, so the existing pipeline is *slower* than fp16 baseline. To produce
an apples-to-apples speedup number we need either real low-bit kernels or a
projected analysis. This folder ships both.

## Files

| File | What it does |
| --- | --- |
| `utils.py` | `setup_pipeline(model, precision)`, `run_timed`, `capture_transformer_inputs`, formatting helpers. |
| `kernels/torch_int4.py` | **W4A16** backend using `torch._weight_int4pack_mm`. Repacks DiRotQ's fake-quantized fp weights into int4 with group scales. |
| `kernels/triton_w4a4.py` | **W4A4** Triton kernel. Both activations and weights int4 in the low region; uses int8 mma with int32 accumulation + per-group scales. |
| `mixed_linear.py` | `patch_forward_real(transformer, backend)` — replaces `ActQuantWrapper.forward` with a real-kernel forward (rotation → real int4 GEMM on low region → fp tail GEMM → bias). Per-head and Hadamard layers fall back to fake. |
| `latency.py` | End-to-end / per-step latency, comparing `{fp16, dirotq-fake, dirotq-torch, dirotq-triton}`. Mirrors nunchaku's `latency.py`. |
| `layer_bench.py` | Per-layer microbenchmark across the four paths on the shapes that actually appear in the model (no checkpoint required). |
| `theoretical_speedup.py` | Walks an instrumented transformer and computes projected speedup from FLOP / byte ratios. Useful for paper tables. |

## Prerequisites

- The basis file `models/<model>/basis/U-<model>.pt`
- The rotation file `models/<model>/basis/R-<model>.pt`
- The fused quantized cache `models/<model>/quantized_cache/int4_g64_rtn_model.pt`
   (or the corresponding `gptq` / `nvfp4` variant)

`latency.py` and `theoretical_speedup.py` need all three. `layer_bench.py`
does not — it generates synthetic weights for shape-level timing.

## Usage

End-to-end latency on flux-dev:

```bash
python -m speedup.latency --model flux-dev \
    --precisions fp16 dirotq-fake dirotq-torch dirotq-triton \
    --mode end2end --warmup-times 2 --test-times 5
```

Per-step (transformer-only) latency:

```bash
python -m speedup.latency --model flux-dev --mode step
```

Per-layer microbenchmark (no checkpoint needed):

```bash
python -m speedup.layer_bench --model flux-dev --batch-tokens 4608
```

Theoretical speedup analysis:

```bash
python -m speedup.theoretical_speedup --model flux-dev --batch-tokens 4608
```

All scripts also write a JSON dump of the raw timings to `speedup/results/`.

## Caveats

1. **Triton W4A4 kernel is a reference implementation.** It uses int8 tensor
    cores (after unpacking int4 → int8) for portability — true int4 mma exists
    only on Ampere/Ada (sm_75–sm_89). On Hopper / Blackwell the int4 tensor
    cores were removed, so int8 mma is the right model. The kernel is correct
    but not heavily tuned; expect it to be modestly faster than fp16 on
    memory-bound shapes and roughly on-par on compute-bound shapes. A
    production kernel would add software pipelining, swizzled tiling, and
    fused dequant-into-mma like nunchaku's CUTLASS kernels.

2. **Per-head and Hadamard layers stay on the fake path.** Per-head layers
    interleave `[d_q | hlen]` channels per head, which doesn't match the
    standard W4A4 layout assumed by both real backends. Roughly 80% of
    quantized layers in flux-dev (QKV-img, QKV-txt, ff_up, ff_down, single
    QKV, single proj_mlp, single proj_out_mlp) are eligible. `patch_forward_real`
    prints the breakdown.

3. **End-to-end latency is the right metric, not per-layer.** The per-layer
    benchmark is useful for sanity-checking that the kernels work and for
    debugging slow shapes; the headline number should always come from
    `latency.py --mode end2end` (or `step`).

4. **Cache compatibility.** The quantized cache must match the precision
    config you're benchmarking. Default is `int4_g64_rtn_model.pt`. Add
    `--gptq` or `--nvfp4` to switch caches.
