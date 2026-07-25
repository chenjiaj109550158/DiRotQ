# DiRotQ — Speedup Measurement

End-to-end latency + memory comparison: fp16 baseline vs DiRotQ W4A4
running on nunchaku's SVDQuant fused kernel (rotation enabled on K=3072
ops, no rotation on K=12288 ff_down). Two quant formats:

- **INT4** (`report_flux_w4a4_no_rot_ffdown.py`) — `mma.m16n8k64.s4.s4.s32`
  int4 tensor cores, **Turing→Ada (sm_75–89)** only. Measured on RTX 4090.
- **NVFP4** (`report_flux_nvfp4_no_rot_ffdown.py`) — FP4 E2M1 + FP8 group-16
  micro-scales, **Blackwell (sm_120)** tensor cores. Measured on RTX PRO
  6000 Blackwell. This is the format to use on Blackwell, where int4 tensor
  cores were removed (the int4 kernel falls back to int8 there).

## Files

| File | What it does |
| --- | --- |
| `report_flux_w4a4_no_rot_ffdown.py` | INT4 measurement. Two-phase orchestrator: subprocess 1 = fp16 with `accelerate.cpu_offload` (fp16 OOMs at B≥2 on 24 GB), subprocess 2 = DiRotQ W4A4 on the SVDQuant nunchaku v2 fused kernel. Prints a combined table and writes JSON. |
| `report_flux_nvfp4_no_rot_ffdown.py` | NVFP4 measurement (Blackwell). Same pipeline with `SVDQW4A4Linear(precision='nvfp4')` + the FP4 packer. fp16 baseline is measured **native** at every batch (96 GB fits — no offload), so the speedup is the clean compute-bound figure. |
| `nunchaku_pack.py` | Self-contained port of deepcompressor's `NunchakuWeightPacker`. `convert_dirotq_low_weight_to_nunchaku` (INT4, group-64, s4 layout) and `convert_dirotq_low_weight_to_nunchaku_fp4` (NVFP4, group-16, E2M1 + FP8-e4m3 micro-scales). |
| `kernels/int8_rotation.py` | Triton int8 `y = x @ U`. Injects DiRotQ rotation before every K=3072 op. Wins on **Ada** — used by the INT4 script. |
| `kernels/bf16_rotation.py` | Plain bf16 cuBLAS `y = x @ U`. Full-precision dense fallback for the NVFP4 path. |
| `kernels/fp8_rotation.py` | Fused FP8 (e4m3) `y = x @ U`: one Triton pass quantizes activations rowwise to e4m3, then `torch._scaled_mm` runs the FP8 GEMM (~1.7× over bf16 cuBLAS on Blackwell). Default for the NVFP4 script. |
| `report_flux_a16w4_nvfp4.py` | **Weight-only** NVFP4 (A16W4): bf16 activations, 4-bit weights, no rotation. Isolates the memory win from the compute win. |
| `report_latency_breakdown.py` | Kernel-level latency breakdown (INT4/RTX 4090): builds the DiRotQ-W4A4 model and profiles it with `torch.profiler`, bucketing every CUDA kernel into rotation / int4 GEMM / quant-dequant / attention / high-precision-bf16 / norm-elementwise. See "Kernel-level latency breakdown" below. |
| `weight_only_nvfp4.py` | `WeightOnlyNVFP4Linear` — real 4-bit storage (E2M1 codes + per-channel global + group-16 e4m3 scales); fused Triton dequant → cuBLAS forward. |
| `results/flux_w4a4_no_rot_ffdown.json` | INT4 output (RTX 4090, B=1/2/4). |
| `results/flux_nvfp4_no_rot_ffdown.json` | NVFP4 W4A4 output (RTX PRO 6000 Blackwell). |
| `results/flux_a16w4_nvfp4.json` | NVFP4 A16W4 (weight-only) output (Blackwell). |
| `results/latency_breakdown.json` | Kernel-level latency breakdown output (RTX 4090, B=1/2/4). |

## NVFP4 results (RTX PRO 6000 Blackwell, 96 GB, M=4608)

With the fused FP8 rotation (`kernels/fp8_rotation.py`):

| B | fp16 native | NVFP4 | speedup | peak VRAM (fp16 → NVFP4) |
|---|-------------|-------|---------|--------------------------|
| 1 | 391 ms      | 192 ms| 2.04×   | 24.3 → 12.1 GB |
| 2 | 797 ms      | 370 ms| 2.15×   | 24.8 → 12.4 GB |
| 4 | 1663 ms     | 745 ms| 2.23×   | 25.8 → 13.1 GB |

Weights 2.01× smaller. The speedup is **compute-bound** (~2.0–2.2×): unlike
the 24 GB INT4/4090 run, fp16 never offloads on this 96 GB card, so there is
no PCIe-driven batch-2/4 spike — this is the apples-to-apples number.

### Rotation kernel evolution (B=1 / B=2 / B=4 speedup)

| Rotation | per-K=3072 cost | NVFP4 speedup |
|----------|-----------------|---------------|
| int8-Triton (Ada-tuned) | ~0.64 ms | 1.53 / 1.66 / 1.71× |
| bf16 cuBLAS | ~0.31 ms | 1.76 / 1.95 / 2.00× |
| **fused FP8** | **~0.18 ms** | **2.04 / 2.15 / 2.23×** |

The rotation is a dense GEMM injected before every K=3072 op and was ~25% of
the forward; profiling showed *it*, not the FP4 GEMM (~0.13 ms), was the
Blackwell bottleneck. On sm_120 the int8-Triton kernel runs well below
hardware throughput; bf16 cuBLAS fixed that, and FP8 tensor cores (~1.7×
bf16) squeezed it further. The FP8 rotation computes `x @ U` in e4m3
(rel err ~0.04), which is fine because its output is FP4-quantized by the next
kernel anyway. Use `kernels/bf16_rotation.py` if you need the rotation at full
bf16 precision. fp16 on Blackwell is itself fast (391 ms vs the 4090's 624 ms),
which is why even at 2.2× the absolute FP4 win looks modest.

## A16W4 vs W4A4 (weight-only vs full 4-bit), Blackwell, M=4608

`report_flux_a16w4_nvfp4.py` — weights NVFP4 (4-bit), activations bf16, no rotation:

| config | B=1 | B=2 | B=4 | peak VRAM (B=1) |
|--------|-----|-----|-----|------------------|
| fp16 (native) | 392 ms | 795 ms | 1659 ms | 24.3 GB |
| **A16W4** (weight-only) | 408 ms (0.96×) | 822 ms (0.97×) | 1695 ms (0.98×) | **12.0 GB (2.03× smaller)** |
| **W4A4** (full, FP8 rot) | 192 ms (2.04×) | 370 ms (2.15×) | 745 ms (2.23×) | 12.1 GB (2.0× smaller) |

Takeaway: **weight-only NVFP4 buys memory (~2×), not speed.** On a
compute-bound workload (flux, M=4608) the matmul stays bf16 — there is no
FP4-tensor-core compute win from quantizing weights alone — so A16W4 lands at
~1.0× (no speedup, no slowdown). The compute win requires quantizing
activations too: that is what W4A4 does (2.0–2.2×). Both give ~2× memory.

The forward dequantizes the 4-bit weight to bf16 with a **fused Triton kernel**
(planar packing → coalesced stores, ~1.1 TB/s) then runs cuBLAS; the dequant
adds only ~15 ms total, so A16W4 ≈ fp16. (An eager-torch dequant — LUT gather +
`repeat_interleave` — was ~2× slower; the kernel removes that.)

### A16W4 memory breakdown (peak VRAM = weights + activations)

| B | fp16 W / A / tot | A16W4 W / A / tot | ratio |
|---|------------------|-------------------|-------|
| 1 | 23.80 / 0.51 / 24.31 GB | 11.44 / 0.51 / 11.95 GB | 2.03× |
| 2 | 23.80 / 0.99 / 24.79 GB | 11.44 / 0.99 / 12.42 GB | 2.00× |
| 4 | 23.80 / 1.96 / 25.76 GB | 11.44 / 1.96 / 13.39 GB | 1.92× |

The win is **entirely in weights** (23.8 → 11.4 GB resident, 2.08×): quantized
linears are 4.85 GB of codes + e4m3 micro-scales + bf16 per-channel global
(norms/embeddings stay fp16). **Activations are identical to fp16** (A16 keeps
them 16-bit), so as batch grows the unchanged activations dilute the total
ratio (2.03× → 1.92×). This is the key contrast with W4A4, which also shrinks
activations (~1.6×) by quantizing them — and is what unlocks the compute win.

> NVFP4 packing here is plain group-16 RTN (`amax/6` → e4m3, identity
> per-channel/global scales), enough to drive the real FP4 kernel — latency
> and memory are exact. Fully-calibrated NVFP4 *accuracy* is a separate
> concern, measured by `apply_dirotq.py --nvfp4`.

## Kernel-level latency breakdown (RTX 4090, INT4, DiRotQ-W4A4 no-rot ff_down)

The 273/543/1084 ms end-to-end numbers above are a single wall-clock numbers;
they don't say how much of that time is DiRotQ's own rotation vs. the int4
GEMM vs. the parts of the forward that stay high-precision. `report_latency_
breakdown.py` answers that: it builds the *exact* same model as
`report_flux_w4a4_no_rot_ffdown.py` and profiles one forward pass per batch
with `torch.profiler`, then buckets every individual CUDA kernel launch (read
from the Chrome-trace export, not `key_averages()` — see the note in the
script; naively summing `key_averages()` double-counts any CPU op that wraps
exactly one kernel, e.g. flash-attention, by ~1.5×) into:

| Bucket | What it is |
|---|---|
| `rotation` | DiRotQ's online `x @ U` — the fused Triton int8 kernel (`kernels/int8_rotation.py`), injected before every K=3072 op. |
| `quant_dequant` | nunchaku's `quantize_w4a4_fuse_lora_kernel` — the activation int4-quantize pass launched just before each `gemm_w4a4` kernel (a separate launch, not fused into the GEMM). |
| `int4_gemm` | The `gemm_w4a4` tensor-core kernel itself. RMSNorm+RoPE and GELU are compiled as epilogues into this same kernel launch (nunchaku's fusion), so they aren't separable from it. |
| `attention_bf16` | Flash-attention (`scaled_dot_product_attention`). Q/K/V/O projections are int4; the softmax attention itself runs bf16 — unquantized. |
| `high_precision_bf16` | Everything DiRotQ leaves at full precision: modulator Linears (`norm1.linear` / `norm.linear`), `x_embedder`, `context_embedder`, `proj_out`. |
| `norm_elementwise` | LayerNorm/RMSNorm, residual adds, AdaLN gate multiplies — real GPU time, doesn't cleanly belong to either GEMM path. |
| `other` | Memset/misc. |

Run: `python -u speedup/report_latency_breakdown.py` (add `--dump-raw` to see
the raw per-kernel names before bucketing). Profiled GPU-busy totals
(274/549/1094 ms) land within ~1% of the measured end-to-end numbers
(277/551/1095 ms on this run — see the note above that run-to-run noise on
this rig is <1%), so the buckets below account for essentially the whole
forward pass, not just a sampled fraction of it:

| Bucket | B=1 | B=2 | B=4 |
|---|---|---|---|
| attention (bf16, unquantized) | 96.8 ms (35.3%) | 192.6 ms (35.1%) | 371.8 ms (34.0%) |
| int4 GEMM (tensor core) | 87.1 ms (31.7%) | 161.5 ms (29.4%) | 321.5 ms (29.4%) |
| **rotation (DiRotQ, online)** | **52.4 ms (19.1%)** | **91.8 ms (16.7%)** | **182.7 ms (16.7%)** |
| norm / elementwise | 22.2 ms (8.1%) | 76.8 ms (14.0%) | 160.3 ms (14.7%) |
| high-precision bf16 branch | 7.7 ms (2.8%) | 9.4 ms (1.7%) | 9.9 ms (0.9%) |
| quant/dequant (activation) | 6.1 ms (2.2%) | 12.5 ms (2.3%) | 29.3 ms (2.7%) |
| other | 2.1 ms (0.8%) | 3.8 ms (0.7%) | 18.0 ms (1.6%) |
| **profiled total** | **274.4 ms** | **548.6 ms** | **1093.5 ms** |

Takeaways:

- **Rotation is ~17-19% of the forward, not the bottleneck, and its share
  shrinks (slightly) as batch grows** — it's a dense GEMM (`M × 3072 × 3072`)
  so it scales linearly with M like every other GEMM here, while
  `norm_elementwise` (residual adds, gate multiplies) grows *faster* than
  linear as a share because per-token elementwise ops don't benefit from the
  bigger M the way tensor-core GEMMs do.
- **The single largest bucket is unquantized bf16 attention (~35%), not the
  int4 GEMM (~30%) or the rotation (~17%).** DiRotQ/SVDQuant quantize the
  QKV/O *projections* but not the softmax-attention compute itself — on this
  kernel stack, attention is the actual ceiling on how much further W4A4
  speedup can go, more so than the rotation overhead.
- **Quant/dequant overhead is small (~2-3%) and separable from the GEMM** —
  nunchaku launches it as its own kernel (`quantize_w4a4_fuse_lora_kernel`)
  immediately before `gemm_w4a4`, rather than fusing it into the GEMM's
  epilogue. It does not meaningfully compete with the rotation for the
  "where did the overhead go" question.
- **Whether rotation becomes a bottleneck at other batch sizes/resolutions**:
  since both rotation and the int4 GEMM are dense GEMMs over the same M
  (token count), their ratio is roughly resolution/batch-invariant on this
  kernel stack — going from B=1 to B=4 (4× the tokens) barely moves rotation's
  share (19.1% → 16.7%). The NVFP4/Blackwell numbers below tell the more
  interesting version of this story: there the *rotation kernel choice*
  (int8-Triton vs bf16-cuBLAS vs fused-FP8) swung end-to-end speedup from
  1.53× to 2.23× at fixed batch — i.e. rotation-kernel efficiency, not
  batch/resolution, is what determines whether rotation is a bottleneck.

Raw JSON: `speedup/results/latency_breakdown.json`.

### Memory breakdown, side by side (from `results/flux_w4a4_no_rot_ffdown.json`)

The latency breakdown above answers "where does the time go"; the existing
2-config run already answers "where does the memory go" for this exact
model (fp16 vs. DiRotQ-W4A4 no-rot ff_down) — pulled here from the
already-committed `speedup/results/flux_w4a4_no_rot_ffdown.json`, not
re-measured:

| B | fp16 weights (GB) | fp16 activations (GB) | fp16 peak total (GB) | W4A4 weights (GB) | W4A4 activations (GB) | W4A4 peak total (GB) |
|---|---|---|---|---|---|---|
| 1 | 23.80 | 0.50 | 24.30 | 11.51 | 0.32 | 11.83 |
| 2 | 23.80 | 1.01 | 24.81 | 11.51 | 0.62 | 12.13 |
| 4 | 23.80 | 2.01 | 25.81 | 11.51 | 1.23 | 12.74 |

(fp16 weights and W4A4 weights are constant across batch — only
activations grow with B. fp16 B≥2 rows are the linear extrapolation from
the measured B=1 native peak, since fp16 itself OOMs natively at B≥2 on
24 GB — see the note in `report_flux_w4a4_no_rot_ffdown.py`.)

Reading it together with the latency table: DiRotQ's rotation buys its
2.0–2.3× speedup while *also* cutting weights 2.07× (23.80→11.51 GB) and
activations ~1.6× (e.g. 2.01→1.23 GB at B=4) — the rotation/int4-GEMM/
quantize kernels that cost ~38% of the forward (rotation + quant_dequant)
are the same kernels responsible for the entire memory win, not separate
overhead bolted onto an otherwise-unchanged memory footprint.

## Usage

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...        # gated FLUX.1-dev

# INT4 (Ada / RTX 4090)
python -u speedup/report_flux_w4a4_no_rot_ffdown.py

# NVFP4 (Blackwell / RTX PRO 6000)
python -u speedup/report_flux_nvfp4_no_rot_ffdown.py
```

Each script handles everything — downloads the flux-dev fp16 weights from
HuggingFace, builds rotation matrices, packs each Linear via the
deepcompressor port, wires up `NunchakuFluxTransformerBlock` /
`NunchakuFluxSingleTransformerBlock` from nunchaku's v2 module, monkey-
patches their forward to inject DiRotQ rotation, and measures B=1/2/4
end-to-end. The INT4 fp16 path uses `accelerate.cpu_offload` (24 GB can't
fit fp16 at B≥2); the NVFP4 fp16 path runs native at every batch (96 GB).

To re-run only one phase manually (either script):

```bash
python -u speedup/report_flux_nvfp4_no_rot_ffdown.py --phase=fp16
python -u speedup/report_flux_nvfp4_no_rot_ffdown.py --phase=nunchaku
```

## Requirements

- `nunchaku` with the `transformer_flux_v2` module:
  - INT4: a build for your sm_75–89 GPU + matching torch.
  - NVFP4: a **Blackwell (sm_120)** build. Tested with the prebuilt wheel
    `nunchaku-1.2.1+cu12.8torch2.11-cp312` from the nunchaku releases, on
    **torch 2.11.0+cu128** (the `dirotq` conda env). See `INSTALL.md` §3.
- `diffusers`, `transformers`, `accelerate`, `torch`, `triton`
- `torchao` is **optional** — `transformers ≥ 4.57` imports it lazily and
  the scripts stub/skip it (the NVFP4 stack runs with torchao absent).
- HuggingFace token with access to `black-forest-labs/FLUX.1-dev`
- INT4: RTX 4090 / any sm_89, ≥ 24 GB. NVFP4: any sm_120 Blackwell card.
