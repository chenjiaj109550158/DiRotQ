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
| `weight_only_nvfp4.py` | `WeightOnlyNVFP4Linear` — real 4-bit storage (E2M1 codes + per-channel global + group-16 e4m3 scales); fused Triton dequant → cuBLAS forward. |
| `results/flux_w4a4_no_rot_ffdown.json` | INT4 output (RTX 4090, B=1/2/4). |
| `results/flux_nvfp4_no_rot_ffdown.json` | NVFP4 W4A4 output (RTX PRO 6000 Blackwell). |
| `results/flux_a16w4_nvfp4.json` | NVFP4 A16W4 (weight-only) output (Blackwell). |

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
