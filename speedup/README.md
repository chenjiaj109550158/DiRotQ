# DiRotQ — Speedup Measurement

End-to-end latency + memory comparison: fp16 baseline vs DiRotQ W4A4
running on nunchaku's SVDQuant fused kernel (rotation enabled on K=3072
ops, no rotation on K=12288 ff_down).

## Files

| File | What it does |
| --- | --- |
| `report_flux_w4a4_no_rot_ffdown.py` | The full measurement script. Two-phase orchestrator: subprocess 1 = fp16 with `accelerate.cpu_offload` (since fp16 OOMs at B≥2), subprocess 2 = DiRotQ W4A4 on the SVDQuant nunchaku v2 fused kernel. Prints a combined results table and writes JSON. |
| `nunchaku_pack.py` | Self-contained port of deepcompressor's `NunchakuWeightPacker`. Packs an fp16/bf16 weight matrix into the byte/scale layout that `svdq_gemm_w4a4_cuda` expects (s4 MMA fragment layout for `mma.m16n8k64.s4.s4.s32`). |
| `kernels/int8_rotation.py` | Triton kernel for `y = x @ U` with int8 weight, bf16 activation, bf16 output. Used to inject DiRotQ rotation before every K=3072 op without paying full bf16 cuBLAS cost. |
| `results/flux_w4a4_no_rot_ffdown.json` | Latest measurement output (speedup + memory breakdown across B=1, 2, 4 on RTX 4090). |

## Usage

```bash
python -u speedup/report_flux_w4a4_no_rot_ffdown.py
```

The script handles everything — downloads the flux-dev fp16 weights from
HuggingFace, builds rotation matrices, packs each Linear via the
deepcompressor port, wires up `NunchakuFluxTransformerBlock` /
`NunchakuFluxSingleTransformerBlock` from nunchaku's v2 module, monkey-
patches their forward to inject DiRotQ rotation, and measures B=1/2/4
end-to-end. fp16 path uses `accelerate.cpu_offload` (the same hook
diffusers' `enable_sequential_cpu_offload` uses) because the model
doesn't fit on a 24 GB GPU at B≥2.

To re-run only one phase manually:

```bash
python -u speedup/report_flux_w4a4_no_rot_ffdown.py --phase=fp16
python -u speedup/report_flux_w4a4_no_rot_ffdown.py --phase=nunchaku
```

## Requirements

- `nunchaku` (≥ v2 with `transformer_flux_v2` module)
- `diffusers`, `transformers`, `accelerate`, `torch`, `triton`
- `torchao` (the `Float8WeightOnlyConfig` symbols are stubbed inside the
  script if your `torchao` version doesn't define them — needed for
  `transformers ≥ 4.57` to import cleanly)
- HuggingFace token with access to `black-forest-labs/FLUX.1-dev`
- RTX 4090 (or any sm_89 GPU); ≥ 24 GB VRAM for the fp16 native B=1 path
