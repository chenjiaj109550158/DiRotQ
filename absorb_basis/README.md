# DiRotQ-absorb-basis

A DiRotQ variant whose deployed kernel path exactly matches the quality-measurement
path — fixing the flaw where the accuracy pipeline used a per-layer PCA basis but the
speed/memory pipeline (speedup/) used one shared rotation.

## Method

Original DiRotQ (per W4A4 linear, per-layer PCA basis U_l, random rotation R):

```
Y = Q4(X U_l R) @ Q4(R^T U_l^T W^T)
```

DiRotQ-absorb-basis removes the online rotation entirely and absorbs the basis into
the offline weights. Split the basis into the top-r part U_r (r = 32, same rank as
SVDQuant) and the complement:

```
Y = (X U_r)(W U_r)^T          <- 16-bit low-rank branch (lora_down = U_r, lora_up = W U_r)
  + Q4(X) @ Q4(W_res^T)       <- NVFP4 main branch, W_res = W - (W U_r) U_r^T
```

- No online rotation matmul, no smoothing (smooth = 1).
- `Q4(W_res)` is quantized offline with GPTQ on the exact two-level NVFP4 grid the
  nunchaku kernel dequants (per-group-16 e4m3 micro-scale × bf16 top scale;
  per-channel `wcscales` for fused qkv, per-tensor `wtscale` otherwise).
- The GPTQ Hessian and the PCA basis come from the same per-layer input covariance
  H = 2/n Σ XᵀX over the DiRotQ calibration set (COCO 128 prompts × 4 steps).
- Layers with the same input activation share one basis (fused qkv shares one
  lora_down by construction; single-block qkv_proj and mlp_fc1 use the same U).
- down_proj (mlp_fc2 / mlp_context_fc2, K = 12288), the W4A16 int4-g64
  adaptive-norm linears, biases, and all unquantized layers are copied verbatim
  from the official SVDQuant checkpoint (`svdq-fp4_r32-flux.1-schnell`), so those
  layers follow SVDQuant's method exactly.

The result is a drop-in nunchaku-format checkpoint: same tensor set, same shapes,
same kernels as SVDQuant → identical memory footprint and latency by construction;
quality is where the methods differ.

## Pipeline (FLUX.1-schnell)

```bash
# 1. calibration caches (bf16, COCO 128 prompts x 4 steps) — dirotq env
python models/flux-schnell/collect_calibration_dataset.py \
    --prompts models/flux-schnell/calib_prompts.yaml \
    --output models/flux-schnell/calibration_dataset \
    --num-samples 128 --num-steps 4 --guidance-scale 0 --cpu-offload

# 2. input covariances + top-32 PCA bases — dirotq env
python absorb_basis/collect_cov.py

# 3. GPTQ + pack + assemble nunchaku checkpoint — dirotq env
python absorb_basis/build_checkpoint.py

# 4. kernel-level validation — svdquant env (needs nunchaku)
python absorb_basis/validate_kernel.py --ckpt models/flux-schnell/absorb_basis/dirotq-absorb-basis-fp4_r32-flux.1-schnell.safetensors

# 5. MJHQ-32 generation + PSNR/LPIPS/SSIM vs the SVDQuant bf16 reference,
#    plus transformer-only memory / forward latency — svdquant env,
#    same protocol as the SVDQuant baseline run.
```

## Smoothing variants (MJHQ-32 vs bf16 reference, FLUX.1-schnell, nunchaku fp4, RTX 5090)

`--smooth a05` applies classic SmoothQuant (alpha=0.5, per-channel act-amax from
`collect_act_amax.py`) before the PCA; `--smooth svdq` reuses the official
SVDQuant per-layer calibrated smooth factors (unpacked from the checkpoint).
In both cases the PCA basis is computed in the smoothed domain
(cov(X/s) = D^-1 H D^-1), the weight becomes W*s, GPTQ uses the smoothed
Hessian, and the kernel's built-in smooth mechanism is used (stored
lora_down = U/s since the kernel applies the low-rank branch on the raw input).

| Variant                        | PSNR ↑ | LPIPS ↓ | SSIM ↑ |
| ------------------------------ | ------ | ------- | ------ |
| SVDQuant NVFP4 (official)      | 19.22  | 0.2284  | 0.7466 |
| absorb-basis (no smooth)       | 19.12  | 0.2302  | 0.7436 |
| absorb-basis + SmoothQuant a=.5| 19.00  | 0.2302  | 0.7436 |
| absorb-basis + SVDQuant smooth | 18.87  | 0.2331  | 0.7399 |

Smoothing before the PCA does not help this method (PSNR drops 0.1-0.25 dB):
the activation-derived PCA branch already absorbs the dominant activation
outliers, and rescaling channels before the eigendecomposition distorts the
variance structure the basis exploits, while pushing outlier mass into the
weights. Weight-QSNR actually improves slightly with smoothing (18.1 -> 18.3 dB
median), confirming the loss is on the activation/basis side, not the GPTQ side.
