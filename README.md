# DiRotQ: Rotation-Aware Quantization for 4-bit Diffusion Transformers

Official implementation for the NeurIPS 2026 submission.

## Abstract

Diffusion Transformers (DiTs) achieve state-of-the-art image generation quality but
incur substantial memory and computational costs at inference. While aggressive
post-training quantization (PTQ) to 4-bit precision offers significant efficiency
gains, it typically results in severe quality degradation. Existing approaches —
including smoothing, mixed-precision, rotation, and low-rank residual methods —
partially mitigate this issue but still leave a noticeable gap to FP16/BF16
performance.

We introduce **DiRotQ**, a W4A4 PTQ framework that mitigates this degradation
through rotation-aware activation quantization. DiRotQ identifies a low-rank
subspace capturing dominant activation variance via principal component analysis
(PCA), preserving coefficients in this subspace at higher precision while
quantizing the remaining components to 4-bit. Activations are rotated into the
PCA basis at inference time using calibration-derived orthogonal transformations,
while the inverse rotation is fused into layer weights offline. Combined with
GPTQ-based weight quantization, DiRotQ achieves an FID of 15.9 and PSNR of
19.1 dB on PixArt-Σ over the MJHQ dataset, outperforming the prior
state-of-the-art SVDQuant (FID 18.9, PSNR 17.6) under the same INT W4A4
setting. Further, our Triton-based custom kernel reduces memory usage of the 12B
FLUX.1-dev model by 2.1× and achieves a 2.3× speedup over the 4-bit
weight-only baseline (W4A16) on a 24 GB RTX 4090 GPU.

## Repository layout

```
apply_dirotq.py               # main entry point: PCA rotation + RTN/GPTQ quantization + image generation
get_basis.py                  # collect calibration activations and compute PCA eigendecomposition
gen_rotation.py               # generate the random rotation matrices R1, R2, R_down
dirotq_fused_unrotation.py    # fp32 fused-unrotation forward (debug/fallback)
dirotq_fused_unrotation_fast.py  # fast bf16/fp16 fused-unrotation forward (default)
diagnose_layer_qsnr.py        # per-layer QSNR diagnostic
utils/
    quant_utils.py            # ActQuantWrapper, RTN/NVFP4 weight quant, PCA permutation helpers
    gptq_utils.py             # GPTQ Hessian collection + weight quantization
    hadamard_utils.py         # Hadamard transform utilities
models/
    pixart-sigma/             # config.yaml, model_utils.py, basis_utils.py, calib_prompts.yaml
    flux-dev/                 # ...
    flux-schnell/             # ...
    sana-1.6b/                # ...
datasets/
    mjhq_5000_samples.json    # 5K MJHQ prompts (with deterministic per-prompt seeds)
    sdci_5000_samples.json    # 5K sDCI prompts
metrics/                      # FID / LPIPS / PSNR / CLIP / Image Reward scripts
speedup/                      # FLUX.1-dev end-to-end speedup measurement (Triton + nunchaku)
```

## Environment setup

Tested with Python 3.12 and CUDA 12.4. Python 3.12 is required for reproducible
random rotation generation.

```bash
conda create -n dirotq python=3.12 -y
conda activate dirotq

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

You will also need a HuggingFace account with access to the gated diffusion
model checkpoints (e.g., FLUX.1-dev). Set `HUGGING_FACE_HUB_TOKEN` in your
environment, or run `huggingface-cli login` once.

## Quick start: DiRotQ INT W4A4 on PixArt-Σ + MJHQ

The command below quantizes PixArt-Σ to INT W4A4 using DiRotQ (PCA rotation +
GPTQ weight quantization) and generates 5000 images for the MJHQ prompt set.
The `--skip-quant-layers ff.net.2` flag keeps the FFN down-projection
activations in 16-bit, consistent with our reported numbers (which match the
"down-projection layers in PixArt-Σ kept in higher precision (W4A16)" note in
the paper).

```bash
python apply_dirotq.py \
    --model pixart-sigma \
    --dataset datasets/mjhq_5000_samples.json \
    --gptq \
    --gptq-batch-size 4 \
    --skip-quant-layers ff.net.2 \
    --output-dir models/pixart-sigma/generated_images_gptq_a4w4_mjhq \
    --batch-size 32
```

What this does on first run:

1. Generates the random rotation matrices `R1`, `R2`, `R_down` if missing
   (`gen_rotation.py`).
2. Collects calibration activations from 128 COCO-Caption prompts and
   computes the per-layer PCA basis if missing (`get_basis.py`). Activation
   caches are stored under `models/pixart-sigma/calibration_dataset/caches/`.
3. Wraps every linear layer with an `ActQuantWrapper`, assigns the online
   PCA rotation `evec @ R1` (or per-head / down-proj variants) to each layer,
   and configures the mixed-precision activation quantizer.
4. Collects per-layer GPTQ Hessians from the calibration data and runs
   GPTQ with damping 0.01 and block size 128.
5. Saves the quantized weights to
   `models/pixart-sigma/quantized_cache/int4_g64_gptq_skip<hash>_model.pt`.
6. Generates images with the fused fp16 unrotation kernel and writes them to
   the `--output-dir`.

Subsequent runs with the same flags skip steps 1–5 by loading the cached
basis, rotation, Hessians, and quantized weights.

## Other formats and models

```bash
# NVFP4 W4A4 (group-size 16, FP8 scaling) on PixArt-Σ
python apply_dirotq.py --model pixart-sigma --dataset datasets/mjhq_5000_samples.json \
    --gptq --nvfp4 --gptq-batch-size 4 --skip-quant-layers ff.net.2 --batch-size 32

# INT W4A8 on PixArt-Σ
python apply_dirotq.py --model pixart-sigma --dataset datasets/mjhq_5000_samples.json \
    --gptq --gptq-batch-size 4 --a-bits 8 --skip-quant-layers ff.net.2 --batch-size 32

# FLUX.1-dev INT W4A4
python apply_dirotq.py --model flux-dev --dataset datasets/mjhq_5000_samples.json \
    --gptq --gptq-batch-size 2 --batch-size 4

# SANA-1.6B INT W4A4
python apply_dirotq.py --model sana-1.6b --dataset datasets/mjhq_5000_samples.json \
    --gptq --gptq-batch-size 4 --batch-size 16
```

For the sDCI prompt set, swap the dataset to `datasets/sdci_5000_samples.json`.

## Useful flags

| Flag | Purpose |
| --- | --- |
| `--gptq` | Use GPTQ for weight quantization (default: RTN) |
| `--nvfp4` | Use NVFP4 (FP4 E2M1) instead of INT4 |
| `--a-bits N` | Override activation bits (default from `models/<m>/config.yaml`) |
| `--skip-quant-layers PAT [PAT …]` | Keep activations of matching layers in 16-bit |
| `--pca-only-layers PAT [PAT …]` | Use PCA channel permutation (no rotation matmul) for matching layers |
| `--max-images N` | Generate only the first N images (debug) |
| `--quantized-cache PATH` | Override cache path |
| `--slow-unrotation` | Use the fp32 unrotation path (debug / fallback) |

## Evaluation

After generating images, compute metrics via the scripts under `metrics/`. The
folder is organized per-model and per-dataset:

```
metrics/<model>/<dataset>/<metric>/<run_tag>.txt
```

Each script consumes a directory of generated images and writes a single-line
text result. Example:

```bash
python metrics/pixart-sigma/mjhq/fid/run_fid.py \
    --gen-dir models/pixart-sigma/generated_images_gptq_a4w4_mjhq \
    --out metrics/pixart-sigma/mjhq/fid/dirotq_int4.txt
```

## End-to-end speedup measurement (FLUX.1-dev)

See `speedup/README.md` for the full FLUX.1-dev W4A4 latency + memory
measurement protocol on RTX 4090.

## Notes on reproducibility

Per-prompt seeds are derived deterministically from each prompt id via a small
hash (`apply_dirotq.py:hash_str_to_int`), so a given (model, prompt) pair maps
to a fixed seed across runs. However, because we batch generation,
`set_seed(seeds[0])` depends on which image is first in the batch — different
`--batch-size` values produce slightly different pixel outputs even with the
same per-prompt seeds. The quantized cache itself is bytewise reproducible
across runs.
