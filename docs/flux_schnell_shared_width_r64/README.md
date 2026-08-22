# FLUX.1-schnell shared-width r64 reproduction

This checkpoint contains the shared-PCA implementations and the matched
FLUX.1-schnell evaluation path used for the 32-image comparison.  Large model,
basis, rotation, cache, and image artifacts are intentionally not in Git.

## Frozen experiment

- Model: `black-forest-labs/FLUX.1-schnell`
- Revision: `741f7c3ce8b383c54771c7003378a50191e9efe9`
- Dataset: first 32 entries of `datasets/mjhq_5000_samples.json`
- Dataset SHA-256:
  `07ce5ef172dc0454c0267ad7a68a16e21ae2e695356651a51a9f303a166b120e`
- Seed: rolling base-31 hash of the image ID modulo `1_000_000_007`
- Generation: 4 steps, guidance scale 0, batch size 1, 1024 x 1024
- DiRotQ basis: shared-width PCA, high rank 64, random residual rotation seed 42
- Main INT4 path: group-64 GPTQ for active DiRotQ layers
- Explicit RTN layers: `.net.2` and `proj_out.linears.1`
- Adaptive-normalization Linear layers: persistent INT4 weight / BF16 activation
  (W4A16), group 64
- The shared-width PCA excludes the down-projection family.  The down
  projections follow the explicit `.net.2` RTN protocol and remain W4A4.

The five shared-basis schemes (`per-layer`, `shared-width`,
`shared-operator`, `shared-operator-stage4`, and `representative-operator`),
the packed-INT4 runtime, and the W4A16 exception path are already part of this
branch's history.  This checkpoint adds the Schnell-specific routing and
portable evaluation commands.

## End-to-end quick start

### 1. Clone the source

The implementation checkpoint is `f79a89314f99b3ff0eb87cce6e96ffee7e80ecdd`
on `exp/flux-r64-shared-pca-fused-kernels`:

```bash
git clone --branch exp/flux-r64-shared-pca-fused-kernels --single-branch \
  https://github.com/chenjiaj109550158/DiRotQ.git
cd DiRotQ
git merge-base --is-ancestor \
  f79a89314f99b3ff0eb87cce6e96ffee7e80ecdd HEAD
```

An exit status of zero confirms that the checkout contains the frozen
implementation.  Later documentation-only commits on this branch do not
change the quantization artifacts.  To create the tested environment from
scratch:

```bash
conda create -n dirotq python=3.12 -y
conda activate dirotq
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Download the exact base model

FLUX.1-schnell is gated on Hugging Face.  Accept its access terms first, then
authenticate without putting a token in a command or log.  The custom DiRotQ
repository may be public, but authentication is still required for the gated
base model.

```bash
hf auth login

MODEL_DIR=/path/with/enough/space/FLUX.1-schnell-741f7c3
hf download black-forest-labs/FLUX.1-schnell \
  --revision 741f7c3ce8b383c54771c7003378a50191e9efe9 \
  --local-dir "$MODEL_DIR"
```

Do not substitute `main`: the packed-weight provenance was validated against
the exact revision above.

### 3. Download the custom DiRotQ artifacts

The uploaded model repository and immutable artifact revision are:

```text
chenjiaj109550158/dirotq-flux-schnell-shared-width-r64
fbfe7133e6a2def4831ba9defd0553bbaadf2bc1
```

At the time this README was updated, the repository was private.  The owner
must change its visibility to public before unrelated users can download it;
while private, only the owner and explicitly authorized users have access.

```bash
ARTIFACT_ROOT=/path/with/enough/space/dirotq-flux-schnell-shared-width-r64
ARTIFACT_REVISION=fbfe7133e6a2def4831ba9defd0553bbaadf2bc1

hf download chenjiaj109550158/dirotq-flux-schnell-shared-width-r64 \
  --revision "$ARTIFACT_REVISION" \
  --local-dir "$ARTIFACT_ROOT"
```

Verify every downloaded byte before loading the PyTorch caches:

```bash
(cd "$ARTIFACT_ROOT" && sha256sum -c ARTIFACTS.sha256)
```

All nine entries must print `OK`.  This download contains four runtime
artifacts plus their provenance manifests:

1. `U-flux-schnell-shared-width.pt` (PCA basis)
2. `R-flux-schnell-shared-width-r64.pt` (residual rotation)
3. `int4_g64_gptq_model.fp32-scales.packed-int4.pt` (packed active weights)
4. `flux-schnell-adaptive-norm-int4-g64-bf16.pt` (packed W4A16 weights)

It deliberately excludes the 23.8 GB dense reconstructed cache, the 14.3 GB
Hessian, and calibration activations.  The packed inference path validates
their immutable producer hashes and does not need to download them.

### 4. Verify the dataset and generate the matched 32 images

```bash
REPO=$PWD
OUT=$REPO/models/flux-schnell/reproduction/real_int4_fused/images
GPU=0

echo \
  "07ce5ef172dc0454c0267ad7a68a16e21ae2e695356651a51a9f303a166b120e  datasets/mjhq_5000_samples.json" \
  | sha256sum -c -

env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES="$GPU" \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run --no-capture-output -n dirotq python apply_dirotq.py \
    --model flux-schnell \
    --model-id "$MODEL_DIR" \
    --dataset datasets/mjhq_5000_samples.json \
    --basis-path "$ARTIFACT_ROOT/U-flux-schnell-shared-width.pt" \
    --rotation-path "$ARTIFACT_ROOT/R-flux-schnell-shared-width-r64.pt" \
    --gptq --gptq-calib-files 512 --gptq-batch-size 4 \
    --gptq-rtn-layers .net.2 proj_out.linears.1 \
    --quantized-cache "$ARTIFACT_ROOT/int4_g64_gptq_model.pt" \
    --real-int4 \
    --real-int4-cache "$ARTIFACT_ROOT/int4_g64_gptq_model.fp32-scales.packed-int4.pt" \
    --real-int4-fake-cache-sha256 85875969ca86126e409771efc7315d6947e047c9fa944b7f2b942065cbff73fc \
    --real-int4-hessian-sha256 9018d14f44a4595a0336e1972741a8215fa46909f21cfe1a83257a155513c729 \
    --real-int4-kernel-mode fused \
    --real-w4a16-modulators \
    --real-w4a16-cache "$ARTIFACT_ROOT/flux-schnell-adaptive-norm-int4-g64-bf16.pt" \
    --generate --batch-size 1 --max-images 32 --output-dir "$OUT"
```

This produces the first 32 MJHQ images using deterministic image-ID seeds, 4
steps, guidance scale 0, batch size 1, and 1024 x 1024 output.  Successful
startup must print that it loaded the exact packed INT4 sidecar; it must not
collect Hessians or rebuild GPTQ.

## Optional: official SVDQuant weights

The official Nunchaku/SVDQuant checkpoints are independently available from
the migrated `nunchaku-tech/nunchaku-flux.1-schnell` repository:

```bash
NUNCHAKU_DIR=/path/to/weights/nunchaku-flux.1-schnell
hf download nunchaku-tech/nunchaku-flux.1-schnell \
  svdq-int4_r32-flux.1-schnell.safetensors \
  --local-dir "$NUNCHAKU_DIR"

# Blackwell-only native NVFP4 checkpoint; do not use its kernel on Ada.
hf download nunchaku-tech/nunchaku-flux.1-schnell \
  svdq-fp4_r32-flux.1-schnell.safetensors \
  --local-dir "$NUNCHAKU_DIR"
```

The old `mit-han-lab/nunchaku-flux.1-schnell` repository contains the same
named files but its model card says it migrated to `nunchaku-tech`.

## Packed-real command details

The path assigned to `--quantized-cache` below is a logical provenance path;
it is allowed to be absent for inference because the two producer hashes are
explicit.  The packed sidecar is read directly and is not reconstructed from a
BF16 cache.

```bash
REPO=/path/to/DiRotQ
MODEL_DIR=/path/with/enough/space/FLUX.1-schnell-741f7c3
ARTIFACT_ROOT=/path/with/enough/space/dirotq-flux-schnell-shared-width-r64
OUT=$REPO/models/flux-schnell/reproduction/real_int4_fused/images

cd "$REPO"
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run --no-capture-output -n dirotq python apply_dirotq.py \
    --model flux-schnell \
    --model-id "$MODEL_DIR" \
    --dataset datasets/mjhq_5000_samples.json \
    --basis-path "$ARTIFACT_ROOT/U-flux-schnell-shared-width.pt" \
    --rotation-path "$ARTIFACT_ROOT/R-flux-schnell-shared-width-r64.pt" \
    --gptq --gptq-calib-files 512 --gptq-batch-size 4 \
    --gptq-rtn-layers .net.2 proj_out.linears.1 \
    --quantized-cache "$ARTIFACT_ROOT/int4_g64_gptq_model.pt" \
    --real-int4 \
    --real-int4-cache "$ARTIFACT_ROOT/int4_g64_gptq_model.fp32-scales.packed-int4.pt" \
    --real-int4-fake-cache-sha256 85875969ca86126e409771efc7315d6947e047c9fa944b7f2b942065cbff73fc \
    --real-int4-hessian-sha256 9018d14f44a4595a0336e1972741a8215fa46909f21cfe1a83257a155513c729 \
    --real-int4-kernel-mode fused \
    --real-w4a16-modulators \
    --real-w4a16-cache "$ARTIFACT_ROOT/flux-schnell-adaptive-norm-int4-g64-bf16.pt" \
    --generate --batch-size 1 --max-images 32 --output-dir "$OUT"
```

This is a persistent packed integer implementation with a fused Triton
activation-packing/output path.  It is not a claim of native packed INT4
tensor-core execution.

For the matched dense fake-quant run, retain the calibration directory and
dense GPTQ cache and use:

```bash
env NVIDIA_TF32_OVERRIDE=0 \
  conda run --no-capture-output -n dirotq python \
  metrics/run_dirotq_flux_schnell_int4_matched32.py \
    --model-snapshot "$MODEL_DIR" \
    --run-root /path/to/full/reproduction-root \
    --cache-root /path/to/dense-cache-root \
    --candidate-gpus 4 --max-images 32
```

## Reproduce official SVDQuant INT4

Old Nunchaku 0.1.x loaders require a lossless two-file view of the current
monolithic official checkpoint.  This split does not decode or requantize it:

```bash
conda run -n deepcompressor python metrics/prepare_nunchaku_legacy_checkpoint.py \
  --input "$NUNCHAKU_DIR/svdq-int4_r32-flux.1-schnell.safetensors" \
  --output-dir /path/to/nunchaku-int4-r32-legacy-view

env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 \
  conda run --no-capture-output -n deepcompressor python \
  metrics/run_nunchaku_flux_schnell_matched32.py \
    --model-snapshot "$MODEL_DIR" \
    --checkpoint-dir /path/to/nunchaku-int4-r32-legacy-view \
    --source-checkpoint "$NUNCHAKU_DIR/svdq-int4_r32-flux.1-schnell.safetensors" \
    --dataset datasets/mjhq_5000_samples.json \
    --output-dir models/flux-schnell/reproduction/svdquant-int4-r32 \
    --max-images 32
```

## Evaluation

Cross-framework pixel metrics must use each framework's own BF16 reference;
otherwise framework-level BF16 drift is incorrectly attributed to the
quantizer:

```bash
env CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n dirotq python \
  metrics/evaluate_flux_schnell_svdquant_dirotq.py \
    --dataset datasets/mjhq_5000_samples.json \
    --svdquant-reference /path/to/svdquant/bf16-reference \
    --svdquant-images /path/to/svdquant/int4-images \
    --dirotq-reference /path/to/dirotq/bf16-reference \
    --dirotq-images /path/to/dirotq/images \
    --output-dir /path/to/metrics
```

Packed-real versus dense fake-quant parity can be recomputed after placing the
three DiRotQ image directories under a common run root:

```bash
env CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n dirotq python \
  metrics/evaluate_flux_schnell_real_int4_parity.py \
    --run-root /path/to/run-root
```

## Recorded matched32 results

These numbers are degradation metrics against the corresponding BF16
framework reference.  Higher PSNR/SSIM/CLIP and lower LPIPS are better.

| Configuration | PSNR | LPIPS | SSIM | CLIP |
|---|---:|---:|---:|---:|
| DiRotQ shared-width r64 NVFP4 | 19.44969 | 0.235342 | 0.757688 | 26.39627 |
| DiRotQ shared-width r64 fake INT4 | 18.87220 | 0.253799 | 0.739722 | 26.37788 |
| DiRotQ shared-width r64 packed-real INT4 | 19.02319 | 0.248347 | 0.742840 | 26.35556 |
| Official SVDQuant/Nunchaku INT4 r32 | 18.21215 | 0.296874 | 0.710342 | 26.15678 |

The packed-real and dense fake INT4 images are not bitwise identical.  Across
32 prompts, packed-real minus fake was +0.15099 dB PSNR, -0.005452 LPIPS,
+0.003118 SSIM, and -0.02232 CLIP; their paired confidence intervals crossed
zero.  Treat this as numerical execution-path variation, not proof of a speed
or quality advantage.

The measured packed-real transformer-only B=1 memory on RTX 6000 Ada was:

- Persistent transformer/frames: 6,860,510,464 bytes (6.389 GiB)
- PyTorch peak allocated: 7,604,061,184 bytes (7.083 GiB)
- PyTorch peak reserved: 8,201,961,472 bytes (7.639 GiB)
- `nvidia-smi` process peak: 8,330 MiB
- Mean synthetic transformer forward: 4.322 s

This is transformer-only synthetic-forward memory, not full text encoder/VAE
pipeline memory and not a production latency claim.
