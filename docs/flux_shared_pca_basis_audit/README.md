# FLUX.1-dev shared-PCA basis audit

This experiment applies the four pre-registered sharing rules from the PixArt
audit to the repository's FLUX.1-dev quality path.  It does **not** treat the
literal speed script as a quality arm: that script uses an unseeded random
matrix, does not counter-transform the weight, zeros the protected branch, and
uses RTN.  Here every arm counter-transforms its own weights and uses matched
GPTQ; only the PCA sharing rule changes.

## Frozen quality protocol

- model: `black-forest-labs/FLUX.1-dev`, revision
  `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`;
- dataset: first 32 entries of `datasets/mjhq_5000_samples.json`, deterministic
  image-ID seeds;
- generation: repository FLUX.1-dev settings (25 steps, guidance 3.5,
  1024x1024, max sequence length 512), batch size 1;
- quantization: repository INT W4A4, group size 64, 1/8 protected BF16 tail,
  random residual rotation and GPTQ;
- calibration: repository FLUX.1-dev protocol (128 prompts x 25 steps);
- FFN-down contract: no PCA rotation and no protected tail, matching the
  checked speed scripts' `no-rot ff_down` contract.  FFN-down remains INT4
  W4A4/GPTQ in its original basis in every arm;
- reference: untouched BF16 FLUX.1-dev generated in the same batch shape;
- quality: paired PSNR, LPIPS, SSIM and CLIP, with 5000 prompt bootstrap
  resamples.  FID is not estimated from 32 images.

The per-source PCA arm is the matched quality baseline.  A shared arm is not
claimed to retain quality unless its paired PSNR and LPIPS are both within
`-0.10 dB / +0.002`, their confidence intervals do not support degradation,
and CLIP does not significantly decrease.

## Schemes

1. `shared-width`: one dense basis for every active K=3072 source.  This is the
   closest correctness-preserving analogue of the speed script's single
   K=3072 transform.
2. `shared-operator`: nine active bases: the six non-down double-block source
   families and three non-down single-block source families.  All depths share
   within one semantic family.
3. `shared-operator-stage4`: the nine active families split into four fixed depth
   stages.  Double blocks use `[0,5), [5,10), [10,15), [15,19)` and single
   blocks use `[0,10), [10,19), [19,29), [29,38)`.
4. `representative-operator`: the same nine families as scheme 2, but every
   family reuses the frozen middle source (double block 9 or single block 19)
   instead of pooling calibration covariances.

Pooled bases are equal-source means reconstructed from each official PCA
eigensystem, followed by the repository's 1% isotropic damping and ascending
eigendecomposition.  The high-energy tail and random residual rotation remain
unchanged.  Each Linear still transforms and quantizes its own weight.

## Memory scopes

The online PCA matrices are dense BF16 at generation time.  Their exact
payload (before allocator overhead) is:

| scheme | unique bases | online basis bytes | reduction vs per-source |
|---|---:|---:|---:|
| per-source PCA | 228 | 4,303,355,904 | 1.00x |
| shared-width | 1 | 18,874,368 | 228.00x |
| shared-operator | 9 | 169,869,312 | 25.33x |
| shared-operator-stage4 | 36 | 679,477,248 | 6.33x |
| representative-operator | 9 | 169,869,312 | 25.33x |

These numbers are reported separately from (a) the full fake-quant process
peak, in which reconstructed BF16 weights dominate, and (b) the speed script's
packed INT4 payload.  Adding basis bytes to the latter is only an analytical
corrected-model estimate; it is not a remeasurement of the currently
incorrect speed script.

The full 304-source path was attempted before freezing this contract and
failed correctly on RTX 6000 Ada: one forward queued 76 K=12288 partial
covariances (576 MiB each), exhausted 48 GiB, and the old collector swallowed
the OOM.  The collector now propagates forward errors and rejects zero-count
sources.  Continuing with full dense FFN-down PCA would neither match the
speed claim nor fit the target hardware, so it is not used as a hidden
quality-only exception.

Results and exact artifact hashes are added only after all smoke and formal
arms complete.
