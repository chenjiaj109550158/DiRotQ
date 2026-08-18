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
  W4A4 with explicitly configured RTN in its original basis in every arm;
  this is a frozen protocol choice and is reported separately from GPTQ
  fallback (which must remain zero);
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

## Pilot32 results

All six arms produced exactly the same first 32 MJHQ IDs.  Every output is a
decodable, non-flat RGB 1024x1024 PNG.  Values below are paired to the BF16
reference; deltas and win rates are against the per-source PCA arm.  Confidence
intervals are 5000-resample prompt bootstrap intervals.

| scheme | PSNR | delta [95% CI] | PSNR wins | LPIPS | delta [95% CI] | LPIPS wins |
|---|---:|---:|---:|---:|---:|---:|
| per-source PCA | 23.1607 | 0 | - | 0.19758 | 0 | - |
| shared-width | 23.1752 | +0.0145 [-0.5516,+0.6495] | 40.6% | 0.19995 | +0.00237 [-0.01659,+0.02093] | 40.6% |
| shared-operator | 23.3573 | +0.1966 [-0.9538,+1.4593] | 43.8% | 0.19428 | -0.00330 [-0.04720,+0.03434] | 46.9% |
| shared-operator-stage4 | 23.0559 | -0.1048 [-0.9803,+0.7560] | 46.9% | 0.18664 | -0.01094 [-0.03761,+0.01368] | 46.9% |
| representative-operator | 23.1689 | +0.0082 [-0.9385,+0.8336] | 46.9% | 0.19189 | -0.00569 [-0.03463,+0.02548] | 40.6% |

| scheme | SSIM | delta [95% CI] | CLIP | delta [95% CI] |
|---|---:|---:|---:|---:|
| per-source PCA | 0.83057 | 0 | 25.8126 | 0 |
| shared-width | 0.82735 | -0.00321 [-0.01527,+0.00773] | 26.0751 | +0.2625 [-0.0798,+0.6290] |
| shared-operator | 0.82770 | -0.00286 [-0.02626,+0.01966] | 26.5287 | +0.7161 [+0.1394,+1.4271] |
| shared-operator-stage4 | 0.83754 | +0.00698 [-0.01072,+0.02517] | 26.1367 | +0.3241 [+0.0177,+0.6513] |
| representative-operator | 0.83465 | +0.00409 [-0.00557,+0.02573] | 26.1929 | +0.3803 [-0.1756,+1.0691] |

The frozen screen admits `shared-operator` and `representative-operator` for a
larger experiment.  `shared-width` misses the LPIPS tolerance by 0.00037;
`shared-operator-stage4` misses the PSNR tolerance by 0.00481 dB.  These are
Pilot32 screens, not publication-level equivalence claims: all PSNR/LPIPS
intervals are wide and include zero.  Of the passing arms, `shared-operator`
is the preferred result because it also has the best mean PSNR/LPIPS and is a
pooled, reproducible estimator; the representative arm retains only 29.8% of
the baseline protected energy on average and is correspondingly less robust.

## Measured and deployable-memory views

The generation process used reconstructed BF16 fake-quant weights and CPU
offload.  Its sampled peak is useful for validating that sharing actually
removes online basis storage, but is not the packed-kernel deployment memory.

| scheme | BF16 basis GiB | corrected packed model GiB* | sampled peak GPU MiB | full 32 wall time** |
|---|---:|---:|---:|---:|
| BF16 reference | - | - | 24,360 | 24:59.93 |
| per-source PCA | 4.0078 | 14.7295 | 29,458 | 38:13.10 |
| shared-width | 0.0176 | 10.7393 | 25,372 | 33:24.96 |
| shared-operator | 0.1582 | 10.8799 | 25,516 | 33:14.98 |
| shared-operator-stage4 | 0.6328 | 11.3545 | 26,002 | 33:36.14 |
| representative-operator | 0.1582 | 10.8799 | 25,516 | 45:48.39 |

\* Corrected packed model is the prior measured 11,512,351,232-byte INT4
weight payload plus the exact active dense BF16 bases.  It is an analytical
estimate of a corrected model, not a remeasurement of the incorrect literal
speed script.  The experiment `.pt` caches are about 23.8 GB each because they
materialize reconstructed BF16 weights and must not be reported as packed
deployment storage.

\** Sum of the 4-image smoke and the continuation that generated the remaining
28 images; each part includes a fresh model load.  Python fake-quant wall time
is not a production-latency result.

## Mechanism diagnostic

| scheme | protected-energy mean | relative to per-source | minimum relative |
|---|---:|---:|---:|
| per-source PCA | 0.81306 | 100.0% | 100.0% |
| shared-width | 0.42375 | 51.2% | 14.7% |
| shared-operator | 0.58397 | 71.1% | 34.2% |
| shared-operator-stage4 | 0.70145 | 85.7% | 55.6% |
| representative-operator | 0.23620 | 29.8% | 10.5% |

Energy capture is useful for explaining the bases but is not a reliable proxy
for 32-image quality: stage sharing has the strongest shared-basis energy yet
slightly misses the PSNR screen, whereas the representative arm has the lowest
energy and still lands inside the mean screen with high variance.

## Artifact provenance

- experiment root: `/tmp/dirotq_flux_shared_pca.iQPi5A/formal`;
- model revision: `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`;
- dataset SHA-256: `07ce5ef172dc0454c0267ad7a68a16e21ae2e695356651a51a9f303a166b120e`;
- calibration: 128 prompts x 25 steps, 3200 call files, manifest SHA-256
  `a80d5444c27705a90801db55c8fb0de259286de7e662d3207622df0cf23c02a1`;
- random rotation SHA-256:
  `f8e4fb221b59e8a54a89135e488910d2d75838d234efdf2518afdcfcddd48da6`;
- per-source PCA SHA-256:
  `f5a81020e8ad74de14de35310be97d68ae7bf524abe083d37a63123dfc7cc9e5`;
- shared basis SHA-256 values: width
  `35901742541ff7fe87c82dc9b854558bea6fe365325ce5b66391dea17a82d339`,
  operator
  `6d818475c0f3f0e91e384f2c7b4685125f75915177b122015333ca83b4bf1936`,
  stage4
  `25e52146be5da4752c9c83d9c85bcc368f392d3f5dc1534615196b7d0dc71f4e`,
  representative
  `20a468a59d6bb55e7c3b55d1f8c6ba4e6a72e48a2b5a80e31a694d0fd0c1a9f5`;
- all five quantized arms: 380 GPTQ layers, 76 explicitly configured
  no-rotation FFN-down RTN layers, zero RTN fallback;
- quantized-cache SHA-256 values: per-source
  `933ce66e0b7409cc162f79caa50edc0c28d9323aa7aa357574f365e6d735e4c0`,
  width
  `7179b6761246335cb42f9c5f99d42718ac27c2b4d98f6f366577c1b0c88a1191`,
  operator
  `4566dee3428fd0af254fa9333c1f416d0789a44a3afae9b937ced12c58021855`,
  stage4
  `c9e5cb6dc9d1fc7a3b05f1bfb9009457c17ac637150fa772405edd7f06c9dbd5`,
  representative
  `75b691886ebbcbb707187d410699e648176f5d9612edaaed2c10db8021616a3f`;
- metrics and fixed visual grid:
  `/tmp/dirotq_flux_shared_pca.iQPi5A/formal/metrics`.

The only generation warnings were identical long-prompt CLIP/T5 truncation
warnings in matched arms.  There were no NaN, Inf, OOM, CUDA errors or silent
quantization fallbacks in the completed formal runs.
