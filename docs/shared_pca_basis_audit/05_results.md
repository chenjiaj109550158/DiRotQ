# Matched PixArt-Sigma shared-basis result

## Frozen protocol

- Formal generation source: `9485294`.
- Model: PixArt-Sigma, local revision
  `e102b3591cc82e97071b8b4cb90d834d0c487207`.
- Data: first 128 entries of `mjhq_5000_samples.json`, dataset SHA-256
  `07ce5ef172dc0454c0267ad7a68a16e21ae2e695356651a51a9f303a166b120e`.
- INT W4A4, group 64, GPTQ, random residual rotation,
  `--skip-quant-layers ff.net.2`, batch 4, 20 steps, CFG 4.5.
- All five arms used different basis-keyed transformed-weight caches; each
  completed 224/224 GPTQ layers with zero RTN fallback.
- FP16 reference was the existing matched batch-4 first-128 output.
- Intervals are 5000-sample paired prompt bootstraps. PSNR, LPIPS, SSIM, and
  CLIP are paired to the same FP16 image for every prompt.

## Results

| scheme | active basis bytes | reduction | protected energy / per-layer | PSNR | delta (95% CI) | LPIPS | delta (95% CI) | SSIM | CLIP |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| per-linear PCA | 232,243,200 | 1.0x | 100.0% | 19.5009 | reference | 0.22520 | reference | 0.73418 | 26.6684 |
| shared width | 2,820,096 | 82.35x | 57.74% | 18.6798 | -0.8211 [-1.1731,-0.4662] | 0.26550 | +0.04030 [+0.02678,+0.05388] | 0.69771 | 26.8263 |
| shared operator | 8,294,400 | 28.0x | 69.67% | **18.9514** | -0.5495 [-0.8674,-0.2215] | **0.25342** | +0.02822 [+0.01692,+0.03920] | **0.70952** | 26.8476 |
| operator x stage4 | 33,177,600 | 7.0x | 82.82% | 18.8378 | -0.6631 [-1.0335,-0.3252] | 0.25391 | +0.02870 [+0.01613,+0.04286] | 0.70920 | 26.7850 |
| representative block 14 | 8,294,400 | 28.0x | 49.57% | 18.1143 | -1.3866 [-1.7869,-0.9918] | 0.29448 | +0.06927 [+0.05634,+0.08282] | 0.67319 | 26.9041 |

All 640 generated images were the exact expected IDs, RGB 1024x1024,
decodable, finite, and non-flat. CLIP changes were not significant for any
arm; all PSNR/LPIPS/SSIM regressions were significant.

## Conclusion

Correct shared-PCA implementations are executable and substantially reduce
online basis storage, but none retained the per-linear method's image quality.
The primary interpretation of "same-type Linears share a basis" (shared
operator) was the best arm, yet lost 0.55 dB PSNR and increased LPIPS by 0.028.

The literal speed scripts cannot serve as the missing quality experiment: in
addition to using one unseeded random transform rather than PCA, they rotate
activation without counter-rotating the packed weight, zero the protected
low-rank tensors, and use RTN rather than the quality GPTQ path. The reported
memory/latency method and the reported image-quality method are therefore not
the same numerical model.

No shared scheme passed the pre-registered 128-image gate, so an official 5K
FID run was not performed. The result supports the narrower claim that the
repository's sharing assumption does **not** preserve its per-linear-PCA
quality under a corrected apples-to-apples implementation. It does not
produce a statistically valid estimate of 5K FID.
