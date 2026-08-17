# Scheme 3: operator family x four fixed depth stages

This scheme tests whether layer depth, rather than individual layer identity,
is the information that a shared basis must preserve.  The block partition is
fixed before evaluation:

```text
stage 0: blocks 0-6
stage 1: blocks 7-13
stage 2: blocks 14-20
stage 3: blocks 21-27
```

Within each stage, the six operator families from Scheme 2 share pooled
bases.  Attention-output bases remain per-head and block diagonal.  There are
at most 24 unique tensors, still far fewer than the per-layer quality path.

Expected benefit: captures broad depth drift at modest storage cost.  Risk:
less memory reduction and online reuse than a single per-family basis.

## Result

Failed the quality screen.

- Active online basis storage: 33,177,600 bytes (7.0x reduction).
- Mean protected-energy retention over active sources: 82.82%, the best of
  the shared schemes.
- 128-image means: PSNR 18.8378 dB, LPIPS 0.25391, SSIM 0.70920, CLIP 26.7850.
- Paired versus per-linear PCA: PSNR -0.6631 dB, 95% CI
  [-1.0335, -0.3252], win rate 35.9%; LPIPS +0.02870,
  [+0.01613, +0.04286], win rate 34.4%; SSIM -0.02498,
  [-0.03536, -0.01574].
- CLIP delta +0.1166 had CI [-0.1420, +0.3755].

Conclusion: coarse depth conditioning recovers calibration protected energy,
but did not recover image quality and was slightly worse than the cheaper
per-operator pooled scheme. Protected-energy retention alone is therefore
not a sufficient selector for a shared basis. This arm is not eligible for
5K.
