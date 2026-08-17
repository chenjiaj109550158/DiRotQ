# Scheme 1: one pooled basis per input layout/width

This is the most aggressive valid sharing rule and the closest quality-side
analogue of the speed script.

- One 1152x1152 basis is shared by self-attention input, cross-attention
  query, and FFN-up projections in all 28 blocks.
- One `[16,72,72]` head-isolated basis is shared by self- and cross-attention
  output projections in all blocks.  It never mixes heads.
- One 4608x4608 basis is shared by every FFN-down projection.

Each pooled covariance is the equal-source mean of the matching read-only
pre-rotation Hessians.  Equal-source weighting is deliberate and frozen; the
existing Hessian cache does not retain original row counts.  Isotropic 1%
damping is applied only before eigendecomposition, as in the existing basis
collector.  Eigenvectors are ascending-energy so the protected tail retains
the existing high-rank convention.

Expected benefit: minimum basis/rotation storage (three unique tensors).
Risk: unrelated semantic families may have incompatible high-variance
directions.

## Result

Failed the quality screen.

- Active online basis storage: 2,820,096 bytes (82.35x smaller than the
  232,243,200-byte per-linear active basis set).
- Mean protected-energy retention over active sources: 57.74% of per-linear
  PCA.
- 128-image means: PSNR 18.6798 dB, LPIPS 0.26550, SSIM 0.69771, CLIP 26.8263.
- Paired versus per-linear PCA: PSNR -0.8211 dB, 95% CI
  [-1.1731, -0.4662], win rate 29.7%; LPIPS +0.04030,
  [+0.02678, +0.05388], win rate 21.1%; SSIM -0.03648,
  [-0.04711, -0.02595].
- CLIP delta was +0.1579 with CI [-0.1180, +0.4194], so semantic alignment
  did not significantly regress, but image fidelity did.

Conclusion: sharing only by input layout/width gives the largest basis-memory
reduction and the clearest quality loss. It is not eligible for a 5K run.
