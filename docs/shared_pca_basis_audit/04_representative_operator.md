# Scheme 4: fixed representative basis per operator family

For every operator family in Scheme 2, all blocks use the existing per-layer
PCA basis from block 14.  The representative is fixed in advance (the first
block in the third depth stage), not selected using image or evaluation
metrics.

This arm separates two effects:

- storage/reuse from sharing one basis;
- the benefit of recomputing a pooled covariance.

It requires no new covariance eigensolve and is also a useful deterministic
fallback if pooled-eigensolver provenance fails.  It has the same number of
unique tensors as Scheme 2 but is expected to fit early/late blocks less well.

## Result

Failed the quality screen and was the worst shared-basis arm.

- Active online basis storage: 8,294,400 bytes (28.0x reduction, identical to
  pooled per-operator sharing).
- Mean protected-energy retention over active sources: 49.57%.
- 128-image means: PSNR 18.1143 dB, LPIPS 0.29448, SSIM 0.67319, CLIP 26.9041.
- Paired versus per-linear PCA: PSNR -1.3866 dB, 95% CI
  [-1.7869, -0.9918], win rate 25.0%; LPIPS +0.06927,
  [+0.05634, +0.08282], win rate 14.8%; SSIM -0.06099,
  [-0.07131, -0.05073].
- CLIP delta +0.2357 had CI [-0.0707, +0.5515].

Conclusion: choosing an arbitrary fixed representative layer is not viable.
At the same storage as pooled per-operator sharing, pooling the calibration
covariances is decisively better, although pooling still fails the screen.
