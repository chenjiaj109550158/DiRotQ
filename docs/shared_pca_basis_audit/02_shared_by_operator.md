# Scheme 2: pooled basis per operator family

All blocks share a basis, but different operator semantics do not:

- self-attention Q/K/V input;
- cross-attention query input;
- FFN-up input;
- self-attention output input, per head;
- cross-attention output input, per head;
- FFN-down input.

Covariances are equal-source means over the 28 transformer blocks, using the
same read-only pre-rotation Hessian provenance and damping/order convention as
Scheme 1.  Q/K/V share because they receive the same activation in the model.

Expected benefit: only six unique basis tensors while respecting semantic
families.  This is the primary interpretation of “all same-type Linear layers
use one PCA basis.”

## Result

Failed the quality screen, although it was the best shared-basis image arm.

- Active online basis storage: 8,294,400 bytes (28.0x reduction).
- Mean protected-energy retention over active sources: 69.67%.
- 128-image means: PSNR 18.9514 dB, LPIPS 0.25342, SSIM 0.70952, CLIP 26.8476.
- Paired versus per-linear PCA: PSNR -0.5495 dB, 95% CI
  [-0.8674, -0.2215], win rate 39.1%; LPIPS +0.02822,
  [+0.01692, +0.03920], win rate 28.9%; SSIM -0.02466,
  [-0.03237, -0.01657].
- CLIP delta +0.1792 had CI [-0.0701, +0.4372], so no significant semantic
  regression was detected.

Conclusion: respecting operator semantics is materially better than sharing
only by width, but still clearly worse than per-linear PCA. It also falls
slightly below the paper's quoted 19.1 dB on this first-128 screen and is not
eligible for 5K.
