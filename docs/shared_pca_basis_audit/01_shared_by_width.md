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

Pending formal run.

