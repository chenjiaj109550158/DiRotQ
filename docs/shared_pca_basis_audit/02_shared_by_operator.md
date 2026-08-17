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

Pending formal run.

