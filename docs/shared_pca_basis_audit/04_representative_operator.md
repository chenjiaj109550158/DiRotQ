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

Pending formal run.

