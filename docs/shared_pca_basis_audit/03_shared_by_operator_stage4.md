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

Pending formal run.

