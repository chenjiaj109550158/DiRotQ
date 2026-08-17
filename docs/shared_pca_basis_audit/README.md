# Shared PCA basis quality audit

This directory pre-registers the alternatives used to test whether DiRotQ's
quality survives the basis-sharing assumption needed by the repository's
memory/speed path.  The primary model is PixArt-Sigma because the paper's
quoted 15.9 FID / 19.1 dB PSNR result is a PixArt-Sigma result and the local
workspace contains its complete 5120-call calibration, PCA, and Hessian
provenance.

All quality arms keep the paper configuration fixed: INT W4A4, GPTQ,
`ff.net.2` activation quantization skipped, random residual rotation, 20
steps, CFG 4.5, MJHQ order/seeds, and batch size 4.  Only the PCA basis sharing
rule changes.  Pre-rotation Hessians are reused read-only; each derived basis
gets a distinct transformed-weight GPTQ cache.

The four pre-registered schemes are:

1. [shared by input width](01_shared_by_width.md)
2. [shared by operator family](02_shared_by_operator.md)
3. [shared by operator family and depth stage](03_shared_by_operator_stage4.md)
4. [fixed representative layer per operator](04_representative_operator.md)

The speed implementation audit and its fail-closed interpretation are in
[00_speedup_audit.md](00_speedup_audit.md).  Results are added to the
individual files only after each arm is frozen and run.

## Pre-registered execution ladder

Each scheme must pass, in order:

1. covariance/basis shape and orthogonality tests;
2. unquantized `XW == (XU)(W U)^T` parity;
3. proof that modules in one sharing group reference one GPU rotation storage;
4. 4-image generation smoke;
5. matched 128-prompt paired evaluation against FP16 and the per-layer-PCA
   DiRotQ baseline.

An arm that fails correctness is not allowed to generate images.  An arm that
produces corrupt/flat images or a GPTQ fallback stops at smoke.  FID is not
estimated from 128 samples.  A statement that the paper's 15.9 FID is
reproduced requires the official 5000-image evaluation; pilot metrics can only
establish retention or loss relative to the matched per-layer baseline.

## Quality decision

A shared scheme is considered to retain the paper method on the 128-prompt
screen only when its paired PSNR and LPIPS means versus per-layer PCA are both
no worse than -0.10 dB / +0.002 respectively, neither 95% bootstrap interval
supports degradation, and CLIP is not significantly worse.  Passing this
screen makes it eligible for the full official 5K FID/PSNR run; it does not by
itself reproduce the paper number.

