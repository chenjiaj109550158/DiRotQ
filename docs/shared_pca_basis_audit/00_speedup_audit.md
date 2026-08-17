# Audit of the repository speed/memory path

The checked speed scripts do **not** load the calibration PCA file used by
`apply_dirotq.py`.  Both `report_flux_w4a4_no_rot_ffdown.py` and
`report_flux_nvfp4_no_rot_ffdown.py` create one unseeded dense matrix with
`qr(randn(3072, 3072))` and use it before every K=3072 projection.  K=12288
FFN-down projections are left unrotated.  Thus the implemented grouping is
effectively one shared K=3072 transform, not a per-linear PCA basis.

There is also a correctness problem in the checked scripts: activations are
changed from `X` to `XU`, while `pack_into` receives the original stored
weight.  The packer has no `U` argument and performs only group RTN/packing.
Consequently the script implements `(XU)W`, not the equivalent
`(XU)(U^T W)`.  The memory/latency measurement remains a measurement of
packed tensor sizes and kernel timings, but it is not an accuracy-equivalent
implementation of the model.

The wrappers' low-rank `proj_down`/`proj_up` tensors are explicitly filled
with zeros.  Therefore this speed arm also has no functioning BF16 protected
PCA branch; all packed Linear payload is handled by the low-precision kernel.
That is a third numerical difference from the paper quality method, in
addition to shared random-Q versus per-layer PCA and RTN versus GPTQ.

For that reason the literal speed script is treated as a failed correctness
diagnostic, not as a quality arm.  Every scheme in this audit shares a basis
but also counter-rotates/transforms each weight through the existing DiRotQ
weight path.  No quality conclusion will be attributed to the unchecked
speed arm.

The speed documents also explicitly say that their NVFP4 pack is plain RTN
and that fully calibrated accuracy is separate (`apply_dirotq.py`).  Therefore
latency/memory numbers and paper image-quality numbers currently describe
different numerical methods even before basis sharing is considered.
