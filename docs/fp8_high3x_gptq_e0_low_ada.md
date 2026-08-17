# FP8 high-3r GPTQ + E0-low evaluation on Ada

## Decision

**Classification: `HIGH WEIGHT GPTQ DOES NOT RECOVER RANK HEADROOM`.**

The best new arm was `MX-NEIGHBOR-GPTQ-W`.  It improved the frozen Dev32
teacher-forced raw squared error by 8.42% relative to B0 and won all 32
prompts, but recovered only 40.15% of the B0-to-B1 rank headroom.  The
pre-registered weight-only gate required at least 50%.  No arm passed all
weight-only gates, so the conditional A8 stage was not run.  Final and Pilot
splits were not read, no images were generated, and no kernel code was
changed.

This experiment does not support the hypothesis that high-weight RTN was the
main reason the earlier high-3r methods failed.  GPTQ helped the MXFP8 weight
arm end to end, but not enough to recover half of the available rank ceiling;
for per-channel E4M3 it reduced the fit Hessian objective while making Dev32
worse than matched RTN.

## Frozen provenance

- Branch: `exp/fp8-high3x-gptq-e0-low-ada`
- Formal source commit: `0f8b2f03e2c78689b98a804ea4789c34f9d4a443`
- Parent method checkpoint: `c5c80272d9033377fc9825312c3c4c41da52b8b5`
- Model revision: `e2b3c0cbffebcd09d83805e88b9f5f106afc74ac`
- Dataset SHA-256: `07ce5ef172dc0454c0267ad7a68a16e21ae2e695356651a51a9f303a166b120e`
- Fit manifest: `24aa0754a221d36e9a82235afe155a5ff7cf0a0c63327453041873a75b9168aa`
- Dev manifest: `fa40231eb1867f7991470fc109b8e64a757dffbc7e7185c8ab5beb3263d903e7`
- PCA basis: `7fb5d472e4607b774c545fc3fb9e9c949d5ebe8c531ff3512a28ab90364d9662`
- Matched rank-3r residual rotation: `1cc0e82b2b9f62a5562edb18c1678c27bd7578eaba72045fcaee53504aaa13cd`
- Shared rank-3r low E0 Hessian: `0e96f4b467a5af30dea3a8f7ec297ff619aa2530fee2c8856d1f99779ef2f7bd`
- Shared rank-3r low E0 packing sidecar: `56de9731fb6a79f5cf39797c431b00a90f16ca39805279cc02bd8ecc68bbfc17`
- Shared B1 materialized cache: `7a7ede5f8e4d17ba743d2bea1171d34d024aa1c288fc2dbe395535cb5bf80d6f`
- New high Hessian: `afd77d32538241065a265c872e3f7e12627e75a34cefe2df4d5d4b52baf1bd72`

All frozen input artifacts retained their preflight SHA-256, size, and mtime
after the experiment.  The method freeze records `final_or_pilot_read=false`.
Fit comprises 640 CFG-expanded calls (64 prompts, five frozen timesteps), and
Dev comprises 1,280 CFG-expanded calls (32 prompts, all 20 timesteps).

The split and basis remained rank matched: ordinary input projections use
864 high / 1376 low directions; attention-output projections use 70 isolated
heads with 12 high / 20 low directions per head.  All new arms reuse the
same low E0 payload/scales, low Hessian, PCA, residual rotation, rank, and
teacher inputs byte for byte.

## Environment

- GPU: physical 4, NVIDIA RTX 6000 Ada Generation, compute capability 8.9,
  49,140 MiB; logical CUDA index 0 under `CUDA_VISIBLE_DEVICES=4`
- Driver: 565.57.01
- CUDA toolkit: 12.6 (V12.6.85)
- Python: 3.12.13
- PyTorch: 2.6.0+cu124; PyTorch CUDA: 12.4
- GPU contract: `NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1`

GPU 4 was shared with a pre-existing high-utilization process.  This increased
wall time but did not change batch size, recipes, source, or gates.  The
reported Python/fake-quant wall times are not production FP8 latency.

## High-Hessian and quantizer contract

For every fixed high-3r layer, the fit-only Hessian is

```text
H_h = (2 / sum_c M_c) * sum_c A_h,c^T A_h,c
A_h,c = X_c V_h
```

Collection covered 120/120 layers, 655,360 rows and 640 chunks per layer.
It took 435.45 s, used 351,774,720 tensor bytes, and peaked at 3.875 GiB
allocated / 4.197 GiB reserved.  Damping was 1%; quantization used the same
sequential error-compensation ordering for all GPTQ arms.  There were zero
RTN fallbacks and zero CPU fallbacks.

E4 uses legal signed finite E4M3 bytes with a frozen per-output-channel FP32
scale.  Fit compares the predeclared multipliers 0.875, 1.0, and 1.125;
selection counts were 14,070 / 170,983 / 83,747 output channels.  MX uses
legal signed E4M3 payload plus UE8M0 K32 scale bytes.  `MX-BEST-RTN-W` chose
the fit-frozen no-saturation recipe; `MX-NEIGHBOR-GPTQ-W` freezes neighboring
scale exponents before sequential GPTQ.

The 120-layer build took 840.53 s and peaked at 3.232 GiB allocated / 3.314
GiB reserved.  GPTQ changed 89,717,184 E4 payload elements (38.99%) and
96,192,509 MX payload elements (41.81%), proving the GPTQ paths are not RTN
aliases.  Each GPTQ layer completed on its first Cholesky attempt.

## Fit diagnostics

| Weight arm | Raw weight SSE | H-weighted error | H-error reduction vs matched RTN | Saturation | Saturation rate |
|---|---:|---:|---:|---:|---:|
| E4-PC-RTN | 20,008.437 | 40,907,589.430 | — | 33,641 | 0.014621% |
| E4-PC-GPTQ | 94,133.948 | 10,785,472.494 | 73.64% | 33,641 | 0.014621% |
| MX-BEST-RTN | 18,827.139 | 226,306,985.129 | — | 406 | 0.000176% |
| MX-NEIGHBOR-GPTQ | 121,640.737 | 41,577,667.281 | 81.63% | 792 | 0.000344% |

For MX RTN, the frozen alternatives were: current recipe
362,455,773.273 H-error, no-saturation 226,306,985.135, and neighbor
226,306,985.207.  The dominant UE8M0 exponents were -10 and -9.  The raw SSE
increase under GPTQ is expected because its objective is activation weighted;
the decisive question is transfer to the matched denoiser metric below.

## Serialized persistent-weight budget

| Arm | Persistent bytes | Ratio vs B0 |
|---|---:|---:|
| B0 | 449,344,480 | 1.000000 |
| E4-PC-RTN-W | 440,832,480 | 0.981057 |
| E4-PC-GPTQ-W | 440,832,480 | 0.981057 |
| MX-BEST-RTN-W | 449,165,280 | 0.999601 |
| MX-NEIGHBOR-GPTQ-W | 449,165,280 | 0.999601 |

E4 sidecars contain 230,092,800 payload bytes and 1,075,200 FP32 scale
bytes.  MX sidecars contain 232,243,200 payload bytes and 7,257,600 UE8M0
scale bytes, including required padding.  Decoded BF16 materialization is not
counted as serialized FP8 weight storage.

The frozen batch-independent per-row activation serialization estimates were
1,674 bytes for B0, 1,642 for E4 A8, and 1,665 for MX A8.  They were recorded
before Dev, but A8 was not executed because the weight gate failed.

## Dev32 W8A16 results

Primary `J` is the raw sum of squared differences from the shared BF16
teacher.  `Group delta` is `(arm - B0) / B0`, so positive values are worse.

| Arm | J | Relative MSE | Cosine | Raw gain vs B0 | Equal-prompt mean / median gain | Wins | Early / mid / late delta | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 16,654.745 | 2.81259e-4 | — | — | — | — | — | 0% |
| B1 rank ceiling | 13,160.012 | 2.22241e-4 | — | 20.98% | 21.11% / 22.12% | 32/32 | -6.41% / -26.23% / -33.02% | 100% |
| old E4-W | 16,022.160 | 2.70576e-4 | — | 3.80% | 3.66% / 3.02% | 27/32 | +0.81% / -3.56% / -9.27% | 18.10% |
| old MX-W | 19,218.459 | 3.24554e-4 | — | -15.39% | -16.38% / -15.34% | 1/32 | +10.40% / +20.16% / +16.90% | -73.36% |
| E4-PC-RTN-W | 15,566.385 | 2.62879e-4 | 0.99987015 | 6.53% | 6.34% / 7.32% | 25/32 | +5.93% / -10.15% / -17.60% | 31.14% |
| E4-PC-GPTQ-W | 16,010.415 | 2.70377e-4 | 0.99986610 | 3.87% | 3.46% / 4.85% | 24/32 | +16.27% / -8.98% / -22.38% | 18.44% |
| MX-BEST-RTN-W | 15,889.485 | 2.68335e-4 | 0.99986612 | 4.59% | 4.30% / 4.87% | 28/32 | +0.16% / -3.88% / -10.65% | 21.90% |
| MX-NEIGHBOR-GPTQ-W | 15,251.624 | 2.57563e-4 | 0.99987252 | 8.42% | 8.22% / 7.85% | 32/32 | +1.64% / -7.87% / -20.40% | 40.15% |

E4 GPTQ was 2.85% worse than matched E4 RTN and won only 10/32 paired
prompts.  MX GPTQ was 4.01% better than matched MX RTN and won 25/32 paired
prompts.  Thus RTN was a meaningful MX bottleneck, but removing it was not
sufficient to meet the rank-headroom gate.  E4's fit-optimized GPTQ objective
did not transfer at all.

Dev runtimes were 419.85 s (E4 RTN), 409.65 s (E4 GPTQ), 405.40 s (MX RTN),
and 406.31 s (MX GPTQ).  Every arm peaked at 4.257 GiB allocated / 4.549 GiB
reserved.  These measurements include software materialization and shared-GPU
contention and are not native format latency.

## Gate and scope outcome

The weight-only gate required all of: recovery at least 50%, 24/32 prompt
wins, no early/mid/late degradation over 2%, budget within 1%, and complete
correctness/fallback gates.  The best arm met wins, timestep, budget, and
correctness constraints but failed recovery (40.15%).  E4 arms also failed
recovery and degraded the early group by more than 2%.

Consequently:

- no `E4-PTPC-GPTQ-AW` or `MX-K32-GPTQ-AW` Dev run was started;
- activation-only and joint A8 losses are intentionally unavailable;
- final and Pilot splits were not read;
- no images, VAE decode, 5090 handoff, kernels, CUTLASS, CUBIN, or SASS changes
  were produced;
- Ada native E4 parity is a correctness check only; MXFP8 and E0 remain
  hardware-faithful software references on Ada.

## Commands

All GPU commands used `NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4
HF_HUB_OFFLINE=1` with the `dirotq` environment.  The core commands were:

```bash
conda run -n dirotq python -m pytest -q
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 \
  conda run -n dirotq python -m pytest -q

env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
  conda run --no-capture-output -n dirotq \
  python metrics/evaluate_fp8_high_gptq.py --batch-size 4 preflight
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
  conda run --no-capture-output -n dirotq \
  python metrics/evaluate_fp8_high_gptq.py --batch-size 4 smoke
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
  conda run --no-capture-output -n dirotq \
  python metrics/evaluate_fp8_high_gptq.py --batch-size 4 hessian
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
  conda run --no-capture-output -n dirotq \
  python metrics/evaluate_fp8_high_gptq.py --batch-size 4 build
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
  conda run --no-capture-output -n dirotq \
  python metrics/evaluate_fp8_high_gptq.py --batch-size 4 freeze

# Repeated once per frozen arm:
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
  conda run --no-capture-output -n dirotq \
  python metrics/evaluate_fp8_high_gptq.py --batch-size 4 evaluate \
  --arm ARM_NAME

conda run -n dirotq python metrics/evaluate_fp8_high_gptq.py \
  --batch-size 4 weight-gate
git diff --check
```

One monitoring mistake interrupted the first formal build before it wrote any
sidecar.  Its empty 4 KiB staging directory was preserved as
`high_weights.interrupted-monitoring-20260817`; the successful formal build
then started from a new staging path and produced the hashes above.  No valid
artifact was overwritten or deleted.

The complete untracked experiment directory is
`models/sana-1.6b/fp8_high3x_gptq_e0_low_ada/` (about 1.3 GiB).  It contains
the high Hessian, four packing sidecars, per-layer objectives, method freeze,
four Dev summaries/per-call CSVs, and the deterministic weight-gate report.
