# FP8 3x high-rank + E0 low-branch on SANA-1.6B (Ada)

## Outcome

Classification: **HIGH WEIGHT FP8 BOTTLENECK**.

The matched BF16 3x-rank ceiling improves Dev32 teacher-forced raw SSE by
20.98% over the rebuilt current-E0 baseline. Plain E4M3 W8A16 retains only
18.10% of that ceiling (3.80% improvement over baseline), while MXFP8 W8A16
is already 15.39% worse than baseline. Quantizing the high activation adds a
further 11.12% error for E4M3 and 45.84% for MXFP8. Both W8A8 arms fail the
pre-registered Dev gate, so final and Pilot64 were not read or run.

## Frozen provenance

- Formal implementation commit: `000825aab1ba8bca1b08b45f0781bacf9c216d4d`
- Diagnostics-only commit: `ddcbc85d8260701945c757bd18c9b0744612ff59`
- Model: `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`
- Model revision: `e2b3c0cbffebcd09d83805e88b9f5f106afc74ac`
- Dataset SHA-256: `07ce5ef172dc0454c0267ad7a68a16e21ae2e695356651a51a9f303a166b120e`
- Split-manifest content SHA-256: `877f1a68e197a6731d8c01e6a93d2fb92abdbdf2a76212d60e2d585470e68be2`
- Fit/dev/final/pilot sizes: 64/32/64/64. Fit steps: `[0,5,10,15,19]`.
- The four splits are mutually disjoint by ID, exact prompt hash and normalized
  prompt hash, and do not overlap the locally visible parseable historical
  manifests/image IDs. Historical provenance is incomplete, so no stronger
  leakage claim is made.
- Frozen PCA SHA-256: `7fb5d472e4607b774c545fc3fb9e9c949d5ebe8c531ff3512a28ab90364d9662`
- Matched rank-r R SHA-256: `1dbca2c2ed69dcbc4cfb4ddc9de8920c9d8678180b2ea5d057a7b3ce1a5f097b`
- Matched rank-3r R SHA-256: `1cc0e82b2b9f62a5562edb18c1678c27bd7578eaba72045fcaee53504aaa13cd`

The primary comparison regenerates both residual rotations with the production
algorithm and root seed 42: float64 Gaussian matrix, QR, diagonal-sign
canonicalization. This is necessary because the historical production R has a
280/1960 metadata split while current K16 routing yields 288/1952. Attention
output uses independent head-local rotations (4/28 versus 12/20 per head),
with no cross-head mixing.

## Matched build

Both rank configurations use the same 640 frozen fit chunks (64 prompts, five
steps and two CFG branches). Each rank has its own transformed low target,
hardware-E0 activation Hessian and hardware-E0 GPTQ cache.

| Rank | High/low input | High/low per head | HZ seconds | GPTQ/build seconds | GPTQ | fallback | unquantized max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| r | 288/1952 | 4/28 | 147.35 | 96.03 | 120/120 | 0 | 2.594e-4 |
| 3r | 864/1376 | 12/20 | 145.64 | 78.69 | 120/120 | 0 | 2.747e-4 |

The B1/E4/MX 3r arms share byte-identical low E0 payload/scales and HZ:

- HZ SHA-256: `0e96f4b467a5af30dea3a8f7ec297ff619aa2530fee2c8856d1f99779ef2f7bd`
- Low sidecar SHA-256: `56de9731fb6a79f5cf39797c431b00a90f16ca39805279cc02bd8ecc68bbfc17`
- GPTQ coverage: 120/120; RTN/CPU/silent fallback: 0/0/0.

## Serialized active weights

These figures include payloads, layer-global FP32 scales, K16 UE4M3 low
scales, K32 UE8M0 high scales and padding represented by the sidecars.

| Arm family | Active bytes | Relative to B0 |
|---|---:|---:|
| B0 BF16-r + E0-low | 449,344,480 | 1.0000x |
| B1 BF16-3r + E0-low | 669,850,080 | 1.4906x |
| E4 3r + E0-low | 439,757,760 | 0.9787x |
| MXFP8 3r + E0-low | 449,165,280 | 0.9996x |

Thus exact 3r E4M3 and MXFP8 satisfy the <=1% B0 persistent active-weight
budget and no `r_byte` control is needed. Adding unchanged non-active BF16
transformer state gives theoretical packed transformer totals of 3,648,953,024
bytes (E4), 3,658,360,544 bytes (MX) and 3,658,539,744 bytes (B0). The frozen
PCA file is 815,051,202 bytes; matched rank-r/rank-3r rotation files are
70,640,208/55,301,720 bytes and are reported separately as method constants.

For a batch-4 input projection operand (`M=4096`), the serialized activation
buffers are 6,856,708 bytes (B0), 6,709,256 (E4-AW) and 6,819,844 (MX-AW).
Across the 100 input and 20 output wrapped projections at the same M, the
theoretical totals are 823,296,480, 808,551,360 and 821,821,920 bytes.

## Dev32 teacher-forced result

All arms consume the same 1,280 cached BF16 teacher calls (32 prompts x 20
steps x two CFG branches). Lower SSE is better.

| Arm | Raw J | Relative MSE | Cosine | gain vs B0 | prompt wins | wall s | peak alloc/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 16,654.745 | 2.81259e-4 | 0.99985994 | reference | - | 101.24 | 4.33 / 4.63 |
| B1 | 13,160.012 | 2.22241e-4 | 0.99989131 | +20.983% | 32/32 | 95.54 | 4.26 / 4.55 |
| E4-W | 16,022.160 | 2.70576e-4 | 0.99986604 | +3.798% | 27/32 | 94.69 | 4.26 / 4.55 |
| E4-AW | 17,803.739 | 3.00662e-4 | 0.99985027 | -6.899% | 3/32 | 107.14 | 4.26 / 4.55 |
| MX-W | 19,218.459 | 3.24554e-4 | 0.99983941 | -15.393% | 1/32 | 94.26 | 4.26 / 4.55 |
| MX-AW | 28,028.022 | 4.73326e-4 | 0.99977250 | -68.289% | 0/32 | 108.52 | 4.26 / 4.55 |

Prompt-bootstrap 95% CIs for mean prompt SSE delta versus B0 are: B1
`[-120.60,-98.31]`, E4-W `[-27.28,-13.33]`, E4-AW `[23.53,46.37]`, MX-W
`[68.20,92.53]`, and MX-AW `[331.67,379.45]`.

Dev gate details for E4-AW: aggregate gain -6.899%, 3/32 prompt wins,
early/mid/late changes +4.06%/+9.41%/+7.93%. For MX-AW: -68.289%, 0/32,
+39.87%/+84.13%/+86.81%. Neither arm passes.

## Mechanism diagnostics

- Rank ceiling `B0-B1`: 3,494.733 raw SSE.
- Ceiling retained: E4-W 18.10%; E4-AW -32.88%; MX-W -73.36%; MX-AW -325.44%.
- High-weight relative SSE: BF16 2.712e-6, E4M3 7.148e-4, MXFP8 8.486e-4.
- High-activation relative SSE: E4M3 5.677e-4, MXFP8 1.405e-3.
- High activation saturation: E4M3 0%; MXFP8 0.7684%.
- High weight saturation: E4M3 0%; MXFP8 0.8547%.
- Plain E4M3 activation global scales span 1.492e-5 to 16. MXFP8 decoded
  K32 scales span 1.907e-6 to 16.
- Shared low-weight relative SSE: 1.4753%; runtime low-activation relative SSE
  is about 0.7457% in both W8A8 trajectories.

The E4M3 W8A16 control shows that high-weight RTN already removes most of the
rank benefit; high activation quantization then reverses the remaining gain.
MXFP8's power-of-two K32 scaling has higher high-weight/high-activation error
and legal saturation, and is worse before activation quantization.

## Hardware boundary and stop

Plain E4M3 representative aligned CUDA tiles pass public
`torch._scaled_mm` versus the decoded FP32 reference on Ada. The formal model
quality runs use deterministic fake-quantized BF16 materialization in the
existing DiRotQ wrapper; their Python wall times are not production latency.
E0 low and MXFP8 are software/hardware-faithful references on Ada, not native
Ada execution. No kernel, Nunchaku, CUTLASS, CUBIN or SASS source was changed.

The pre-registered Dev continuation gate failed. Final/Pilot split definitions
were frozen up front for overlap auditing, but no final teacher trajectory,
final model evaluation or Pilot prompt was executed. No free-run trajectory
was evaluated, no image was generated, no Pilot64 metric exists, and no 5090
handoff was produced.

Official format references used for the MX contract:

- OCP Microscaling Formats v1.0: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- CUTLASS Blackwell functionality: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md
