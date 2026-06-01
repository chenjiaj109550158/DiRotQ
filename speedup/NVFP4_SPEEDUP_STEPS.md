# NVFP4 speedup on Blackwell — implementation steps

Goal: reproduce the DiRotQ-vs-fp16 speedup measurement we have for **INT4
W4A4** ([results/flux_w4a4_no_rot_ffdown.json](results/flux_w4a4_no_rot_ffdown.json)),
but using **NVFP4** (FP4 E2M1 + FP8 micro-scales) so it actually runs on
the tensor cores of this machine's **RTX PRO 6000 Blackwell (sm_120)**.

---

## ✅ STATUS: DONE

Implemented and measured end-to-end on the RTX PRO 6000 Blackwell (96 GB).

**Measured (M=4608, real FLUX.1-dev weights), with the fused FP8 rotation:**

| B | fp16 native | NVFP4 | speedup | peak VRAM |
|---|-------------|-------|---------|-----------|
| 1 | 391 ms      | 192 ms| 2.04×   | 24.3 → 12.1 GB |
| 2 | 797 ms      | 370 ms| 2.15×   | 24.8 → 12.4 GB |
| 4 | 1663 ms     | 745 ms| 2.23×   | 25.8 → 13.1 GB |

Weights 2.01× smaller. Compute-bound ~2.0–2.2× (no offload on 96 GB), exactly
as predicted in §4 — the INT4/4090 "9× at B=2" was a 24 GB offload artifact.

> Rotation kernel iterations (the rotation was ~25% of the forward — the real
> Blackwell bottleneck, not the FP4 GEMM at ~0.13 ms):
>
> | rotation | per-K=3072 | speedup (B=1/2/4) |
> |----------|-----------|-------------------|
> | int8-Triton (Ada-tuned) | ~0.64 ms | 1.53 / 1.66 / 1.71× |
> | bf16 cuBLAS | ~0.31 ms | 1.76 / 1.95 / 2.00× |
> | fused FP8 (e4m3) | ~0.18 ms | 2.04 / 2.15 / 2.23× |
>
> The NVFP4 script uses [kernels/fp8_rotation.py](kernels/fp8_rotation.py);
> [kernels/bf16_rotation.py](kernels/bf16_rotation.py) is the full-precision
> fallback; the INT4 script keeps the int8-Triton kernel (wins on Ada).

**What shipped:**
- `convert_dirotq_low_weight_to_nunchaku_fp4` in
  [nunchaku_pack.py](nunchaku_pack.py) (group-16 RTN, `amax/6` → e4m3,
  `float_point=True`).
- Rotation kernels for the NVFP4 path:
  [kernels/fp8_rotation.py](kernels/fp8_rotation.py) (fused e4m3 + `_scaled_mm`,
  default) and [kernels/bf16_rotation.py](kernels/bf16_rotation.py) (dense
  cuBLAS fallback). Both share the int8 rotation's call signature.
- [report_flux_nvfp4_no_rot_ffdown.py](report_flux_nvfp4_no_rot_ffdown.py) —
  cloned from the INT4 script; `SVDQW4A4Linear(precision='nvfp4')`,
  `wcscales`=ones, `wtscale`=1.0, fp16 baseline measured native at all B,
  rotation via the fused FP8 kernel.
- Output: [results/flux_nvfp4_no_rot_ffdown.json](results/flux_nvfp4_no_rot_ffdown.json).

**Actual env (resolved the §2 unknowns):** prebuilt wheel
`nunchaku-1.2.1+cu12.8torch2.11-cp312` (no source build needed) on the
`dirotq` conda env (torch 2.11.0+cu128, Python 3.12). `SVDQW4A4Linear` takes
`precision='nvfp4'`; the NVFP4 scaling is three-level (per-group e4m3
`wscales`, per-channel e4m3 `wcscales`, global float `wtscale`) — we fold
everything into `wscales` and leave the other two at identity. torchao is
**not** required (transformers 4.57 tolerates its absence).

The remainder of this file is the original plan, kept for context.

---

## Original plan (pre-implementation)

---

## 0. Why we need a separate NVFP4 path at all

The current speedup numbers come from nunchaku's SVDQuant `gemm_w4a4`
kernel, which uses `mma.m16n8k64.s4.s4.s32` — **dedicated INT4 tensor
cores**. Those exist only on **sm_75–sm_89 (Turing→Ada, e.g. the RTX 4090
the numbers were taken on)**. This is already called out in
[INSTALL.md](INSTALL.md) §1:

> Hopper (sm_90) and Blackwell removed int4 tensor cores; nunchaku falls
> back to int8 there and you'll see different absolute numbers.

So on this Blackwell card the INT4 script does **not** measure an INT4
kernel — it silently runs INT8 and the headline speedups don't reproduce.

Blackwell's replacement for "fast 4-bit" is **NVFP4**: 4-bit elements in
FP4 **E2M1** format, quantized in **groups of 16** with a per-group **FP8
E4M3** micro-scale, plus a per-tensor FP32 global scale. It has native
tensor-core support on sm_120, so it's the right format to chase the same
"4-bit weight + 4-bit activation, ~2× memory, faster GEMM" story here.

---

## 1. What already exists (don't rebuild these)

| Piece | Where | Status for NVFP4 |
| --- | --- | --- |
| E2M1 codebook + FP4 rounding | [nunchaku_pack.py:63](speedup/nunchaku_pack.py#L63) (`fp_quantize`) | ✅ already the `[0,.5,1,1.5,2,3,4,6]±` E2M1 set |
| FP8-e4m3 micro-scale packing (group 16) | [nunchaku_pack.py:159](speedup/nunchaku_pack.py#L159) (`pack_micro_scale`), asserts `group_size==16` | ✅ ready |
| `float_point` switch in the converter | [nunchaku_pack.py:248](speedup/nunchaku_pack.py#L248), `convert_to_nunchaku_w4x4y16(..., float_point=True)` | ✅ ready — branches to `fp_quantize` at [L288](speedup/nunchaku_pack.py#L288) |
| DiRotQ NVFP4 *accuracy* recipe (group 16, FP8 scaling) | [utils/quant_utils.py:752](utils/quant_utils.py#L752) `_quant_group_nvfp4`, [:806](utils/quant_utils.py#L806) `nvfp4_rtn_quantize_weights`; `--nvfp4` in [apply_dirotq.py:145](apply_dirotq.py#L145) | ✅ confirms the math/group size we should mirror |
| DiRotQ rotation injection (U on K=3072) | [kernels/int8_rotation.py](speedup/kernels/int8_rotation.py) + the patched forwards in the report script | ✅ format-agnostic, reuse as-is (see Step 4) |
| Two-phase subprocess orchestration, fp16 baseline, batch sweep, JSON | [report_flux_w4a4_no_rot_ffdown.py](speedup/report_flux_w4a4_no_rot_ffdown.py) | ✅ clone it |

So the only **missing** pieces are: (a) a NVFP4 packer wrapper, (b) a
nunchaku FP4 module/kernel that runs on Blackwell, and (c) a cloned
report script that wires them together.

---

## 2. Prerequisites (the actual hard part)

The INT4 recipe in [INSTALL.md](INSTALL.md) §3 pins **torch 2.6.0+cu124 +
nunchaku 1.0.1+torch2.6**. That stack predates Blackwell FP4 and will not
work here. We need:

- **torch with cu128** — already installed in the `dirotq` conda env
  (`torch 2.11.0+cu128`, verified running sm_120 kernels). ⚠️ See risk in §7:
  nunchaku may not ship a wheel built against torch 2.11; a downgrade to a
  torch version that has a matching nunchaku Blackwell wheel (e.g. 2.7/2.8
  cu128) may be required, in a *separate* env so we don't disturb `dirotq`.
- **A nunchaku build with NVFP4 / Blackwell support.** Confirm the
  installed nunchaku exposes an FP4 precision before writing code:

  ```bash
  python - <<'PY'
  import nunchaku, inspect
  print("nunchaku", nunchaku.__version__)
  from nunchaku.models.linear import SVDQW4A4Linear
  print("SVDQW4A4Linear args:", inspect.signature(SVDQW4A4Linear.__init__))
  # look for a precision/format kwarg: "nvfp4" | "fp4" | "float4"
  PY
  ```

  nunchaku is **not currently installed** in `dirotq` (checked). Install
  the Blackwell-capable build per nunchaku's own release matrix.
- **FP4 checkpoint** as the structural template (the v2 module is built
  from a checkpoint's shapes, then we re-pack from fp16): the FP4 sibling
  of `nunchaku-tech/nunchaku-flux.1-dev`, i.e. an
  `svdq-fp4_r32-flux.1-dev.safetensors` (verify exact filename on the HF
  repo). The INT4 file used today is named in [INSTALL.md](INSTALL.md) §4.
- FLUX.1-dev fp16 weights — same gated download as the INT4 path, already
  documented in [INSTALL.md](INSTALL.md) §4. No change.

---

## 3. Step-by-step

### Step 1 — Confirm hardware + kernel availability
- `torch.cuda.get_device_capability()` → `(12, 0)` (already verified) and
  `'sm_120' in torch.cuda.get_arch_list()` (already verified for torch 2.11).
- Confirm nunchaku's FP4 GEMM is reachable (the precision kwarg probe in §2).
- If nunchaku has no Blackwell/FP4 wheel for the installed torch, build a
  parallel env (`dirotq-fp4`) on the torch version nunchaku supports.
  **Do not** modify the working `dirotq` env.

### Step 2 — Add an NVFP4 packer wrapper
Add a sibling to [`convert_dirotq_low_weight_to_nunchaku`](speedup/nunchaku_pack.py#L324),
e.g. `convert_dirotq_low_weight_to_nunchaku_fp4`. Differences from the INT4
wrapper, all of which the underlying converter already supports:

- `group_size = 16` (not 64). The micro-scale packer asserts this
  ([nunchaku_pack.py:162](speedup/nunchaku_pack.py#L162)).
- Per-group scale `= amax(|W_g|) / NF4_MAX` where `NF4_MAX = 6.0` (the top
  E2M1 code) — **not** `/7` or `/8`. This mirrors the accuracy recipe at
  [utils/quant_utils.py:763](utils/quant_utils.py#L763).
- Store `wscales` as **`torch.float8_e4m3fn`**; `pack_micro_scale`
  ([nunchaku_pack.py:163](speedup/nunchaku_pack.py#L163)) casts to e4m3 and
  lays them out for the `insn_k // group_size` micro-scale fragment.
- Carry the **per-tensor FP32 global scale** that NVFP4 double-quant needs.
  Check whether nunchaku's FP4 module expects it as a separate field
  (`wtscale` / `global_scale` / similar) and emit it; the INT4 path has no
  such field, so this is the one genuinely new tensor.
- Call the converter with `float_point=True` (routes to `fp_quantize`,
  [nunchaku_pack.py:288](speedup/nunchaku_pack.py#L288)).
- `qweight` stays packed 4-bit (`[N, n_low/2]` int8). The fragment layout
  for FP4 may differ from s4 — `MmaWeightPackerBase`
  ([nunchaku_pack.py:77](speedup/nunchaku_pack.py#L77)) is parameterized by
  `bits`/`comp_k`; verify `pack_weight` produces the layout nunchaku's FP4
  kernel reads (cross-check against deepcompressor's FP4 packer). This is
  the main correctness risk in the packer — see §6 validation.

### Step 3 — Clone the report script
Copy [report_flux_w4a4_no_rot_ffdown.py](speedup/report_flux_w4a4_no_rot_ffdown.py)
→ `report_flux_nvfp4_no_rot_ffdown.py`. Keep the whole skeleton (subprocess
phases, fp16 baseline, B=1/2/4 sweep, peak-mem capture, JSON writer). Change
only Phase 2:

- Instantiate the **FP4** v2 module / `SVDQW4A4Linear(precision="nvfp4")`
  (exact kwarg from the §2 probe) instead of the INT4 default.
- Call the new `convert_dirotq_low_weight_to_nunchaku_fp4` in `pack_into`
  ([report...#L314](speedup/report_flux_w4a4_no_rot_ffdown.py#L314)) and set
  the extra global-scale field on the module alongside `qweight`/`wscales`.
- Keep the **no-rot ff_down** policy: rotation on the K=3072 ops only,
  K=12288 `ff_down` stays unrotated inside `fused_gelu_mlp`. The FP4 kernel
  quantizes activations to FP4 the same way the INT4 kernel quantized to
  INT4 — the rotation-placement logic is unchanged.
- Update output path → `results/flux_nvfp4_no_rot_ffdown.json` and the
  report labels (`W4A4`→`NVFP4 W4A4`).

### Step 4 — Rotation kernel: reuse, with one check
The DiRotQ rotation `y = x @ U` injected before each K=3072 op
([report...#L196-300](speedup/report_flux_w4a4_no_rot_ffdown.py#L196-L300))
is **independent of the weight quant format** — it operates on bf16/fp16
activations before they enter the fused quant+GEMM. Reuse
[kernels/int8_rotation.py](speedup/kernels/int8_rotation.py) unchanged.
Only verify the rotated activation's dtype is what the FP4 kernel's fused
activation-quant expects (bf16 in, FP4 produced inside the kernel). If the
FP4 kernel wants fp16 vs bf16 specifically, match `DTYPE` accordingly.

### Step 5 — Run + record
```bash
conda activate dirotq      # or dirotq-fp4 if a separate env was needed
export HUGGING_FACE_HUB_TOKEN=hf_...        # gated repos, see INSTALL.md §4
python -u speedup/report_flux_nvfp4_no_rot_ffdown.py
# phases can be run alone, same flags as the int4 script:
#   --phase=fp16   /   --phase=nunchaku
```
Writes `speedup/results/flux_nvfp4_no_rot_ffdown.json` with the same schema
as the INT4 result, so the two are directly diff-able.

### Step 6 — Validate before trusting the numbers
A fast kernel that computes garbage is worse than no number. Before
recording, check **correctness**, then speed:

- **Numerical:** for one Linear, compare FP4-kernel output vs the fp16
  reference (`W_low @ x`) and vs the repo's *fake-quant* NVFP4
  (`_quant_group_nvfp4`, [utils/quant_utils.py:752](utils/quant_utils.py#L752)).
  Relative error should be in the same ballpark as the fake-quant error,
  not arbitrarily large. Large error ⇒ packer fragment-layout or
  global-scale wiring is wrong (§2/Step 2).
- **It's actually FP4:** confirm no INT8 fallback warning, and that perf
  scales like 4-bit (peak weight memory ≈ INT4's, ~2× smaller than fp16).
- **Determinism:** the rotation `U` is built with `torch.randn` + QR
  ([report...#L309](speedup/report_flux_w4a4_no_rot_ffdown.py#L309)); seed
  it if you want run-to-run comparability, per
  [requirements.txt:8](requirements.txt#L8) note on reproducible rotations.

---

## 4. Expected results & honest caveats

**Compute speedup should be comparable to (plausibly better than) INT4-on-Ada.**
NVFP4 tensor cores on Blackwell are native and high-throughput, so the
per-GEMM win over fp16 should land in a similar 2–2.5× range to the INT4
B=1 number (272 ms vs 624 ms fp16 = 2.29× on the 4090, see
[results JSON](results/flux_w4a4_no_rot_ffdown.json)). Memory: ~2× weight
savings, same as INT4 (4-bit weights either way).

**The big batch-2/4 "9×" numbers will NOT reproduce, and that's expected.**
Those came from VRAM pressure, not compute: on the **24 GB** 4090, fp16
OOMs at B≥2 and falls back to `accelerate.cpu_offload`, which is
PCIe-bound (~26 GB/s), inflating fp16 to ~5 s at B=2 — that's where the
9.05× at B=2 comes from ([INSTALL.md](INSTALL.md) §1, §7; summary in the
JSON). This RTX PRO 6000 Blackwell has **~96 GB VRAM**, so fp16 fits
natively well past B=4 — no offload, no PCIe penalty. The fp16 baseline
will be its fast *native* latency, so NVFP4's speedup becomes a clean
**compute-bound ~2–2.5×** across all batch sizes rather than the
offload-driven spike.

➡️ When comparing to the INT4 table, **compare the B=1 native-vs-native
speedup** (the apples-to-apples figure). The B=2/4 INT4 speedups are a
property of the 24 GB card, not of the quant format. The new JSON's
`fp16_label` will read `native` (not `offload`) at B=2/4 on this GPU — note
this in the report so nobody reads it as a regression.

---

## 5. Deliverables checklist
- [ ] `speedup/nunchaku_pack.py`: add `convert_dirotq_low_weight_to_nunchaku_fp4`
- [ ] `speedup/report_flux_nvfp4_no_rot_ffdown.py`: cloned + FP4-wired
- [ ] `speedup/results/flux_nvfp4_no_rot_ffdown.json`: generated output
- [ ] `speedup/README.md` + `speedup/INSTALL.md`: add NVFP4 row / Blackwell
      env notes (torch cu128, nunchaku FP4 build, FP4 checkpoint name)
- [ ] Validation log (numerical error vs fp16 + fake-quant; FP4-not-INT8)

---

## 6. Open risks / things to verify first (in priority order)
1. **nunchaku Blackwell+FP4 wheel vs torch 2.11.** Highest-risk unknown.
   If no wheel matches torch 2.11+cu128, pick the torch version nunchaku
   does support and build a separate env. Don't disturb `dirotq`.
2. **Exact FP4 API surface in the installed nunchaku** — the precision
   kwarg name, the v2 FP4 module class, and whether a per-tensor global
   scale field is required. Probe before coding (§2).
3. **FP4 weight fragment layout.** Confirm `MmaWeightPackerBase.pack_weight`
   emits the layout the FP4 kernel reads (it may differ from the s4
   `mma.m16n8k64` layout). Validate numerically (Step 6) on one layer
   before packing all 19+38 blocks.
4. **Per-tensor global scale** — the one new tensor NVFP4 needs that INT4
   didn't. Make sure it's both emitted by the packer and consumed by the
   module.
5. **fp16 baseline interpretation** — on 96 GB it's native at all batches;
   update labels/expectations so the speedup isn't misread (§4).
