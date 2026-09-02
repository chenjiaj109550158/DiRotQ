#!/usr/bin/env bash
# PLAN_SELFSMOOTH full chain: six-model self-contained smoothing menu.
# Zero SVDQuant calibration artifacts anywhere in the pipeline (strict
# 2026-09-02 ruling: svdq_model_dump/*, official-checkpoint smooth unpack
# and cov_actq_smooth are all banned; the FLUX --official file is a layout
# container only, verified by the driver's tensor audit).
#
# Stages: (1) regenerate pixart/sana qdiff-128 caches (deleted in disk
# cleanups; deterministic per-prompt seeds), (2) flux-dev act amax over the
# caches_sub strided subset, (3) closed-form s vectors + gain measurement
# for all six models, (4) SANA pilot rms@0.5 vs amax@0.5, (5) family
# decision, (6) six-model alpha-grid rollout with gating + officials on
# config change, (7) backup.
#
# Every step is cache-aware (safe to relaunch after a kill; NEVER edit this
# file while it runs — write a continuation script instead).
set -uo pipefail
REPO=/home/dev/DiRotQ
DC=/home/dev/deepcompressor/examples/diffusion
PY=/home/dev/.conda/envs/svdquant/bin/python
RESULTS=$REPO/absorb_basis/results
DEV_ID=black-forest-labs/FLUX.1-dev
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 45 ]; then echo "SELFSMOOTH_FAILED disk-low ${avail_g}G"; exit 1; fi
}
count_pt() { find "$1" -name '*.pt' -size +0 2>/dev/null | wc -l; }

# ---- 1. regenerate pixart / sana calibration caches ------------------------
for spec in "pixart-sigma" "sana-1.6b"; do
  CD=$REPO/models/$spec/calibration_dataset
  if [ "$(count_pt "$CD/caches")" -lt 5120 ]; then
    guard
    step caches-$spec
    (cd "$REPO" && $PY models/$spec/collect_calibration_dataset.py \
        --prompts models/$spec/calib_prompts.yaml --output "$CD")
    echo "STEP caches-$spec exit $?"
    [ "$(count_pt "$CD/caches")" -ge 5120 ] || { echo "SELFSMOOTH_FAILED caches-$spec"; exit 1; }
  else echo "STEP caches-$spec exit 0 (cached)"; fi
done

# ---- 2. flux-dev act amax (caches_sub strided subset, act-sample protocol) -
AMAX_DEV=$REPO/models/flux-dev/basis/absorb_act_amax.pt
if [ ! -f "$AMAX_DEV" ]; then
  guard
  step amax-fluxdev
  (cd "$REPO" && $PY absorb_basis/collect_act_amax.py --model-id "$DEV_ID" \
      --calib-dir models/flux-dev/calibration_dataset/caches_sub \
      --out "$AMAX_DEV")
  echo "STEP amax-fluxdev exit $?"
  [ -f "$AMAX_DEV" ] || { echo "SELFSMOOTH_FAILED amax-fluxdev"; exit 1; }
else echo "STEP amax-fluxdev exit 0 (cached)"; fi

# ---- 3. closed-form s vectors + gains, all six models ----------------------
vec_done() { [ -f "$1/selfsmooth_rms.pt" ] && [ -f "$1/selfsmooth_amax.pt" ] \
  && [ -f "$1/selfsmooth_gain_rms.json" ] && [ -f "$1/selfsmooth_gain_amax.json" ]; }

if ! vec_done "$REPO/models/flux-schnell/absorb_basis"; then
  guard; step vectors-flux
  (cd "$REPO" && $PY absorb_basis/selfsmooth_vectors_flux.py --lam 0.01 \
      --cov models/flux-schnell/basis/absorb_cov_basis.pt \
      --act-samples models/flux-schnell/basis/absorb_act_samples.pt \
      --act-amax models/flux-schnell/basis/absorb_act_amax.pt \
      --out-dir models/flux-schnell/absorb_basis)
  echo "STEP vectors-flux exit $?"
  vec_done "$REPO/models/flux-schnell/absorb_basis" || { echo "SELFSMOOTH_FAILED vectors-flux"; exit 1; }
else echo "STEP vectors-flux exit 0 (cached)"; fi

if ! vec_done "$REPO/models/flux-dev/absorb_basis"; then
  guard; step vectors-fluxdev
  (cd "$REPO" && $PY absorb_basis/selfsmooth_vectors_flux.py --lam 0.3 \
      --model-id "$DEV_ID" \
      --cov models/flux-dev/basis/absorb_cov_basis.pt \
      --act-samples models/flux-dev/basis/absorb_act_samples.pt \
      --act-amax "$AMAX_DEV" \
      --out-dir models/flux-dev/absorb_basis)
  echo "STEP vectors-fluxdev exit $?"
  vec_done "$REPO/models/flux-dev/absorb_basis" || { echo "SELFSMOOTH_FAILED vectors-fluxdev"; exit 1; }
else echo "STEP vectors-fluxdev exit 0 (cached)"; fi

declare -A HOOK_LAM=( [pixart]=0.1 [sana]=0.3 [sdxl-turbo]=0.3 [sdxl-base]=0.001 )
declare -A HOOK_DIR=( [pixart]=pixart-sigma [sana]=sana-1.6b
                      [sdxl-turbo]=sdxl-turbo [sdxl-base]=sdxl-base )
for f in pixart sana sdxl-turbo sdxl-base; do
  OD=$REPO/models/${HOOK_DIR[$f]}/absorb_basis
  if ! vec_done "$OD"; then
    guard; step vectors-$f
    (cd "$REPO" && $PY absorb_basis/selfsmooth_vectors_hook.py \
        --family "$f" --lam "${HOOK_LAM[$f]}")
    echo "STEP vectors-$f exit $?"
    vec_done "$OD" || { echo "SELFSMOOTH_FAILED vectors-$f"; exit 1; }
  else echo "STEP vectors-$f exit 0 (cached)"; fi
done

# ---- 4. SANA pilot: both families at alpha=0.5 -----------------------------
if [ ! -f "$RESULTS/sana_selfsmooth_pilot.json" ]; then
  guard; step pilot-sana
  (cd "$REPO" && $PY -u absorb_basis/selfsmooth_driver.py sana \
      --families rms,amax --alphas 0.5 --skip-official)
  echo "STEP pilot-sana exit $?"
  [ -f "$RESULTS/sana_selfsmooth_qdiff128.json" ] || { echo "SELFSMOOTH_FAILED pilot"; exit 1; }
  cp "$RESULTS/sana_selfsmooth_qdiff128.json" "$RESULTS/sana_selfsmooth_pilot.json"
else echo "STEP pilot-sana exit 0 (cached)"; fi

# ---- 5. family decision ----------------------------------------------------
step decide-family
$PY - <<'PYEOF'
import json
r = json.load(open("/home/dev/DiRotQ/absorb_basis/results/sana_selfsmooth_pilot.json"))
base, c = r["base"], r["candidates"]
def wins(a, b):
    return sum([a["psnr"] > b["psnr"], a["lpips"] < b["lpips"],
                a["ssim"] > b["ssim"], a["fid_proxy_128"] < b["fid_proxy_128"]])
rm, am = c["S_rms@0.5"], c["S_amax@0.5"]
p_r, p_a = wins(rm, base) >= 3, wins(am, base) >= 3
if p_r and p_a:
    fam = "rms" if wins(rm, am) >= wins(am, rm) else "amax"
elif p_a:
    fam = "amax"
else:
    fam = "rms"  # theory-preferred; per-model gate still protects
open("/home/dev/DiRotQ/absorb_basis/results/selfsmooth_family.txt", "w").write(fam)
print(f"PILOT rms_vs_base={wins(rm, base)} amax_vs_base={wins(am, base)} "
      f"rms_vs_amax={wins(rm, am)} -> family={fam}")
PYEOF
echo "STEP decide-family exit $?"
FAM=$(cat "$RESULTS/selfsmooth_family.txt")
echo "SELFSMOOTH_FAMILY $FAM"

# ---- 6. six-model rollout (full alpha grid + officials on config change) ---
for m in sana pixart sdxl sdxlb flux fluxdev; do
  guard
  step rollout-$m
  (cd "$REPO" && $PY -u absorb_basis/selfsmooth_driver.py "$m" \
      --families "$FAM" --alphas 0.25,0.5,0.75,1.0)
  rc=$?
  echo "STEP rollout-$m exit $rc"
  [ $rc -eq 0 ] || { echo "SELFSMOOTH_FAILED rollout-$m"; exit 1; }
done

# ---- 7. backup -------------------------------------------------------------
step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "SELFSMOOTH_ALL_DONE"
