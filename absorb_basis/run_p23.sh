#!/usr/bin/env bash
# PLAN_NEXTQ P3 (clip re-audit: pixart + schnell) and P2 (FLUX down-proj
# closed-form S). Cache-aware; never edit while running.
set -uo pipefail
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
V=/vault/dirotq-absorb-backup/DiRotQ/models
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
guard() {
  local g; g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  [ "$g" -lt 45 ] && { echo "P23_FAILED disk-low ${g}G"; exit 1; }; return 0
}
npt() { find "$1" -name '*.pt' 2>/dev/null | wc -l; }

# ---- caches restore --------------------------------------------------------
if [ "$(npt $REPO/models/flux-schnell/calibration_dataset/caches)" -lt 512 ]; then
  step restore-schnell-caches
  mkdir -p "$REPO/models/flux-schnell/calibration_dataset"
  cp -au "$V/flux-schnell/calibration_dataset/caches" \
    "$REPO/models/flux-schnell/calibration_dataset/"
  echo "STEP restore-schnell-caches exit $?"
fi
if [ "$(npt $REPO/models/flux-dev/calibration_dataset/caches)" -lt 6400 ]; then
  step restore-dev-caches
  mkdir -p "$REPO/models/flux-dev/calibration_dataset"
  cp -au "$V/flux-dev/calibration_dataset/caches" \
    "$REPO/models/flux-dev/calibration_dataset/"
  echo "STEP restore-dev-caches exit $?"
fi

# ---- P2 prerequisites: down act samples + vectors/gains --------------------
if [ ! -f "$REPO/models/flux-schnell/basis/absorb_act_samples_down.pt" ]; then
  guard; step down-act-schnell
  (cd "$REPO" && $PY absorb_basis/collect_act_samples_down.py \
      --calib-dir models/flux-schnell/calibration_dataset/caches \
      --out models/flux-schnell/basis/absorb_act_samples_down.pt)
  echo "STEP down-act-schnell exit $?"
fi
if [ ! -f "$REPO/models/flux-dev/basis/absorb_act_samples_down.pt" ]; then
  guard; step down-act-dev
  (cd "$REPO" && $PY absorb_basis/collect_act_samples_down.py \
      --model-id black-forest-labs/FLUX.1-dev --stride 5 \
      --calib-dir models/flux-dev/calibration_dataset/caches \
      --out models/flux-dev/basis/absorb_act_samples_down.pt)
  echo "STEP down-act-dev exit $?"
fi
if [ ! -f "$REPO/models/flux-schnell/absorb_basis/selfsmooth_down_rms.pt" ]; then
  guard; step down-vectors-schnell
  (cd "$REPO" && $PY absorb_basis/selfsmooth_down_flux.py --lam 0.01 \
      --cov-down-dir models/flux-schnell/basis/absorb_cov_down \
      --act-samples models/flux-schnell/basis/absorb_act_samples_down.pt \
      --out-dir models/flux-schnell/absorb_basis)
  echo "STEP down-vectors-schnell exit $?"
fi
if [ ! -f "$REPO/models/flux-dev/absorb_basis/selfsmooth_down_rms.pt" ]; then
  guard; step down-vectors-dev
  (cd "$REPO" && $PY absorb_basis/selfsmooth_down_flux.py --lam 0.3 \
      --model-id black-forest-labs/FLUX.1-dev \
      --cov-down-dir models/flux-dev/basis/absorb_cov_down \
      --act-samples models/flux-dev/basis/absorb_act_samples_down.pt \
      --out-dir models/flux-dev/absorb_basis)
  echo "STEP down-vectors-dev exit $?"
fi

# ---- P3 then P2 -------------------------------------------------------------
for job in "p3 pixart" "p3 flux" "p2 flux" "p2 fluxdev"; do
  guard
  step "job-${job// /-}"
  (cd "$REPO" && $PY -u absorb_basis/p23_driver.py $job)
  rc=$?
  echo "STEP job-${job// /-} exit $rc"
  [ $rc -eq 0 ] || { echo "P23_FAILED $job"; exit 1; }
done

step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "P23_ALL_DONE"
