#!/usr/bin/env bash
# PLAN_SELFSMOOTH continuation (fresh shell after the container-audit stop):
# the audit caught the 76 adaLN modulation linears still carrying SVDQuant's
# W4A16 tensors. build_checkpoint now requantizes them data-free by default,
# the flux/fluxdev menus rebuild their no-S base with fresh qdiff images, and
# both flux models always rerun officials (adanorm bits changed).
# Remaining stages: rollout-flux, rollout-fluxdev, backup.
set -uo pipefail
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
RESULTS=$REPO/absorb_basis/results
FAM=$(cat "$RESULTS/selfsmooth_family.txt")
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 45 ]; then echo "SELFSMOOTH_FAILED disk-low ${avail_g}G"; exit 1; fi
}
echo "SELFSMOOTH_FAMILY $FAM (continuation)"
for m in flux fluxdev; do
  guard
  step rollout-$m
  (cd "$REPO" && $PY -u absorb_basis/selfsmooth_driver.py "$m" \
      --families "$FAM" --alphas 0.25,0.5,0.75,1.0)
  rc=$?
  echo "STEP rollout-$m exit $rc"
  [ $rc -eq 0 ] || { echo "SELFSMOOTH_FAILED rollout-$m"; exit 1; }
done
step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "SELFSMOOTH_ALL_DONE"
