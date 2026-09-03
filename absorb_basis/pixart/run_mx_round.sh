#!/usr/bin/env bash
# PLAN_MX morning chain: ours menu -> finals -> svdq convert+gen -> metrics -> backup
set -uo pipefail
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
for stage in menu final svdq metrics; do
  step mx-$stage
  (cd "$REPO" && $PY -u absorb_basis/mx_round_driver.py "$stage")
  rc=$?
  echo "STEP mx-$stage exit $rc"
  [ $rc -eq 0 ] || { echo "MX_ROUND_FAILED $stage"; exit 1; }
done
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "MX_ROUND_DONE"
