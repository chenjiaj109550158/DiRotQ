#!/usr/bin/env bash
# PLAN_LAMBDAEXT chain: lambda-grid top-end extension for sdxl-turbo and
# flux-dev. Snapshots the current selfsmooth vectors (they depend on the
# incumbent lambda*) before any regeneration, then runs the extension
# driver per model. Cache-aware; never edit while running.
set -uo pipefail
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }

# snapshot incumbent-lambda vectors (idempotent)
for spec in "sdxl-turbo:0.3" "flux-dev:0.3"; do
  d=$REPO/models/${spec%%:*}/absorb_basis; L=${spec##*:}
  for f in selfsmooth_rms.pt selfsmooth_amax.pt selfsmooth_gain_rms.json selfsmooth_gain_amax.json; do
    [ -f "$d/$f" ] && [ ! -f "$d/${f%.*}_lam$L.${f##*.}" ] && cp -a "$d/$f" "$d/${f%.*}_lam$L.${f##*.}"
  done
done
echo "vector snapshots done"

for m in sdxl fluxdev; do
  step lext-$m
  (cd "$REPO" && $PY -u absorb_basis/lambdaext_driver.py "$m")
  rc=$?
  echo "STEP lext-$m exit $rc"
  [ $rc -eq 0 ] || { echo "LEXT_FAILED $m"; exit 1; }
done
step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "LEXT_ALL_DONE"
