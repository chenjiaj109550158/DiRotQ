#!/usr/bin/env bash
set -uo pipefail
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
cd /home/dev/DiRotQ
for m in pixart sdxl sana flux; do
  /home/dev/.conda/envs/svdquant/bin/python -u absorb_basis/round3_driver.py $m
  echo "MODEL $m exit $?"
done
echo "ROUND3_ALL_DONE"
