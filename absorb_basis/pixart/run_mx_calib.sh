#!/usr/bin/env bash
# PLAN_MX: SVDQuant full recalibration for MXFP4e2 on PixArt-Sigma
# (their native pipeline incl. per-layer GridSearch smoothing, MX config
# bitwise-verified against our reference quantizers).
set -uo pipefail
cd /home/dev/deepcompressor/examples/diffusion
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
echo "=== MX-CALIB start $(date '+%F %H:%M:%S') ==="
/home/dev/.conda/envs/svdquant/bin/python -m deepcompressor.app.diffusion.ptq \
    configs/model/pixart-sigma.yaml configs/svdquant/mxfp4.yaml \
    --skip-eval --skip-gen --save-model /home/dev/DiRotQ/models/pixart-sigma/svdq_mx_dump
rc=$?
echo "=== MX-CALIB exit $rc $(date '+%F %H:%M:%S') ==="
if [ $rc -eq 0 ]; then
  bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
  echo "MX_CALIB_DONE"
else
  echo "MX_CALIB_FAILED"
fi
