#!/usr/bin/env bash
# Detached SVDQuant SDXL-Turbo calibration + immediate vault backup.
set -uo pipefail
cd /home/dev/deepcompressor/examples/diffusion
export HF_HUB_DISABLE_XET=1
echo "=== CALIB start $(date '+%F %H:%M:%S') ==="
/home/dev/.conda/envs/svdquant/bin/python -m deepcompressor.app.diffusion.ptq \
    configs/model/sdxl-turbo.yaml configs/svdquant/nvfp4.yaml \
    --skip-eval --skip-gen --save-model /home/dev/DiRotQ/models/sdxl-turbo/svdq_model_dump
rc=$?
echo "=== CALIB exit $rc $(date '+%F %H:%M:%S') ==="
if [ $rc -eq 0 ]; then
  bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
  echo "=== POST-CALIB BACKUP exit $? ==="
  echo "SDXL_CALIB_DONE"
else
  echo "SDXL_CALIB_FAILED"
fi
