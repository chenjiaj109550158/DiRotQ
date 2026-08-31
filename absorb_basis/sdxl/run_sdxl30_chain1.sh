#!/usr/bin/env bash
# SDXL-base (30 steps, CFG 5.0) chain-1: calib dataset collection + SVDQuant
# NVFP4 calibration + save-model dump + svdq kernel conversion + validation.
# Resumable; disk guard; vault backup at milestones. See PLAN_SDXL30.md.
set -uo pipefail
DC=/home/dev/deepcompressor/examples/diffusion
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
M=$REPO/models/sdxl-base
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SDXL_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
CACHES=$DC/datasets/torch.float16/sdxl/euler30-g5.0/qdiff/s128/caches
export SDXL_CACHES="$CACHES/*.pt"
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 50 ]; then echo "SDXLB_CHAIN1_FAILED disk-low ${avail_g}G"; exit 1; fi
}
mkdir -p "$M/absorb_basis" "$M/basis"

# ---- 1a. calibration dataset (128 prompts x 30 steps x 2 guidances) --------
guard
NCACHE=$(find "$CACHES" -name '*.pt' 2>/dev/null | wc -l)
if [ "$NCACHE" -lt 7680 ]; then
  step collect
  (cd "$DC" && $PY -m deepcompressor.app.diffusion.dataset.collect.calib \
      configs/model/sdxl.yaml configs/collect/qdiff.yaml)
  echo "STEP collect exit $?"
  NCACHE=$(find "$CACHES" -name '*.pt' 2>/dev/null | wc -l)
  [ "$NCACHE" -ge 7680 ] || { echo "SDXLB_CHAIN1_FAILED collect ($NCACHE caches)"; exit 1; }
else echo "STEP collect exit 0 (cached, $NCACHE)"; fi

# ---- 1b. SVDQuant NVFP4 calibration + dump ---------------------------------
guard
if [ ! -f "$M/svdq_model_dump/model.pt" ]; then
  step svdq-calib
  (cd "$DC" && $PY -m deepcompressor.app.diffusion.ptq \
      configs/model/sdxl.yaml configs/svdquant/nvfp4.yaml \
      --skip-eval --skip-gen --save-model "$M/svdq_model_dump")
  echo "STEP svdq-calib exit $?"
  [ -f "$M/svdq_model_dump/model.pt" ] || { echo "SDXLB_CHAIN1_FAILED calib"; exit 1; }
else echo "STEP svdq-calib exit 0 (cached)"; fi
# dump smooth.pt/branch.pt may be relative symlinks into the runs cache:
# materialize them so later steps and vault backups see real files.
for f in smooth.pt branch.pt wgts.pt; do
  p=$M/svdq_model_dump/$f
  if [ -L "$p" ]; then cp --remove-destination -L "$p" "$p.real" && mv "$p.real" "$p"; fi
done

step backup-postcalib
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup-postcalib exit $?"

# ---- 1c. svdq dump -> nunchaku kernel + validation -------------------------
guard
if [ ! -f "$M/absorb_basis/sdxlb_svdq_kernel.pt" ]; then
  step svdq-kernel
  (cd "$REPO" && $PY absorb_basis/sdxl/build_sdxl_kernel_from_svdq.py \
      --dump "$M/svdq_model_dump" --out "$M/absorb_basis/sdxlb_svdq_kernel.pt")
  echo "STEP svdq-kernel exit $?"
  [ -f "$M/absorb_basis/sdxlb_svdq_kernel.pt" ] || { echo "SDXLB_CHAIN1_FAILED svdq-kernel"; exit 1; }
else echo "STEP svdq-kernel exit 0 (cached)"; fi

step validate-svdq
(cd "$REPO" && $PY absorb_basis/sdxl/validate_sdxl_kernel.py \
    "$M/absorb_basis/sdxlb_svdq_kernel.pt")
echo "STEP validate-svdq exit $?"

echo "SDXLB_CHAIN1_DONE"
