#!/usr/bin/env bash
# SDXL-Turbo chain-1: fp16 refs + our covariances + lambda builds.
# Resumable; df guard before每個大步驟; vault backup at completion.
set -uo pipefail
DC=/home/dev/deepcompressor/examples/diffusion
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
M=$REPO/models/sdxl-turbo
export HF_HUB_DISABLE_XET=1
count_png() { find "$1" -name '*.png' -size +0 2>/dev/null | wc -l; }
step() { echo "=== STEP $1 start $(date '+%H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 50 ]; then echo "SDXL_CHAIN1_FAILED disk-low ${avail_g}G"; exit 1; fi
}
CACHES=$DC/datasets/torch.float16/sdxl-turbo/eulera4-g0/qdiff/s128/caches
EMPTY=$M/absorb_basis/empty_kernel.pt
mkdir -p "$M/absorb_basis" "$M/basis"
[ -f "$EMPTY" ] || $PY -c "import torch; torch.save({}, '$EMPTY')"

guard
REFQ=$DC/runs/sdxl-qdiff-ref/samples/YAML/qdiff-128
if [ "$(count_png "$REFQ")" -ne 128 ]; then
  step ref-qdiff128
  (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/run_sdxl_kernel_generate.py" \
      configs/model/sdxl-turbo.yaml --kernel-weights "$EMPTY" \
      --gen-root runs/sdxl-qdiff-ref --stats-out "$DC/runs/sdxl-qdiff-ref/stats.json" \
      --eval-benchmarks prompts/qdiff.yaml --eval-num-samples 128 \
      --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
  echo "STEP ref-qdiff128 exit $?"
else echo "STEP ref-qdiff128 exit 0 (cached)"; fi

for part in linear conv; do
  guard
  OUT=$M/basis/absorb_cov_sdxl_$part.pt
  if [ ! -f "$OUT" ]; then
    step cov-$part
    (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/collect_cov_sdxl.py" \
        --calib-dir "$CACHES" --out "$OUT" --part $part)
    echo "STEP cov-$part exit $?"
    [ -f "$OUT" ] || { echo "SDXL_CHAIN1_FAILED cov-$part"; exit 1; }
  else echo "STEP cov-$part exit 0 (cached)"; fi
done

for d in 0.003 0.01 0.1; do
  guard
  OUT=$M/absorb_basis/sdxl_absorb_damp${d}_kernel.pt
  if [ ! -f "$OUT" ]; then
    step build-damp$d
    (cd "$REPO" && $PY absorb_basis/sdxl/build_sdxl_kernel.py \
        --cov "$M/basis/absorb_cov_sdxl_linear.pt" \
        --cov-conv "$M/basis/absorb_cov_sdxl_conv.pt" \
        --out "$OUT" --hsvd-damping "$d")
    echo "STEP build-damp$d exit $?"
    [ -f "$OUT" ] || { echo "SDXL_CHAIN1_FAILED build-damp$d"; exit 1; }
  else echo "STEP build-damp$d exit 0 (cached)"; fi
done

guard
REF25=$DC/runs/sdxl-ref/samples/MJHQ/MJHQ-2500
if [ "$(count_png "$REF25")" -ne 2500 ]; then
  step ref-mjhq2500
  (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/run_sdxl_kernel_generate.py" \
      configs/model/sdxl-turbo.yaml --kernel-weights "$EMPTY" \
      --gen-root runs/sdxl-ref --stats-out "$DC/runs/sdxl-ref/stats.json" \
      --eval-benchmarks MJHQ --eval-num-samples 2500 --eval-num-gpus 1 \
      --eval-batch-size 1 --skip-eval)
  echo "STEP ref-mjhq2500 exit $?"
else echo "STEP ref-mjhq2500 exit 0 (cached)"; fi

step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "SDXL_CHAIN1_DONE"
