#!/usr/bin/env bash
# PLAN A+B on PixArt-Sigma: sequential-calibration build (A) + hsvd-damping
# sweep (B), ranked on MJHQ-500 against the fp16 reference.
#
# Resumable: every step checks for its output first, so re-running after an
# interruption only redoes what's missing.
#
# Usage:  SCRATCH=<dir with ~100GB free> bash absorb_basis/pixart/run_plan_ab.sh
set -uo pipefail

REPO=/home/dev/DiRotQ
DC=/home/dev/deepcompressor/examples/diffusion
PY=/home/dev/.conda/envs/svdquant/bin/python
M=$REPO/models/pixart-sigma/absorb_basis
CACHES=$DC/datasets/torch.float16/pixart-sigma/dpm20-g4.5/qdiff/s128/caches
REF2500=$DC/baselines/torch.float16/pixart-sigma/dpm20-g4.5/samples/MJHQ/MJHQ-2500
BASE2500=$DC/runs/pixart-absorb-hsvd-down/samples/MJHQ/MJHQ-2500
RESULTS=$REPO/absorb_basis/results
SCRATCH=${SCRATCH:-/tmp/pixart_seq_scratch}
export HF_HUB_DISABLE_XET=1

# ---- 1. sequential-calibration build (PLAN A) -------------------------------
if [ ! -f "$M/pixart_absorb_seq.pt" ]; then
  echo "=== STEP build-seq start $(date '+%H:%M:%S') ==="
  mkdir -p "$SCRATCH"
  $PY "$REPO/absorb_basis/pixart/build_pixart_sequential.py" \
      --calib-dir "$CACHES" --out "$M/pixart_absorb_seq.pt" \
      --scratch-dir "$SCRATCH" --chunk 16
  echo "STEP build-seq exit $?"
  [ -f "$M/pixart_absorb_seq.pt" ] || { echo "PIXART_PLAN_AB_FAILED build-seq"; exit 1; }
else
  echo "STEP build-seq exit 0 (cached)"
fi

# ---- 2. MJHQ-500 generation for the four candidates -------------------------
declare -A WEIGHTS=(
  [damp0.003]="$M/pixart_absorb_damp0.003.pt"
  [damp0.03]="$M/pixart_absorb_damp0.03.pt"
  [damp0.1]="$M/pixart_absorb_damp0.1.pt"
  [seq]="$M/pixart_absorb_seq.pt"
)
for tag in damp0.003 damp0.03 damp0.1 seq; do
  out=$DC/runs/pixart-absorb-$tag/samples/MJHQ/MJHQ-500
  cnt=$(find "$out" -name '*.png' 2>/dev/null | wc -l)
  if [ "$cnt" -lt 500 ]; then
    echo "=== STEP gen-$tag start $(date '+%H:%M:%S') ==="
    (cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_sim_generate.py" \
        configs/model/pixart-sigma.yaml \
        --sim-weights "${WEIGHTS[$tag]}" --gen-root "runs/pixart-absorb-$tag" \
        --eval-benchmarks MJHQ --eval-num-samples 500 --eval-num-gpus 1 \
        --eval-batch-size 1 --skip-eval)
    echo "STEP gen-$tag exit $?"
    cnt=$(find "$out" -name '*.png' 2>/dev/null | wc -l)
    [ "$cnt" -eq 500 ] || { echo "PIXART_PLAN_AB_FAILED gen-$tag ($cnt/500)"; exit 1; }
  else
    echo "STEP gen-$tag exit 0 (cached, $cnt imgs)"
  fi
done

# ---- 3. hardlink 500-image subsets of the fp16 ref and the damp0.01 baseline
#         (MJHQ sample sets are nested: the 500 names are a subset of the 2500)
NAMES=$DC/runs/pixart-absorb-damp0.003/samples/MJHQ/MJHQ-500
REF500=$DC/runs/pixart-ref-500sub/MJHQ-500
BASE500=$DC/runs/pixart-absorb-baseline500/MJHQ-500
for pair in "$REF2500:$REF500" "$BASE2500:$BASE500"; do
  src=${pair%%:*}; dst=${pair##*:}
  cnt=$(find "$dst" -name '*.png' 2>/dev/null | wc -l)
  [ "$cnt" -eq 500 ] && { echo "STEP subset-$(basename "$(dirname "$dst")") exit 0 (cached)"; continue; }
  mkdir -p "$dst"
  miss=0
  for f in "$NAMES"/*.png; do
    b=$(basename "$f")
    if [ -f "$src/$b" ]; then ln -f "$src/$b" "$dst/$b"; else miss=$((miss+1)); fi
  done
  [ "$miss" -eq 0 ] || { echo "PIXART_PLAN_AB_FAILED subset $dst missing $miss"; exit 1; }
  echo "STEP subset-$(basename "$(dirname "$dst")") exit 0"
done

# ---- 4. similarity metrics (PSNR/LPIPS/SSIM) vs the fp16 ref-500 subset -----
echo "=== STEP metrics start $(date '+%H:%M:%S') ==="
$PY - "$REF500" "$BASE500" "$DC" "$RESULTS/pixart_plan_ab_mjhq500.json" <<'EOF'
import json, sys
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
ref, base500, dc, out = sys.argv[1:5]
gens = {"baseline-damp0.01": base500}
for tag in ("damp0.003", "damp0.03", "damp0.1", "seq"):
    gens[tag] = f"{dc}/runs/pixart-absorb-{tag}/samples/MJHQ/MJHQ-500"
res = {}
for tag, d in gens.items():
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr", "lpips", "ssim"), num_workers=0)
    res[tag] = {k: float(v) for k, v in m.items()}
    print(f"METRICS {tag}: " + json.dumps(res[tag]))
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print("saved:", out)
EOF
rc=$?
echo "STEP metrics exit $rc"
[ $rc -eq 0 ] || { echo "PIXART_PLAN_AB_FAILED metrics"; exit 1; }
echo "PIXART_PLAN_AB_DONE"
