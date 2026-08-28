#!/usr/bin/env bash
# Port of the PixArt PLAN-B winner to FLUX.1-schnell: build hsvd+down-absorb
# checkpoints with hsvd damping 0.1 (PixArt winner) and 0.003 (runner-up),
# generate MJHQ-500 with the nunchaku fp4 kernel, and rank them against the
# existing damp-0.01 checkpoint and SVDQuant on the bf16 ref-500 subset.
#
# Resumable: every step checks for its output first.
# Usage:  bash absorb_basis/run_flux_damping.sh
set -uo pipefail

REPO=/home/dev/DiRotQ
DC=/home/dev/deepcompressor/examples/diffusion
PY=/home/dev/.conda/envs/svdquant/bin/python
M=$REPO/models/flux-schnell/absorb_basis
GEN=/vault/dirotq-absorb-backup/scripts/run_nvfp4_nunchaku.py
REF1K=$DC/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1000
BASE1K=$DC/runs/dirotq-absorb-hsvd-down-flux.1-schnell/samples/MJHQ/MJHQ-1000
SVDQ1K=$DC/runs/nvfp4-nunchaku-flux.1-schnell/samples/MJHQ/MJHQ-1000
RESULTS=$REPO/absorb_basis/results
export HF_HUB_DISABLE_XET=1

# ---- 1. build the two damping-variant checkpoints ---------------------------
for d in 0.1 0.003; do
  out=$M/dirotq-absorb-hsvd-down-damp$d-fp4_r32-flux.1-schnell.safetensors
  if [ ! -f "$out" ]; then
    echo "=== STEP build-damp$d start $(date '+%H:%M:%S') ==="
    (cd "$REPO" && $PY absorb_basis/build_checkpoint.py \
        --basis hsvd --down-absorb --hsvd-damping "$d" --out "$out")
    echo "STEP build-damp$d exit $?"
    [ -f "$out" ] || { echo "FLUX_DAMPING_FAILED build-damp$d"; exit 1; }
  else
    echo "STEP build-damp$d exit 0 (cached)"
  fi
done

# ---- 2. MJHQ-500 generation with the nunchaku kernel ------------------------
for d in 0.1 0.003; do
  root=$DC/runs/dirotq-absorb-hsvd-down-damp$d-flux.1-schnell
  out=$root/samples/MJHQ/MJHQ-500
  cnt=$(find "$out" -name '*.png' 2>/dev/null | wc -l)
  if [ "$cnt" -lt 500 ]; then
    echo "=== STEP gen-damp$d start $(date '+%H:%M:%S') ==="
    (cd "$DC" && $PY -u "$GEN" --num-samples 500 \
        --weight-path "$M/dirotq-absorb-hsvd-down-damp$d-fp4_r32-flux.1-schnell.safetensors" \
        --out-root "$root" --stats-out "$root/stats500.json")
    echo "STEP gen-damp$d exit $?"
    cnt=$(find "$out" -name '*.png' 2>/dev/null | wc -l)
    [ "$cnt" -eq 500 ] || { echo "FLUX_DAMPING_FAILED gen-damp$d ($cnt/500)"; exit 1; }
  else
    echo "STEP gen-damp$d exit 0 (cached, $cnt imgs)"
  fi
done

# ---- 3. hardlink 500-image subsets (MJHQ sample sets are nested) ------------
NAMES=$DC/runs/dirotq-absorb-hsvd-down-damp0.1-flux.1-schnell/samples/MJHQ/MJHQ-500
REF500=$DC/runs/flux-ref-500sub/MJHQ-500
BASE500=$DC/runs/flux-absorb-baseline500/MJHQ-500
SVDQ500=$DC/runs/flux-svdq-500sub/MJHQ-500
for pair in "$REF1K:$REF500" "$BASE1K:$BASE500" "$SVDQ1K:$SVDQ500"; do
  src=${pair%%:*}; dst=${pair##*:}
  cnt=$(find "$dst" -name '*.png' 2>/dev/null | wc -l)
  [ "$cnt" -eq 500 ] && { echo "STEP subset-$(basename "$(dirname "$dst")") exit 0 (cached)"; continue; }
  mkdir -p "$dst"
  miss=0
  for f in "$NAMES"/*.png; do
    b=$(basename "$f")
    if [ -f "$src/$b" ]; then ln -f "$src/$b" "$dst/$b"; else miss=$((miss+1)); fi
  done
  [ "$miss" -eq 0 ] || { echo "FLUX_DAMPING_FAILED subset $dst missing $miss"; exit 1; }
  echo "STEP subset-$(basename "$(dirname "$dst")") exit 0"
done

# ---- 4. similarity metrics vs the bf16 ref-500 subset -----------------------
echo "=== STEP metrics start $(date '+%H:%M:%S') ==="
$PY - "$REF500" "$BASE500" "$SVDQ500" "$DC" "$RESULTS/flux_damping_mjhq500.json" <<'EOF'
import json, sys
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
ref, base500, svdq500, dc, out = sys.argv[1:6]
gens = {"svdquant": svdq500, "baseline-damp0.01": base500}
for d in ("0.1", "0.003"):
    gens[f"damp{d}"] = f"{dc}/runs/dirotq-absorb-hsvd-down-damp{d}-flux.1-schnell/samples/MJHQ/MJHQ-500"
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
[ $rc -eq 0 ] || { echo "FLUX_DAMPING_FAILED metrics"; exit 1; }
echo "FLUX_DAMPING_DONE"
