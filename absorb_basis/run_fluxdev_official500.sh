#!/usr/bin/env bash
# FLUX.1-dev official MJHQ-500 continuation (fresh shell — the original
# chain's bash had buffered the pre-edit MJHQ-1000 section; killed at
# 20:35 and resumed here per the user's 500-sample decision).
# Final config: damp0.3 (results/fluxdev_final_config.txt).
set -uo pipefail
DC=/home/dev/deepcompressor/examples/diffusion
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
MD=$REPO/models/flux-dev
DEV_ID=black-forest-labs/FLUX.1-dev
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 40 ]; then echo "FLUXDEV_FAILED disk-low ${avail_g}G"; exit 1; fi
}
count_png() { find "$1" -name '*.png' -size +0 2>/dev/null | wc -l; }
FTAG=$(cat "$REPO/absorb_basis/results/fluxdev_final_config.txt")
FCKPT=$MD/absorb_basis/fluxdev_damp0.3.safetensors
OFF=$MD/svdq-fp4_r32-flux.1-dev.safetensors
[ "$FTAG" = "damp0.3" ] || { echo "FLUXDEV_FAILED unexpected config $FTAG"; exit 1; }
[ -f "$FCKPT" ] && [ -f "$OFF" ] || { echo "FLUXDEV_FAILED missing ckpt"; exit 1; }

gen() {  # tag weight_arg root n
  local tag=$1 warg=$2 root=$3 n=$4
  local OUT=$DC/runs/$root/samples/MJHQ/MJHQ-$n
  if [ "$(count_png "$OUT")" -ne "$n" ]; then
    guard
    step gen-$tag
    (cd "$DC" && $PY "$REPO/absorb_basis/flux_gen_nunchaku.py" \
        --base-model "$DEV_ID" $warg \
        --num-steps 50 --guidance-scale 3.5 \
        --benchmark MJHQ --num-samples "$n" \
        --out-root "$DC/runs/$root" --stats-out "$DC/runs/$root/stats_$tag.json")
    echo "STEP gen-$tag exit $?"
    [ "$(count_png "$OUT")" -eq "$n" ] || { echo "FLUXDEV_FAILED gen-$tag"; exit 1; }
  else echo "STEP gen-$tag exit 0 (cached)"; fi
}

gen ref-mjhq500 "--bf16-ref" fluxdev-ref 500
gen final-absorb "--weight-path $FCKPT" fluxdev-absorb-final 500
gen final-svdq "--weight-path $OFF" fluxdev-svdq-final 500

step final-metrics
$PY - "$FTAG" <<'PYEOF'
import json, sys
import scipy.linalg as sla
_orig = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig(A); return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
ftag = sys.argv[1]
DC = "/home/dev/deepcompressor/examples/diffusion"
ref = f"{DC}/runs/fluxdev-ref/samples/MJHQ/MJHQ-500"
import os, shutil
gt = f"{DC}/benchmarks/MJHQ-GT-500"
if not os.path.isdir(gt) or len(os.listdir(gt)) != 500:
    os.makedirs(gt, exist_ok=True)
    names = set(os.listdir(ref))
    n = 0
    for f in names:
        src = f"{DC}/benchmarks/MJHQ-GT-2500/{f}"
        if os.path.exists(src):
            shutil.copy2(src, f"{gt}/{f}"); n += 1
    assert n == 500, f"GT-500 build: only {n}/500 matched"
res = {}
for tag, d in [(f"absorb-{ftag}", f"{DC}/runs/fluxdev-absorb-final/samples/MJHQ/MJHQ-500"),
               ("svdquant", f"{DC}/runs/fluxdev-svdq-final/samples/MJHQ/MJHQ-500")]:
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr","lpips","ssim"), num_workers=0)
    res[tag] = {k: float(v) for k, v in m.items()}
    res[tag]["fid_vs_ref"] = fid.compute_fid(ref, d, verbose=False)
    res[tag]["fid_vs_gt"] = fid.compute_fid(gt, d, verbose=False)
    print(f"FINAL {tag}: " + json.dumps(res[tag]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/fluxdev_final_test500.json", "w"), indent=2)
PYEOF
echo "STEP final-metrics exit $?"

step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "FLUXDEV_DONE"
