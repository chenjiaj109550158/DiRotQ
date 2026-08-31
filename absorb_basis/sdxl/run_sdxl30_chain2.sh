#!/usr/bin/env bash
# SDXL-base (30 steps) chain-2: fp16 refs + cov + Algorithm-1 menu
# (6-point lambda grid + S alpha gate) + official MJHQ-2500 head-to-head.
# Requires chain-1 done (svdq dump + kernel). See PLAN_SDXL30.md.
set -uo pipefail
DC=/home/dev/deepcompressor/examples/diffusion
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
M=$REPO/models/sdxl-base
RESULTS=$REPO/absorb_basis/results
BASE_ID=stabilityai/stable-diffusion-xl-base-1.0
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SDXL_MODEL_ID=$BASE_ID SDXL_M_DIR=$M
CACHES=$DC/datasets/torch.float16/sdxl/euler30-g5.0/qdiff/s128/caches
export SDXL_CACHES="$CACHES/*.pt"
count_png() { find "$1" -name '*.png' -size +0 2>/dev/null | wc -l; }
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 50 ]; then echo "SDXLB_CHAIN2_FAILED disk-low ${avail_g}G"; exit 1; fi
}
EMPTY=$M/absorb_basis/empty_kernel.pt
mkdir -p "$M/absorb_basis" "$M/basis"
[ -f "$EMPTY" ] || $PY -c "import torch; torch.save({}, '$EMPTY')"

gen() {  # tag ckpt root bench n
  local tag=$1 ckpt=$2 root=$3 bench=$4 n=$5 sub
  case "$bench" in *yaml) sub="YAML/qdiff-$n";; *) sub="MJHQ/MJHQ-$n";; esac
  local OUT=$DC/runs/$root/samples/$sub
  if [ "$(count_png "$OUT")" -ne "$n" ]; then
    guard
    step gen-$tag
    (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/run_sdxl_kernel_generate.py" \
        configs/model/sdxl.yaml --kernel-weights "$ckpt" \
        --gen-root "runs/$root" --stats-out "$DC/runs/$root/stats_$tag.json" \
        --eval-benchmarks "$bench" --eval-num-samples "$n" \
        --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
    echo "STEP gen-$tag exit $?"
    [ "$(count_png "$OUT")" -eq "$n" ] || { echo "SDXLB_CHAIN2_FAILED gen-$tag"; exit 1; }
  else echo "STEP gen-$tag exit 0 (cached)"; fi
}

# ---- refs ------------------------------------------------------------------
gen qdiff-ref "$EMPTY" sdxlb-qdiff-ref prompts/qdiff.yaml 128

# ---- covariances (full qdiff-128 cache replay) -----------------------------
for part in linear conv; do
  guard
  OUT=$M/basis/absorb_cov_sdxl_$part.pt
  if [ ! -f "$OUT" ]; then
    step cov-$part
    (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/collect_cov_sdxl.py" \
        --model-id "$BASE_ID" --calib-dir "$CACHES" --out "$OUT" --part $part)
    echo "STEP cov-$part exit $?"
    [ -f "$OUT" ] || { echo "SDXLB_CHAIN2_FAILED cov-$part"; exit 1; }
  else echo "STEP cov-$part exit 0 (cached)"; fi
done

# ---- Algorithm 1 stage A: 6-point lambda grid ------------------------------
for d in 0.001 0.003 0.01 0.03 0.1 0.3; do
  guard
  OUT=$M/absorb_basis/sdxlb_damp${d}.pt
  if [ ! -f "$OUT" ]; then
    step build-damp$d
    (cd "$REPO" && $PY absorb_basis/sdxl/build_sdxl_kernel.py \
        --model-id "$BASE_ID" \
        --cov "$M/basis/absorb_cov_sdxl_linear.pt" \
        --cov-conv "$M/basis/absorb_cov_sdxl_conv.pt" \
        --out "$OUT" --hsvd-damping "$d")
    echo "STEP build-damp$d exit $?"
    [ -f "$OUT" ] || { echo "SDXLB_CHAIN2_FAILED build-damp$d"; exit 1; }
  else echo "STEP build-damp$d exit 0 (cached)"; fi
  gen qdiff-damp$d "$OUT" sdxlb-qdiff-damp$d prompts/qdiff.yaml 128
done

step rank-lambda
$PY - <<'PYEOF'
import gc, json, numpy as np, torch
import scipy.linalg as sla
_orig = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig(A); return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid as cf
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
DC = "/home/dev/deepcompressor/examples/diffusion"
ref = f"{DC}/runs/sdxlb-qdiff-ref/samples/YAML/qdiff-128"
model = cf.build_feature_extractor("clean", torch.device("cuda"))
fr = cf.get_folder_features(ref, model=model, num_workers=0, batch_size=64, device=torch.device("cuda"), verbose=False)
mu_r, S_r = fr.mean(0), np.cov(fr, rowvar=False)
def stats(d):
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr","lpips","ssim"), num_workers=0)
    r = {k: float(v) for k, v in m.items()}
    f = cf.get_folder_features(d, model=model, num_workers=0, batch_size=64, device=torch.device("cuda"), verbose=False)
    mu, S = f.mean(0), np.cov(f, rowvar=False)
    cm = _orig(S @ S_r)
    if np.iscomplexobj(cm): cm = cm.real
    r["fid_proxy_128"] = float(((mu-mu_r)**2).sum() + np.trace(S)+np.trace(S_r)-2*np.trace(cm))
    gc.collect(); torch.cuda.empty_cache()
    return r
res = {}
for d in ("0.001", "0.003", "0.01", "0.03", "0.1", "0.3"):
    res[f"damp{d}"] = stats(f"{DC}/runs/sdxlb-qdiff-damp{d}/samples/YAML/qdiff-128")
    print(f"METRICS damp{d}: " + json.dumps(res[f"damp{d}"]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/sdxlb_lambda_qdiff128.json", "w"), indent=2)
def wins(a, b):
    return sum([res[a]["psnr"] > res[b]["psnr"], res[a]["lpips"] < res[b]["lpips"],
                res[a]["ssim"] > res[b]["ssim"], res[a]["fid_proxy_128"] < res[b]["fid_proxy_128"]])
best = max(res, key=lambda k: sum(wins(k, o) for o in res if o != k))
open("/home/dev/DiRotQ/absorb_basis/results/sdxlb_lambda_winner.txt", "w").write(best.replace("damp", ""))
print("SDXLB_LAMBDA_WINNER lambda=" + best.replace("damp", ""))
PYEOF
echo "STEP rank-lambda exit $?"
WIN=$(cat "$RESULTS/sdxlb_lambda_winner.txt")

# ---- Algorithm 1 stage C: S alpha grid (greedy, >=3/4 gate) ----------------
if [ ! -f "$M/absorb_basis/smooth_gain.json" ]; then
  step measure-S
  (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/measure_smooth_gain_sdxl.py" "$WIN")
  echo "STEP measure-S exit $?"
  [ -f "$M/absorb_basis/smooth_gain.json" ] || { echo "SDXLB_CHAIN2_FAILED measure-S"; exit 1; }
else echo "STEP measure-S exit 0 (cached)"; fi

for a in 0.25 0.5 0.75 1.0; do
  guard
  OUT=$M/absorb_basis/sdxlb_damp${WIN}_S${a}.pt
  if [ ! -f "$OUT" ]; then
    step build-S$a
    (cd "$REPO" && $PY absorb_basis/sdxl/build_sdxl_kernel.py \
        --model-id "$BASE_ID" \
        --cov "$M/basis/absorb_cov_sdxl_linear.pt" \
        --cov-conv "$M/basis/absorb_cov_sdxl_conv.pt" \
        --out "$OUT" --hsvd-damping "$WIN" \
        --smooth-pt "$M/svdq_model_dump/smooth.pt" \
        --gains "$M/absorb_basis/smooth_gain.json" --smooth-alpha "$a")
    echo "STEP build-S$a exit $?"
    [ -f "$OUT" ] || { echo "SDXLB_CHAIN2_FAILED build-S$a"; exit 1; }
  else echo "STEP build-S$a exit 0 (cached)"; fi
  gen qdiff-S$a "$OUT" sdxlb-qdiff-S$a prompts/qdiff.yaml 128
done

step rank-S
FINAL=$($PY - "$WIN" <<'PYEOF'
import gc, json, sys, numpy as np, torch
import scipy.linalg as sla
_orig = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig(A); return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid as cf
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
win = sys.argv[1]
DC = "/home/dev/deepcompressor/examples/diffusion"
ref = f"{DC}/runs/sdxlb-qdiff-ref/samples/YAML/qdiff-128"
model = cf.build_feature_extractor("clean", torch.device("cuda"))
fr = cf.get_folder_features(ref, model=model, num_workers=0, batch_size=64, device=torch.device("cuda"), verbose=False)
mu_r, S_r = fr.mean(0), np.cov(fr, rowvar=False)
def stats(d):
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr","lpips","ssim"), num_workers=0)
    r = {k: float(v) for k, v in m.items()}
    f = cf.get_folder_features(d, model=model, num_workers=0, batch_size=64, device=torch.device("cuda"), verbose=False)
    mu, S = f.mean(0), np.cov(f, rowvar=False)
    cm = _orig(S @ S_r)
    if np.iscomplexobj(cm): cm = cm.real
    r["fid_proxy_128"] = float(((mu-mu_r)**2).sum() + np.trace(S)+np.trace(S_r)-2*np.trace(cm))
    gc.collect(); torch.cuda.empty_cache()
    return r
def wins(a, b):
    return sum([a["psnr"] > b["psnr"], a["lpips"] < b["lpips"],
                a["ssim"] > b["ssim"], a["fid_proxy_128"] < b["fid_proxy_128"]])
res = {"base": stats(f"{DC}/runs/sdxlb-qdiff-damp{win}/samples/YAML/qdiff-128")}
cur = "base"
for a in ("0.25", "0.5", "0.75", "1.0"):
    res[f"S@{a}"] = stats(f"{DC}/runs/sdxlb-qdiff-S{a}/samples/YAML/qdiff-128")
    print(f"METRICS S@{a}: " + json.dumps(res[f"S@{a}"]), file=sys.stderr)
    if wins(res[f"S@{a}"], res[cur]) >= 3:
        cur = f"S@{a}"
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/sdxlb_S_qdiff128.json", "w"), indent=2)
print(f"SDXLB_S_GATE winner={cur}", file=sys.stderr)
print(cur)
PYEOF
)
echo "STEP rank-S exit $?"
echo "SDXLB_S_GATE winner=$FINAL"
if [ "$FINAL" = "base" ]; then
  FCKPT=$M/absorb_basis/sdxlb_damp${WIN}.pt; FTAG="damp${WIN}"
else
  A=${FINAL#S@}
  FCKPT=$M/absorb_basis/sdxlb_damp${WIN}_S${A}.pt; FTAG="damp${WIN}+S@${A}"
fi
echo "$FTAG" > "$RESULTS/sdxlb_final_config.txt"
echo "SDXLB_FINAL config=$FTAG"

step backup-postmenu
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup-postmenu exit $?"

# ---- official MJHQ-2500 ----------------------------------------------------
gen ref-mjhq2500 "$EMPTY" sdxlb-ref MJHQ 2500
gen final-absorb "$FCKPT" sdxlb-absorb-final-kernel MJHQ 2500
gen final-svdq "$M/absorb_basis/sdxlb_svdq_kernel.pt" sdxlb-svdq-final-kernel MJHQ 2500

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
ref = f"{DC}/runs/sdxlb-ref/samples/MJHQ/MJHQ-2500"
gt = f"{DC}/benchmarks/MJHQ-GT-2500"
res = {}
for tag, d in [(f"absorb-{ftag}", f"{DC}/runs/sdxlb-absorb-final-kernel/samples/MJHQ/MJHQ-2500"),
               ("svdquant", f"{DC}/runs/sdxlb-svdq-final-kernel/samples/MJHQ/MJHQ-2500")]:
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr","lpips","ssim"), num_workers=0)
    res[tag] = {k: float(v) for k, v in m.items()}
    res[tag]["fid_vs_ref"] = fid.compute_fid(ref, d, verbose=False)
    res[tag]["fid_vs_gt"] = fid.compute_fid(gt, d, verbose=False)
    print(f"FINAL {tag}: " + json.dumps(res[tag]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/sdxlb_final_test2500.json", "w"), indent=2)
PYEOF
echo "STEP final-metrics exit $?"

step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "SDXLB_CHAIN2_DONE"
