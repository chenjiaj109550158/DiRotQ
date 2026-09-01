#!/usr/bin/env bash
# FLUX.1-dev (50 steps, guidance 3.5) full chain: calib collection + cov +
# Algorithm-1 menu (6 lambda + S alpha gate) + official MJHQ-1000 vs the
# official nunchaku FLUX.1-dev NVFP4 checkpoint. See PLAN_FLUXDEV.md.
set -uo pipefail
DC=/home/dev/deepcompressor/examples/diffusion
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
MD=$REPO/models/flux-dev
RESULTS=$REPO/absorb_basis/results
DEV_ID=black-forest-labs/FLUX.1-dev
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHES=$MD/calibration_dataset/caches
SUB=$MD/calibration_dataset/caches_sub
step() { echo "=== STEP $1 start $(date '+%F %H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 40 ]; then echo "FLUXDEV_FAILED disk-low ${avail_g}G"; exit 1; fi
}
count_png() { find "$1" -name '*.png' -size +0 2>/dev/null | wc -l; }
mkdir -p "$MD/absorb_basis" "$MD/basis"

# ---- 1. calibration caches (128 prompts x 50 steps x 1 guidance) -----------
guard
N=$(find "$CACHES" -name '*.pt' 2>/dev/null | wc -l)
if [ "$N" -lt 6400 ]; then
  step collect
  (cd "$REPO" && $PY models/flux-schnell/collect_calibration_dataset.py \
      --model-id "$DEV_ID" --prompts models/flux-schnell/calib_prompts.yaml \
      --output "$MD/calibration_dataset" --num-samples 128 \
      --num-steps 50 --guidance-scale 3.5 --cpu-offload)
  echo "STEP collect exit $?"
  N=$(find "$CACHES" -name '*.pt' 2>/dev/null | wc -l)
  [ "$N" -ge 6400 ] || { echo "FLUXDEV_FAILED collect ($N caches)"; exit 1; }
else echo "STEP collect exit 0 (cached, $N)"; fi
# strided 1/5 subset for the 12288-dim down covs + act samples
if [ "$(find "$SUB" -name '*.pt' 2>/dev/null | wc -l)" -lt 1200 ]; then
  step subset
  mkdir -p "$SUB"
  i=0; for f in $(find "$CACHES" -name '*.pt' | sort); do
    [ $((i % 5)) -eq 0 ] && ln -sf "$f" "$SUB/$(basename "$f")"
    i=$((i+1))
  done
  echo "STEP subset exit 0 ($(ls "$SUB" | wc -l))"
fi

# ---- 2. official svdq dev checkpoint ---------------------------------------
OFF=$MD/svdq-fp4_r32-flux.1-dev.safetensors
if [ ! -f "$OFF" ]; then
  step svdq-download
  P=$($PY -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('mit-han-lab/nunchaku-flux.1-dev','svdq-fp4_r32-flux.1-dev.safetensors'))")
  cp -L "$P" "$OFF"
  echo "STEP svdq-download exit $?"
fi

# ---- 3. covariances + act samples ------------------------------------------
guard
if [ ! -f "$MD/basis/absorb_cov_basis.pt" ]; then
  step cov-main
  (cd "$REPO" && $PY absorb_basis/collect_cov.py --model-id "$DEV_ID" \
      --calib-dir "$CACHES" --out "$MD/basis/absorb_cov_basis.pt" --batch-size 2)
  echo "STEP cov-main exit $?"
  [ -f "$MD/basis/absorb_cov_basis.pt" ] || { echo "FLUXDEV_FAILED cov-main"; exit 1; }
else echo "STEP cov-main exit 0 (cached)"; fi
if [ "$(ls "$MD/basis/absorb_cov_down" 2>/dev/null | wc -l)" -lt 76 ]; then
  step cov-down
  (cd "$REPO" && $PY absorb_basis/collect_cov_down.py --model-id "$DEV_ID" \
      --calib-dir "$SUB" --out-dir "$MD/basis/absorb_cov_down" --batch-size 2)
  echo "STEP cov-down exit $?"
  [ "$(ls "$MD/basis/absorb_cov_down" | wc -l)" -ge 76 ] || { echo "FLUXDEV_FAILED cov-down"; exit 1; }
else echo "STEP cov-down exit 0 (cached)"; fi
if [ ! -f "$MD/basis/absorb_act_samples.pt" ]; then
  step act-samples
  (cd "$REPO" && $PY absorb_basis/collect_act_samples.py --model-id "$DEV_ID" \
      --calib-dir "$SUB" --out "$MD/basis/absorb_act_samples.pt" --batch-size 2)
  echo "STEP act-samples exit $?"
else echo "STEP act-samples exit 0 (cached)"; fi

step backup-precalc
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup-precalc exit $?"

# ---- 4. gen helper ---------------------------------------------------------
gen() {  # tag weight_arg root bench n  (weight_arg: --bf16-ref | --weight-path X | --weight-repo ...)
  local tag=$1 warg=$2 root=$3 bench=$4 n=$5 sub
  case "$bench" in *yaml) sub="YAML/qdiff-$n";; *) sub="MJHQ/MJHQ-$n";; esac
  local OUT=$DC/runs/$root/samples/$sub
  if [ "$(count_png "$OUT")" -ne "$n" ]; then
    guard
    step gen-$tag
    (cd "$DC" && $PY "$REPO/absorb_basis/flux_gen_nunchaku.py" \
        --base-model "$DEV_ID" $warg \
        --num-steps 50 --guidance-scale 3.5 \
        --benchmark "$bench" --num-samples "$n" \
        --out-root "$DC/runs/$root" --stats-out "$DC/runs/$root/stats_$tag.json")
    echo "STEP gen-$tag exit $?"
    [ "$(count_png "$OUT")" -eq "$n" ] || { echo "FLUXDEV_FAILED gen-$tag"; exit 1; }
  else echo "STEP gen-$tag exit 0 (cached)"; fi
}

gen qdiff-ref "--bf16-ref" fluxdev-qdiff-ref prompts/qdiff.yaml 128

# ---- 5. Algorithm 1 stage A: 6-point lambda grid ---------------------------
for d in 0.001 0.003 0.01 0.03 0.1 0.3; do
  guard
  OUT=$MD/absorb_basis/fluxdev_damp${d}.safetensors
  if [ ! -f "$OUT" ]; then
    step build-damp$d
    (cd "$REPO" && $PY absorb_basis/build_checkpoint.py \
        --model-id "$DEV_ID" --official "$OFF" \
        --cov "$MD/basis/absorb_cov_basis.pt" \
        --cov-down-dir "$MD/basis/absorb_cov_down" \
        --basis hsvd --down-absorb --hsvd-damping "$d" --out "$OUT")
    echo "STEP build-damp$d exit $?"
    [ -f "$OUT" ] || { echo "FLUXDEV_FAILED build-damp$d"; exit 1; }
  else echo "STEP build-damp$d exit 0 (cached)"; fi
  gen qdiff-damp$d "--weight-path $OUT" fluxdev-qdiff-damp$d prompts/qdiff.yaml 128
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
ref = f"{DC}/runs/fluxdev-qdiff-ref/samples/YAML/qdiff-128"
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
    res[f"damp{d}"] = stats(f"{DC}/runs/fluxdev-qdiff-damp{d}/samples/YAML/qdiff-128")
    print(f"METRICS damp{d}: " + json.dumps(res[f"damp{d}"]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/fluxdev_lambda_qdiff128.json", "w"), indent=2)
def wins(a, b):
    return sum([res[a]["psnr"] > res[b]["psnr"], res[a]["lpips"] < res[b]["lpips"],
                res[a]["ssim"] > res[b]["ssim"], res[a]["fid_proxy_128"] < res[b]["fid_proxy_128"]])
best = max(res, key=lambda k: sum(wins(k, o) for o in res if o != k))
open("/home/dev/DiRotQ/absorb_basis/results/fluxdev_lambda_winner.txt", "w").write(best.replace("damp", ""))
print("FLUXDEV_LAMBDA_WINNER lambda=" + best.replace("damp", ""))
PYEOF
echo "STEP rank-lambda exit $?"
WIN=$(cat "$RESULTS/fluxdev_lambda_winner.txt")
# free ~35G: drop losing lambda checkpoints (rebuildable from cov)
for d in 0.001 0.003 0.01 0.03 0.1 0.3; do
  [ "$d" = "$WIN" ] || rm -f "$MD/absorb_basis/fluxdev_damp${d}.safetensors"
done

# ---- 6. Algorithm 1 stage C: S alpha grid ----------------------------------
if [ ! -f "$MD/absorb_basis/smooth_gain.json" ]; then
  step measure-S
  (cd "$REPO" && $PY absorb_basis/measure_smooth_gain_flux.py --lam "$WIN" \
      --model-id "$DEV_ID" --official "$OFF" \
      --cov "$MD/basis/absorb_cov_basis.pt" \
      --act-samples "$MD/basis/absorb_act_samples.pt" \
      --out "$MD/absorb_basis/smooth_gain.json")
  echo "STEP measure-S exit $?"
  [ -f "$MD/absorb_basis/smooth_gain.json" ] || { echo "FLUXDEV_FAILED measure-S"; exit 1; }
else echo "STEP measure-S exit 0 (cached)"; fi

for a in 0.25 0.5 0.75 1.0; do
  guard
  OUT=$MD/absorb_basis/fluxdev_damp${WIN}_S${a}.safetensors
  if [ ! -f "$OUT" ]; then
    step build-S$a
    (cd "$REPO" && $PY absorb_basis/build_checkpoint.py \
        --model-id "$DEV_ID" --official "$OFF" \
        --cov "$MD/basis/absorb_cov_basis.pt" \
        --cov-down-dir "$MD/basis/absorb_cov_down" \
        --basis hsvd --down-absorb --hsvd-damping "$WIN" --out "$OUT" \
        --select-smooth-gains "$MD/absorb_basis/smooth_gain.json" \
        --select-smooth-alpha "$a")
    echo "STEP build-S$a exit $?"
    [ -f "$OUT" ] || { echo "FLUXDEV_FAILED build-S$a"; exit 1; }
  else echo "STEP build-S$a exit 0 (cached)"; fi
  gen qdiff-S$a "--weight-path $OUT" fluxdev-qdiff-S$a prompts/qdiff.yaml 128
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
ref = f"{DC}/runs/fluxdev-qdiff-ref/samples/YAML/qdiff-128"
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
res = {"base": stats(f"{DC}/runs/fluxdev-qdiff-damp{win}/samples/YAML/qdiff-128")}
cur = "base"
for a in ("0.25", "0.5", "0.75", "1.0"):
    res[f"S@{a}"] = stats(f"{DC}/runs/fluxdev-qdiff-S{a}/samples/YAML/qdiff-128")
    print(f"METRICS S@{a}: " + json.dumps(res[f"S@{a}"]), file=sys.stderr)
    if wins(res[f"S@{a}"], res[cur]) >= 3:
        cur = f"S@{a}"
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/fluxdev_S_qdiff128.json", "w"), indent=2)
print(f"FLUXDEV_S_GATE winner={cur}", file=sys.stderr)
print(cur)
PYEOF
)
echo "STEP rank-S exit $?"
echo "FLUXDEV_S_GATE winner=$FINAL"
if [ "$FINAL" = "base" ]; then
  FCKPT=$MD/absorb_basis/fluxdev_damp${WIN}.safetensors; FTAG="damp${WIN}"
else
  A=${FINAL#S@}
  FCKPT=$MD/absorb_basis/fluxdev_damp${WIN}_S${A}.safetensors; FTAG="damp${WIN}+S@${A}"
fi
echo "$FTAG" > "$RESULTS/fluxdev_final_config.txt"
echo "FLUXDEV_FINAL config=$FTAG"
# free the losing alpha checkpoints
for a in 0.25 0.5 0.75 1.0; do
  C=$MD/absorb_basis/fluxdev_damp${WIN}_S${a}.safetensors
  [ "$C" = "$FCKPT" ] || rm -f "$C"
done

step backup-postmenu
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup-postmenu exit $?"

# ---- 7. official MJHQ-500 (user decision 2026-09-01: 500 first) ------------
gen ref-mjhq500 "--bf16-ref" fluxdev-ref MJHQ 500
gen final-absorb "--weight-path $FCKPT" fluxdev-absorb-final MJHQ 500
gen final-svdq "--weight-path $OFF" fluxdev-svdq-final MJHQ 500

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
# GT-500: match the generated filenames against the GT-2500 superset
# (MJHQ sampling is subset-consistent: GT-1000 is a subset of GT-2500)
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
