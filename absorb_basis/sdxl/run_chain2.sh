#!/usr/bin/env bash
# SDXL-Turbo chain-2: Algorithm-1 configuration (lambda ranking + S gate)
# then the official MJHQ-2500 head-to-head. Resumable; df guard; backups.
set -uo pipefail
DC=/home/dev/deepcompressor/examples/diffusion
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
M=$REPO/models/sdxl-turbo
RESULTS=$REPO/absorb_basis/results
export HF_HUB_DISABLE_XET=1
count_png() { find "$1" -name '*.png' -size +0 2>/dev/null | wc -l; }
step() { echo "=== STEP $1 start $(date '+%H:%M:%S') ==="; }
guard() {
  local avail_g
  avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$avail_g" -lt 50 ]; then echo "SDXL_CHAIN2_FAILED disk-low ${avail_g}G"; exit 1; fi
}

gen() {  # tag ckpt root bench n
  local tag=$1 ckpt=$2 root=$3 bench=$4 n=$5 sub
  case "$bench" in *yaml) sub="YAML/qdiff-$n";; *) sub="MJHQ/MJHQ-$n";; esac
  local OUT=$DC/runs/$root/samples/$sub
  if [ "$(count_png "$OUT")" -ne "$n" ]; then
    guard
    step gen-$tag
    (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/run_sdxl_kernel_generate.py" \
        configs/model/sdxl-turbo.yaml --kernel-weights "$ckpt" \
        --gen-root "runs/$root" --stats-out "$DC/runs/$root/stats_$tag.json" \
        --eval-benchmarks "$bench" --eval-num-samples "$n" \
        --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
    echo "STEP gen-$tag exit $?"
    [ "$(count_png "$OUT")" -eq "$n" ] || { echo "SDXL_CHAIN2_FAILED gen-$tag"; exit 1; }
  else echo "STEP gen-$tag exit 0 (cached)"; fi
}

# ---- L: lambda candidates on qdiff-128 -------------------------------------
for d in 0.003 0.01 0.1; do
  gen qdiff-damp$d "$M/absorb_basis/sdxl_absorb_damp${d}_kernel.pt" "sdxl-qdiff-damp$d" prompts/qdiff.yaml 128
done

step rank-lambda
$PY - <<'PYEOF'
import json, numpy as np, torch
import scipy.linalg as sla
_orig = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig(A); return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid as cf
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
DC = "/home/dev/deepcompressor/examples/diffusion"
ref = f"{DC}/runs/sdxl-qdiff-ref/samples/YAML/qdiff-128"
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
    return r
res = {}
for d in ("0.003", "0.01", "0.1"):
    res[f"damp{d}"] = stats(f"{DC}/runs/sdxl-qdiff-damp{d}/samples/YAML/qdiff-128")
    print(f"METRICS damp{d}: " + json.dumps(res[f"damp{d}"]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/sdxl_lambda_qdiff128.json", "w"), indent=2)
def wins(a, b):
    return sum([res[a]["psnr"] > res[b]["psnr"], res[a]["lpips"] < res[b]["lpips"],
                res[a]["ssim"] > res[b]["ssim"], res[a]["fid_proxy_128"] < res[b]["fid_proxy_128"]])
best = max(res, key=lambda k: sum(wins(k, o) for o in res if o != k))
open("/home/dev/DiRotQ/absorb_basis/results/sdxl_lambda_winner.txt", "w").write(best.replace("damp", ""))
print("SDXL_LAMBDA_WINNER lambda=" + best.replace("damp", ""))
PYEOF
echo "STEP rank-lambda exit $?"
WIN=$(cat "$RESULTS/sdxl_lambda_winner.txt")

# ---- S: gain measurement + build + gate ------------------------------------
if [ ! -f "$M/absorb_basis/smooth_gain.json" ]; then
  step measure-S
  (cd "$DC" && $PY "$REPO/absorb_basis/sdxl/measure_smooth_gain_sdxl.py" "$WIN")
  echo "STEP measure-S exit $?"
fi
SOUT=$M/absorb_basis/sdxl_r2_S_kernel.pt
if [ ! -f "$SOUT" ]; then
  guard
  step build-S
  (cd "$REPO" && $PY absorb_basis/sdxl/build_sdxl_kernel.py \
      --cov "$M/basis/absorb_cov_sdxl_linear.pt" --cov-conv "$M/basis/absorb_cov_sdxl_conv.pt" \
      --out "$SOUT" --hsvd-damping "$WIN" \
      --smooth-pt "$M/svdq_model_dump/smooth.pt" --gains "$M/absorb_basis/smooth_gain.json")
  echo "STEP build-S exit $?"
fi
gen qdiff-S "$SOUT" "sdxl-qdiff-r2S" prompts/qdiff.yaml 128

step rank-S
FINAL=$($PY - "$WIN" <<'PYEOF'
import json, sys, numpy as np, torch
import scipy.linalg as sla
_orig = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig(A); return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid as cf
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
win = sys.argv[1]
DC = "/home/dev/deepcompressor/examples/diffusion"
ref = f"{DC}/runs/sdxl-qdiff-ref/samples/YAML/qdiff-128"
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
    return r
base = stats(f"{DC}/runs/sdxl-qdiff-damp{win}/samples/YAML/qdiff-128")
sres = stats(f"{DC}/runs/sdxl-qdiff-r2S/samples/YAML/qdiff-128")
print("METRICS base: " + json.dumps(base), file=sys.stderr)
print("METRICS S: " + json.dumps(sres), file=sys.stderr)
json.dump({"base": base, "S": sres}, open("/home/dev/DiRotQ/absorb_basis/results/sdxl_S_qdiff128.json", "w"), indent=2)
w = sum([sres["psnr"] > base["psnr"], sres["lpips"] < base["lpips"],
         sres["ssim"] > base["ssim"], sres["fid_proxy_128"] < base["fid_proxy_128"]])
print("S" if w >= 3 else "base")
PYEOF
)
echo "STEP rank-S exit $?"
echo "SDXL_S_GATE winner=$FINAL"
if [ "$FINAL" = "S" ]; then FCKPT=$SOUT; FTAG="damp${WIN}+S"; else FCKPT=$M/absorb_basis/sdxl_absorb_damp${WIN}_kernel.pt; FTAG="damp${WIN}"; fi
echo "$FTAG" > "$RESULTS/sdxl_final_config.txt"

# ---- T: official MJHQ-2500 --------------------------------------------------
gen final-absorb "$FCKPT" "sdxl-absorb-final-kernel" MJHQ 2500
gen final-svdq "$M/absorb_basis/sdxl_svdq_kernel.pt" "sdxl-svdq-final-kernel" MJHQ 2500

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
ref = f"{DC}/runs/sdxl-ref/samples/MJHQ/MJHQ-2500"
gt = f"{DC}/benchmarks/MJHQ-GT-2500"
res = {}
for tag, d in [(f"absorb-{ftag}", f"{DC}/runs/sdxl-absorb-final-kernel/samples/MJHQ/MJHQ-2500"),
               ("svdquant", f"{DC}/runs/sdxl-svdq-final-kernel/samples/MJHQ/MJHQ-2500")]:
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr","lpips","ssim"), num_workers=0)
    res[tag] = {k: float(v) for k, v in m.items()}
    res[tag]["fid_vs_ref"] = fid.compute_fid(ref, d, verbose=False)
    res[tag]["fid_vs_gt"] = fid.compute_fid(gt, d, verbose=False)
    print(f"FINAL {tag}: " + json.dumps(res[tag]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/sdxl_final_test2500.json", "w"), indent=2)
PYEOF
echo "STEP final-metrics exit $?"

step backup
bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1
echo "STEP backup exit $?"
echo "SDXL_CHAIN2_DONE"
