#!/usr/bin/env bash
# A/B validation with a PROPER protocol: hsvd damping (lambda) is selected on
# a held-out validation set DISJOINT from the test set (PixArt: MJHQ shuffle
# positions 2500..2999; FLUX: 1000..1499), then frozen and evaluated once on
# the official test protocol. Everything runs on the real nunchaku kernel.
#
# Resumable: every step checks its outputs first.
# Usage: bash absorb_basis/run_ab_validation.sh
set -uo pipefail

REPO=/home/dev/DiRotQ
DC=/home/dev/deepcompressor/examples/diffusion
PY=/home/dev/.conda/envs/svdquant/bin/python
M=$REPO/models/pixart-sigma/absorb_basis
MF=$REPO/models/flux-schnell/absorb_basis
RESULTS=$REPO/absorb_basis/results
VAULT_SCRIPTS=/vault/dirotq-absorb-backup/scripts
PREF2500=$DC/baselines/torch.float16/pixart-sigma/dpm20-g4.5/samples/MJHQ/MJHQ-2500
FREF1000=$DC/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1000
FREF1500=$DC/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1500
export HF_HUB_DISABLE_XET=1

step() { echo "=== STEP $1 start $(date '+%H:%M:%S') ==="; }
fin()  { echo "STEP $1 exit $2"; [ "$2" -eq 0 ] || { echo "AB_VALIDATION_FAILED $1"; exit 1; }; }

count_png() { find "$1" -name '*.png' -size +0 2>/dev/null | wc -l; }

# link real files for the given name list from src into dst
link_names() {  # src dst names_json
  $PY - "$1" "$2" "$3" <<'PYEOF'
import json, os, sys
src, dst, names = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
os.makedirs(dst, exist_ok=True)
miss = 0
for n in names:
    s, d = os.path.join(src, n + ".png"), os.path.join(dst, n + ".png")
    if not os.path.exists(s): miss += 1; continue
    if not os.path.exists(d): os.link(s, d)
assert miss == 0, f"{miss} missing in {src}"
PYEOF
}

# create 0-byte placeholders for names present in ref_dir but absent in dst
placeholders() {  # ref_names_dir dst
  $PY - "$1" "$2" <<'PYEOF'
import os, sys
ref, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(ref):
    if f.endswith(".png") and not os.path.exists(os.path.join(dst, f)):
        open(os.path.join(dst, f), "w").close()
PYEOF
}

metrics_json() {  # ref_dir out_json  name1:dir1 name2:dir2 ...
  local ref=$1 out=$2; shift 2
  $PY - "$ref" "$out" "$@" <<'PYEOF'
import json, sys
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics
ref, out = sys.argv[1], sys.argv[2]
res = {}
for pair in sys.argv[3:]:
    tag, d = pair.split(":", 1)
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr", "lpips", "ssim"), num_workers=0)
    res[tag] = {k: float(v) for k, v in m.items()}
    print(f"METRICS {tag}: " + json.dumps(res[tag]))
json.dump(res, open(out, "w"), indent=2)
print("saved:", out)
PYEOF
}

PVAL=$RESULTS/pixart_val500_names.json
FVAL=$RESULTS/flux_val500_names.json
EMPTY=$M/empty_sim.pt
[ -f "$EMPTY" ] || $PY -c "import torch; torch.save({}, '$EMPTY')"

# =====================  PIXART  =============================================
# ---- P1: kernel checkpoints for the candidate lambdas ----------------------
for d in 0.003 0.01; do
  out=$M/pixart_absorb_damp${d}_kernel.pt
  if [ ! -f "$out" ]; then
    step p1-build-kernel-$d
    (cd "$REPO" && $PY absorb_basis/pixart/build_pixart_kernel.py \
        --cov models/pixart-sigma/basis/absorb_cov_pixart.pt \
        --out "$out" --hsvd-damping "$d")
    fin p1-build-kernel-$d $?
  else echo "STEP p1-build-kernel-$d exit 0 (cached)"; fi
done

# ---- P2: fp16 reference for the validation 500 -----------------------------
PREFV=$DC/runs/pixart-ref-val500/MJHQ-500
if [ "$(count_png "$PREFV")" -ne 500 ]; then
  step p2-ref-val
  R3=$DC/runs/pixart-ref-3000/samples/MJHQ/MJHQ-3000
  mkdir -p "$R3"
  (cd "$R3" && for f in "$PREF2500"/*.png; do b=$(basename "$f"); [ -e "$b" ] || ln "$f" "$b"; done)
  (cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_sim_generate.py" \
      configs/model/pixart-sigma.yaml --sim-weights "$EMPTY" \
      --gen-root runs/pixart-ref-3000 --eval-benchmarks MJHQ \
      --eval-num-samples 3000 --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
  rc=$?
  link_names "$R3" "$PREFV" "$PVAL" || rc=1
  [ "$(count_png "$PREFV")" -eq 500 ] || rc=1
  fin p2-ref-val $rc
else echo "STEP p2-ref-val exit 0 (cached)"; fi

# ---- P3: kernel generation of the validation 500 for each lambda -----------
for d in 0.003 0.01 0.1; do
  VD=$DC/runs/pixart-val-damp$d/MJHQ-500
  if [ "$(count_png "$VD")" -ne 500 ]; then
    step p3-val-gen-$d
    G3=$DC/runs/pixart-kval-damp$d/samples/MJHQ/MJHQ-3000
    placeholders "$PREF2500" "$G3"
    (cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_kernel_generate.py" \
        configs/model/pixart-sigma.yaml \
        --kernel-weights "$M/pixart_absorb_damp${d}_kernel.pt" \
        --gen-root "runs/pixart-kval-damp$d" \
        --stats-out "$DC/runs/pixart-kval-damp$d/stats.json" \
        --eval-benchmarks MJHQ --eval-num-samples 3000 --eval-num-gpus 1 \
        --eval-batch-size 1 --skip-eval)
    rc=$?
    find "$G3" -size 0 -delete
    link_names "$G3" "$VD" "$PVAL" || rc=1
    [ "$(count_png "$VD")" -eq 500 ] || rc=1
    fin p3-val-gen-$d $rc
  else echo "STEP p3-val-gen-$d exit 0 (cached)"; fi
done

# ---- P4: validation ranking -> frozen winner --------------------------------
step p4-val-metrics
metrics_json "$PREFV" "$RESULTS/pixart_ab_val500.json" \
    "damp0.003:$DC/runs/pixart-val-damp0.003/MJHQ-500" \
    "damp0.01:$DC/runs/pixart-val-damp0.01/MJHQ-500" \
    "damp0.1:$DC/runs/pixart-val-damp0.1/MJHQ-500"
fin p4-val-metrics $?
PWIN=$($PY - <<'PYEOF'
import json
r = json.load(open("/home/dev/DiRotQ/absorb_basis/results/pixart_ab_val500.json"))
def wins(a, b):
    return sum([r[a]["psnr"] > r[b]["psnr"], r[a]["lpips"] < r[b]["lpips"], r[a]["ssim"] > r[b]["ssim"]])
best = max(r, key=lambda k: sum(wins(k, o) for o in r if o != k))
print(best.replace("damp", ""))
PYEOF
)
echo "PIXART_VAL_WINNER lambda=$PWIN"
echo "$PWIN" > "$RESULTS/pixart_ab_winner.txt"

# ---- P5: official test — frozen winner vs SVDQuant, kernel MJHQ-2500 --------
for pair in "absorb:$M/pixart_absorb_damp${PWIN}_kernel.pt:pixart-absorb-final-kernel" \
            "svdq:$M/pixart_svdq_kernel.pt:pixart-svdq-final-kernel"; do
  tag=${pair%%:*}; rest=${pair#*:}; W=${rest%%:*}; root=${rest##*:}
  OUT=$DC/runs/$root/samples/MJHQ/MJHQ-2500
  if [ "$(count_png "$OUT")" -ne 2500 ]; then
    step p5-test-gen-$tag
    (cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_kernel_generate.py" \
        configs/model/pixart-sigma.yaml --kernel-weights "$W" \
        --gen-root "runs/$root" --stats-out "$DC/runs/$root/stats2500.json" \
        --eval-benchmarks MJHQ --eval-num-samples 2500 --eval-num-gpus 1 \
        --eval-batch-size 1 --skip-eval)
    rc=$?
    [ "$(count_png "$OUT")" -eq 2500 ] || rc=1
    fin p5-test-gen-$tag $rc
  else echo "STEP p5-test-gen-$tag exit 0 (cached)"; fi
done

step p6-test-metrics
metrics_json "$PREF2500" "$RESULTS/pixart_ab_test2500_similarity.json" \
    "absorb-damp$PWIN:$DC/runs/pixart-absorb-final-kernel/samples/MJHQ/MJHQ-2500" \
    "svdquant:$DC/runs/pixart-svdq-final-kernel/samples/MJHQ/MJHQ-2500"
rc=$?
$PY - "$PREF2500" "$DC/benchmarks/MJHQ-GT-2500" "$PWIN" <<'PYEOF' || rc=1
import json, sys
import scipy.linalg as sla
_orig = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig(A)
    return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid
ref, gt, pwin = sys.argv[1], sys.argv[2], sys.argv[3]
DC = "/home/dev/deepcompressor/examples/diffusion"
res = {}
for tag, d in [(f"absorb-damp{pwin}", f"{DC}/runs/pixart-absorb-final-kernel/samples/MJHQ/MJHQ-2500"),
               ("svdquant", f"{DC}/runs/pixart-svdq-final-kernel/samples/MJHQ/MJHQ-2500")]:
    res[tag] = {"fid_vs_ref": fid.compute_fid(ref, d, verbose=False),
                "fid_vs_gt": fid.compute_fid(gt, d, verbose=False)}
    print(f"FID {tag}: " + json.dumps(res[tag]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/pixart_ab_test2500_fid.json", "w"), indent=2)
PYEOF
fin p6-test-metrics $rc

# =====================  FLUX  ===============================================
# ---- F1: bf16 reference for the validation 500 -----------------------------
FREFV=$DC/runs/flux-ref-val500/MJHQ-500
if [ "$(count_png "$FREFV")" -ne 500 ]; then
  step f1-ref-val
  mkdir -p "$FREF1500"
  (cd "$FREF1500" && for f in "$FREF1000"/*.png; do b=$(basename "$f"); [ -e "$b" ] || ln "$f" "$b"; done)
  (cd "$DC" && $PY -u "$VAULT_SCRIPTS/run_bf16_reference.py" configs/model/flux.1-schnell.yaml \
      --output-dirname reference --eval-benchmarks MJHQ --eval-num-samples 1500 \
      --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
  rc=$?
  link_names "$FREF1500" "$FREFV" "$FVAL" || rc=1
  [ "$(count_png "$FREFV")" -eq 500 ] || rc=1
  fin f1-ref-val $rc
else echo "STEP f1-ref-val exit 0 (cached)"; fi

# ---- F2: nunchaku generation of the validation 500 for each lambda ---------
declare -A FW=(
  [0.003]="$MF/dirotq-absorb-hsvd-down-damp0.003-fp4_r32-flux.1-schnell.safetensors"
  [0.01]="$MF/dirotq-absorb-hsvd-down-fp4_r32-flux.1-schnell.safetensors"
  [0.1]="$MF/dirotq-absorb-hsvd-down-damp0.1-fp4_r32-flux.1-schnell.safetensors"
)
declare -A FROOT=(
  [0.003]="$DC/runs/dirotq-absorb-hsvd-down-damp0.003-flux.1-schnell"
  [0.01]="$DC/runs/dirotq-absorb-hsvd-down-flux.1-schnell"
  [0.1]="$DC/runs/dirotq-absorb-hsvd-down-damp0.1-flux.1-schnell"
)
for d in 0.003 0.01 0.1; do
  VD=$DC/runs/flux-val-damp$d/MJHQ-500
  if [ "$(count_png "$VD")" -ne 500 ]; then
    step f2-val-gen-$d
    G15=${FROOT[$d]}/samples/MJHQ/MJHQ-1500
    mkdir -p "$G15"
    for sub in MJHQ-1000 MJHQ-500; do
      S=${FROOT[$d]}/samples/MJHQ/$sub
      [ -d "$S" ] && (cd "$G15" && for f in "$S"/*.png; do b=$(basename "$f"); [ -e "$b" ] || ln "$f" "$b"; done)
    done
    placeholders "$FREF1000" "$G15"
    (cd "$DC" && $PY -u "$VAULT_SCRIPTS/run_nvfp4_nunchaku.py" --num-samples 1500 \
        --weight-path "${FW[$d]}" --out-root "${FROOT[$d]}" \
        --stats-out "${FROOT[$d]}/stats_val.json")
    rc=$?
    find "$G15" -size 0 -delete
    link_names "$G15" "$VD" "$FVAL" || rc=1
    [ "$(count_png "$VD")" -eq 500 ] || rc=1
    fin f2-val-gen-$d $rc
  else echo "STEP f2-val-gen-$d exit 0 (cached)"; fi
done

# ---- F3: validation ranking -> frozen winner --------------------------------
step f3-val-metrics
metrics_json "$FREFV" "$RESULTS/flux_ab_val500.json" \
    "damp0.003:$DC/runs/flux-val-damp0.003/MJHQ-500" \
    "damp0.01:$DC/runs/flux-val-damp0.01/MJHQ-500" \
    "damp0.1:$DC/runs/flux-val-damp0.1/MJHQ-500"
fin f3-val-metrics $?
FWIN=$($PY - <<'PYEOF'
import json
r = json.load(open("/home/dev/DiRotQ/absorb_basis/results/flux_ab_val500.json"))
def wins(a, b):
    return sum([r[a]["psnr"] > r[b]["psnr"], r[a]["lpips"] < r[b]["lpips"], r[a]["ssim"] > r[b]["ssim"]])
best = max(r, key=lambda k: sum(wins(k, o) for o in r if o != k))
print(best.replace("damp", ""))
PYEOF
)
echo "FLUX_VAL_WINNER lambda=$FWIN"
echo "$FWIN" > "$RESULTS/flux_ab_winner.txt"

# ---- F4: official test — frozen winner vs SVDQuant, MJHQ-1000 ---------------
WROOT=${FROOT[$FWIN]}
WOUT=$WROOT/samples/MJHQ/MJHQ-1000
if [ "$(count_png "$WOUT")" -ne 1000 ]; then
  step f4-test-gen
  mkdir -p "$WOUT"
  G15=$WROOT/samples/MJHQ/MJHQ-1500
  [ -d "$G15" ] && (cd "$WOUT" && for f in "$FREF1000"/*.png; do b=$(basename "$f"); [ -e "$b" ] || { [ -f "$G15/$b" ] && ln "$G15/$b" "$b"; }; done)
  (cd "$DC" && $PY -u "$VAULT_SCRIPTS/run_nvfp4_nunchaku.py" --num-samples 1000 \
      --weight-path "${FW[$FWIN]}" --out-root "$WROOT" \
      --stats-out "$WROOT/stats_test.json")
  rc=$?
  [ "$(count_png "$WOUT")" -eq 1000 ] || rc=1
  fin f4-test-gen $rc
else echo "STEP f4-test-gen exit 0 (cached)"; fi

step f5-test-metrics
metrics_json "$FREF1000" "$RESULTS/flux_ab_test1000_similarity.json" \
    "absorb-damp$FWIN:$WOUT" \
    "svdquant:$DC/runs/nvfp4-nunchaku-flux.1-schnell/samples/MJHQ/MJHQ-1000"
rc=$?
$PY - "$FREF1000" "$DC/benchmarks/MJHQ-GT-1000" "$FWIN" "$WOUT" <<'PYEOF' || rc=1
import json, sys
import scipy.linalg as sla
_orig = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig(A)
    return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid
ref, gt, fwin, wout = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
DC = "/home/dev/deepcompressor/examples/diffusion"
res = {}
for tag, d in [(f"absorb-damp{fwin}", wout),
               ("svdquant", f"{DC}/runs/nvfp4-nunchaku-flux.1-schnell/samples/MJHQ/MJHQ-1000")]:
    res[tag] = {"fid_vs_ref": fid.compute_fid(ref, d, verbose=False),
                "fid_vs_gt": fid.compute_fid(gt, d, verbose=False)}
    print(f"FID {tag}: " + json.dumps(res[tag]))
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/flux_ab_test1000_fid.json", "w"), indent=2)
PYEOF
fin f5-test-metrics $rc

echo "AB_VALIDATION_DONE"
