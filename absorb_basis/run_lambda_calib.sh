#!/usr/bin/env bash
# lambda (hsvd damping) selection under SVDQuant's OWN calibration budget:
# the same 128 qdiff prompts (prompts/qdiff.yaml, num_samples=128) their
# calibration uses — no test-domain data, no extra data. The per-model winner
# is frozen, then evaluated once on the official MJHQ protocol (PixArt:
# kernel MJHQ-2500 vs SVDQuant-on-kernel; FLUX: nunchaku MJHQ-1000 vs
# SVDQuant), 5 metrics each.
#
# Resumable: every step checks its outputs first.
# Usage: bash absorb_basis/run_lambda_calib.sh
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
QDIFF=prompts/qdiff.yaml
export HF_HUB_DISABLE_XET=1

step() { echo "=== STEP $1 start $(date '+%H:%M:%S') ==="; }
fin()  { echo "STEP $1 exit $2"; [ "$2" -eq 0 ] || { echo "LAMBDA_CALIB_FAILED $1"; exit 1; }; }
count_png() { find "$1" -name '*.png' -size +0 2>/dev/null | wc -l; }

metrics_json() {  # ref_dir out_json  name1:dir1 ...
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

pick_winner() {  # results_json winner_txt
  $PY - "$1" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
def wins(a, b):
    return sum([r[a]["psnr"] > r[b]["psnr"], r[a]["lpips"] < r[b]["lpips"], r[a]["ssim"] > r[b]["ssim"]])
best = max(r, key=lambda k: sum(wins(k, o) for o in r if o != k))
print(best.replace("damp", ""))
PYEOF
}

EMPTY=$M/empty_sim.pt
[ -f "$EMPTY" ] || $PY -c "import torch; torch.save({}, '$EMPTY')"

# =====================  lambda selection (qdiff-128)  =======================
# ---- L1: PixArt fp16 reference on the 128 calibration prompts --------------
PQREF=$DC/runs/pixart-qdiff-ref/samples/YAML/qdiff-128
if [ "$(count_png "$PQREF")" -ne 128 ]; then
  step l1-pixart-qdiff-ref
  (cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_sim_generate.py" \
      configs/model/pixart-sigma.yaml --sim-weights "$EMPTY" \
      --gen-root runs/pixart-qdiff-ref --eval-benchmarks "$QDIFF" \
      --eval-num-samples 128 --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
  rc=$?
  [ "$(count_png "$PQREF")" -eq 128 ] || rc=1
  fin l1-pixart-qdiff-ref $rc
else echo "STEP l1-pixart-qdiff-ref exit 0 (cached)"; fi

# ---- L2: PixArt kernel candidates on the same 128 prompts -------------------
for d in 0.003 0.01 0.1; do
  OUT=$DC/runs/pixart-qdiff-damp$d/samples/YAML/qdiff-128
  if [ "$(count_png "$OUT")" -ne 128 ]; then
    step l2-pixart-qdiff-$d
    (cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_kernel_generate.py" \
        configs/model/pixart-sigma.yaml \
        --kernel-weights "$M/pixart_absorb_damp${d}_kernel.pt" \
        --gen-root "runs/pixart-qdiff-damp$d" \
        --stats-out "$DC/runs/pixart-qdiff-damp$d/stats.json" \
        --eval-benchmarks "$QDIFF" --eval-num-samples 128 --eval-num-gpus 1 \
        --eval-batch-size 1 --skip-eval)
    rc=$?
    [ "$(count_png "$OUT")" -eq 128 ] || rc=1
    fin l2-pixart-qdiff-$d $rc
  else echo "STEP l2-pixart-qdiff-$d exit 0 (cached)"; fi
done

# ---- L3: PixArt lambda ranking -> frozen winner -----------------------------
step l3-pixart-rank
metrics_json "$PQREF" "$RESULTS/pixart_lambda_qdiff128.json" \
    "damp0.003:$DC/runs/pixart-qdiff-damp0.003/samples/YAML/qdiff-128" \
    "damp0.01:$DC/runs/pixart-qdiff-damp0.01/samples/YAML/qdiff-128" \
    "damp0.1:$DC/runs/pixart-qdiff-damp0.1/samples/YAML/qdiff-128"
fin l3-pixart-rank $?
PWIN=$(pick_winner "$RESULTS/pixart_lambda_qdiff128.json")
echo "PIXART_LAMBDA_WINNER lambda=$PWIN"
echo "$PWIN" > "$RESULTS/pixart_lambda_winner.txt"

# ---- L4: FLUX bf16 reference on the same 128 prompts ------------------------
FQREF=$DC/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/YAML/qdiff-128
if [ "$(count_png "$FQREF")" -ne 128 ]; then
  step l4-flux-qdiff-ref
  (cd "$DC" && $PY -u "$VAULT_SCRIPTS/run_bf16_reference.py" configs/model/flux.1-schnell.yaml \
      --output-dirname reference --eval-benchmarks "$QDIFF" --eval-num-samples 128 \
      --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
  rc=$?
  [ "$(count_png "$FQREF")" -eq 128 ] || rc=1
  fin l4-flux-qdiff-ref $rc
else echo "STEP l4-flux-qdiff-ref exit 0 (cached)"; fi

# ---- L5: FLUX nunchaku candidates on the same 128 prompts -------------------
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
  OUT=${FROOT[$d]}/samples/YAML/qdiff-128
  if [ "$(count_png "$OUT")" -ne 128 ]; then
    step l5-flux-qdiff-$d
    (cd "$DC" && $PY -u "$REPO/absorb_basis/flux_gen_nunchaku.py" --num-samples 128 \
        --benchmark "$QDIFF" --weight-path "${FW[$d]}" --out-root "${FROOT[$d]}" \
        --stats-out "${FROOT[$d]}/stats_qdiff.json")
    rc=$?
    [ "$(count_png "$OUT")" -eq 128 ] || rc=1
    fin l5-flux-qdiff-$d $rc
  else echo "STEP l5-flux-qdiff-$d exit 0 (cached)"; fi
done

# ---- L6: FLUX lambda ranking -> frozen winner -------------------------------
step l6-flux-rank
metrics_json "$FQREF" "$RESULTS/flux_lambda_qdiff128.json" \
    "damp0.003:${FROOT[0.003]}/samples/YAML/qdiff-128" \
    "damp0.01:${FROOT[0.01]}/samples/YAML/qdiff-128" \
    "damp0.1:${FROOT[0.1]}/samples/YAML/qdiff-128"
fin l6-flux-rank $?
FWIN=$(pick_winner "$RESULTS/flux_lambda_qdiff128.json")
echo "FLUX_LAMBDA_WINNER lambda=$FWIN"
echo "$FWIN" > "$RESULTS/flux_lambda_winner.txt"

# =====================  official test (frozen winners)  =====================
# ---- T1: PixArt kernel MJHQ-2500 — winner + SVDQuant ------------------------
for pair in "absorb:$M/pixart_absorb_damp${PWIN}_kernel.pt:pixart-absorb-final-kernel" \
            "svdq:$M/pixart_svdq_kernel.pt:pixart-svdq-final-kernel"; do
  tag=${pair%%:*}; rest=${pair#*:}; W=${rest%%:*}; root=${rest##*:}
  OUT=$DC/runs/$root/samples/MJHQ/MJHQ-2500
  if [ "$(count_png "$OUT")" -ne 2500 ]; then
    step t1-pixart-gen-$tag
    (cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_kernel_generate.py" \
        configs/model/pixart-sigma.yaml --kernel-weights "$W" \
        --gen-root "runs/$root" --stats-out "$DC/runs/$root/stats2500.json" \
        --eval-benchmarks MJHQ --eval-num-samples 2500 --eval-num-gpus 1 \
        --eval-batch-size 1 --skip-eval)
    rc=$?
    [ "$(count_png "$OUT")" -eq 2500 ] || rc=1
    fin t1-pixart-gen-$tag $rc
  else echo "STEP t1-pixart-gen-$tag exit 0 (cached)"; fi
done

step t2-pixart-metrics
metrics_json "$PREF2500" "$RESULTS/pixart_final_test2500_similarity.json" \
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
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/pixart_final_test2500_fid.json", "w"), indent=2)
PYEOF
fin t2-pixart-metrics $rc

# ---- T3: FLUX winner MJHQ-1000 ---------------------------------------------
WROOT=${FROOT[$FWIN]}
WOUT=$WROOT/samples/MJHQ/MJHQ-1000
if [ "$(count_png "$WOUT")" -ne 1000 ]; then
  step t3-flux-gen
  mkdir -p "$WOUT"
  S5=$WROOT/samples/MJHQ/MJHQ-500
  [ -d "$S5" ] && (cd "$WOUT" && for f in "$S5"/*.png; do b=$(basename "$f"); [ -e "$b" ] || ln "$f" "$b"; done)
  (cd "$DC" && $PY -u "$REPO/absorb_basis/flux_gen_nunchaku.py" --num-samples 1000 \
      --weight-path "${FW[$FWIN]}" --out-root "$WROOT" \
      --stats-out "$WROOT/stats_test.json")
  rc=$?
  [ "$(count_png "$WOUT")" -eq 1000 ] || rc=1
  fin t3-flux-gen $rc
else echo "STEP t3-flux-gen exit 0 (cached)"; fi

step t4-flux-metrics
metrics_json "$FREF1000" "$RESULTS/flux_final_test1000_similarity.json" \
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
json.dump(res, open("/home/dev/DiRotQ/absorb_basis/results/flux_final_test1000_fid.json", "w"), indent=2)
PYEOF
fin t4-flux-metrics $rc

echo "LAMBDA_CALIB_DONE"
