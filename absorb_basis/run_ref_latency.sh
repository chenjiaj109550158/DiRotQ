#!/usr/bin/env bash
# fp16/bf16 reference transformer forward-latency benchmarks for
# PixArt-Sigma / SANA-1.6B / FLUX-schnell, using the exact same probes and
# generation protocol as the kernel stats (qdiff-128 prompts, batch 1).
set -uo pipefail
DC=/home/dev/deepcompressor/examples/diffusion
REPO=/home/dev/DiRotQ
PY=/home/dev/.conda/envs/svdquant/bin/python
export HF_HUB_DISABLE_XET=1 TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() { echo "=== STEP $1 start $(date '+%H:%M:%S') ==="; }

step pixart-ref
EMP=$REPO/models/pixart-sigma/absorb_basis/empty_kernel.pt
[ -f "$EMP" ] || $PY -c "import torch; torch.save({}, '$EMP')"
(cd "$DC" && $PY "$REPO/absorb_basis/pixart/run_pixart_kernel_generate.py" \
    configs/model/pixart-sigma.yaml --kernel-weights "$EMP" \
    --gen-root runs/pixart-ref-latency --stats-out "$DC/runs/pixart-ref-latency/stats.json" \
    --eval-benchmarks prompts/qdiff.yaml --eval-num-samples 128 \
    --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
echo "STEP pixart-ref exit $?"

step sana-ref
EMP=$REPO/models/sana-1.6b/absorb_basis/empty_kernel.pt
[ -f "$EMP" ] || $PY -c "import torch; torch.save({}, '$EMP')"
(cd "$DC" && $PY "$REPO/absorb_basis/sana/run_sana_kernel_generate.py" \
    configs/model/sana-1.6b.yaml --kernel-weights "$EMP" \
    --gen-root runs/sana-ref-latency --stats-out "$DC/runs/sana-ref-latency/stats.json" \
    --eval-benchmarks prompts/qdiff.yaml --eval-num-samples 128 \
    --eval-num-gpus 1 --eval-batch-size 1 --skip-eval)
echo "STEP sana-ref exit $?"

step flux-ref
(cd "$DC" && $PY "$REPO/absorb_basis/flux_gen_nunchaku.py" --bf16-ref \
    --benchmark prompts/qdiff.yaml --num-samples 32 --num-steps 4 \
    --guidance-scale 0 --out-root "$DC/runs/flux-ref-latency" \
    --stats-out "$DC/runs/flux-ref-latency/stats.json")
echo "STEP flux-ref exit $?"
echo "REF_LATENCY_DONE"
