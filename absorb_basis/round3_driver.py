"""PLAN_ROUND3 driver: per-model greedy staged selection with Algorithm-1
gating on qdiff-128.

Stage A: lambda 6-point grid -> lambda*
Stage B: + per-channel-top (pixart/sana/sdxl) -> gate
Stage C: + selective smoothing with strength alpha in {0.25,0.5,0.75,1.0} -> gate
Then: if the final config differs from the current official config, rerun the
official benchmark (kernel gen + 5 metrics vs SVDQuant).

Run in the svdquant env from the repo root:
  python absorb_basis/round3_driver.py pixart|sdxl|sana|flux
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/dev/DiRotQ")

REPO = "/home/dev/DiRotQ"
DC = "/home/dev/deepcompressor/examples/diffusion"
PY = "/home/dev/.conda/envs/svdquant/bin/python"
VAULT = "/vault/dirotq-absorb-backup"
RESULTS = f"{REPO}/absorb_basis/results"
LAMBDAS_OLD = ["0.003", "0.01", "0.1"]
LAMBDAS_NEW = ["0.001", "0.03", "0.3"]
ALPHAS = ["0.25", "0.5", "0.75", "1.0"]

import scipy.linalg as sla
_orig_sqrtm = sla.sqrtm
def _compat(A, disp=None, **kw):
    r = _orig_sqrtm(A)
    return (r, 0.0) if disp is False else r
sla.sqrtm = _compat
from cleanfid import fid as cf
from deepcompressor.app.diffusion.eval.metrics.similarity import compute_image_similarity_metrics


def step(name):
    print(f"=== STEP {name} start {time.strftime('%H:%M:%S')} ===", flush=True)


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    return r.returncode


def count_png(d):
    n = 0
    for root, _, files in os.walk(d):
        n += sum(1 for f in files if f.endswith(".png") and
                 os.path.getsize(os.path.join(root, f)) > 0)
    return n


def fetch(src, dst, deref=False):
    # always cp -au (idempotent: only copies missing/newer files)
    os.makedirs(os.path.dirname(dst.rstrip("/")), exist_ok=True)
    flag = "-Lru" if deref else "-au"
    rc = sh(f"cp {flag} '{src}/.' '{dst}/'" if os.path.isdir(src)
            else f"cp {flag} '{src}' '{dst}'")
    print(f"fetched {src} -> {dst} rc={rc}", flush=True)


def disk_guard():
    st = os.statvfs("/")
    avail_g = st.f_bavail * st.f_frsize / 2**30
    assert avail_g > 50, f"disk low: {avail_g:.0f}G"


_FEAT_MODEL = None
_REF_CACHE = {}


def feat_model():
    global _FEAT_MODEL
    if _FEAT_MODEL is None:
        _FEAT_MODEL = cf.build_feature_extractor("clean", torch.device("cuda"))
    return _FEAT_MODEL


def stats_vs_ref(ref, d):
    m = compute_image_similarity_metrics(ref, d, metrics=("psnr", "lpips", "ssim"),
                                         num_workers=0)
    r = {k: float(v) for k, v in m.items()}
    if ref not in _REF_CACHE:
        f = cf.get_folder_features(ref, model=feat_model(), num_workers=0,
                                   batch_size=64, device=torch.device("cuda"), verbose=False)
        _REF_CACHE[ref] = (f.mean(0), np.cov(f, rowvar=False))
    mu_r, S_r = _REF_CACHE[ref]
    f = cf.get_folder_features(d, model=feat_model(), num_workers=0,
                               batch_size=64, device=torch.device("cuda"), verbose=False)
    mu, S = f.mean(0), np.cov(f, rowvar=False)
    cm = _orig_sqrtm(S @ S_r)
    if np.iscomplexobj(cm):
        cm = cm.real
    r["fid_proxy_128"] = float(((mu - mu_r) ** 2).sum() + np.trace(S) + np.trace(S_r)
                               - 2 * np.trace(cm))
    # metric internals accumulate GiBs in the caching allocator across calls,
    # starving the gen subprocesses -> release after every ranking call
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return r


def wins(a, b):
    return sum([a["psnr"] > b["psnr"], a["lpips"] < b["lpips"],
                a["ssim"] > b["ssim"], a["fid_proxy_128"] < b["fid_proxy_128"]])


M = sys.argv[1]
CFG = {
    "pixart": dict(
        ck=lambda tag: f"{REPO}/models/pixart-sigma/absorb_basis/pixart_r3_{tag}.pt",
        old_ck=lambda lam: f"{REPO}/models/pixart-sigma/absorb_basis/pixart_absorb_damp{lam}_kernel.pt",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/pixart/build_pixart_kernel.py "
            f"--cov models/pixart-sigma/basis/absorb_cov_pixart.pt "
            f"--out {out} --hsvd-damping {lam} {extra}"),
        smooth=f"{REPO}/models/pixart-sigma/svdq_model_dump/smooth.pt",
        gains=f"{REPO}/models/pixart-sigma/absorb_basis/smooth_gain.json",
        runner=f"{REPO}/absorb_basis/pixart/run_pixart_kernel_generate.py",
        yaml="configs/model/pixart-sigma.yaml",
        qref=f"{DC}/runs/pixart-qdiff-ref/samples/YAML/qdiff-128",
        old_qdiff=lambda lam: f"{DC}/runs/pixart-qdiff-damp{lam}/samples/YAML/qdiff-128",
        current_final="damp0.1+S@1.0",
        official=dict(n=2500, bench="MJHQ",
                      ref=f"{DC}/baselines/torch.float16/pixart-sigma/dpm20-g4.5/samples/MJHQ/MJHQ-2500",
                      gt=f"{DC}/benchmarks/MJHQ-GT-2500",
                      svdq=f"{DC}/runs/pixart-svdq-final-kernel/samples/MJHQ/MJHQ-2500"),
        fetch=[
            (f"{VAULT}/DiRotQ/models/pixart-sigma/basis", f"{REPO}/models/pixart-sigma/basis", False),
            (f"{VAULT}/DiRotQ/models/pixart-sigma/absorb_basis", f"{REPO}/models/pixart-sigma/absorb_basis", False),
            (f"{VAULT}/DiRotQ/models/pixart-sigma/svdq_model_dump", f"{REPO}/models/pixart-sigma/svdq_model_dump", True),
            (f"{VAULT}/deepcompressor/runs/pixart-qdiff-ref", f"{DC}/runs/pixart-qdiff-ref", False),
        ] + [(f"{VAULT}/deepcompressor/runs/pixart-qdiff-damp{lam}",
              f"{DC}/runs/pixart-qdiff-damp{lam}", False) for lam in LAMBDAS_OLD],
        official_fetch=[
            (f"{VAULT}/deepcompressor/baselines/torch.float16",
             f"{DC}/baselines/torch.float16", False),
            (f"{VAULT}/deepcompressor/runs/pixart-svdq-final-kernel",
             f"{DC}/runs/pixart-svdq-final-kernel", False),
        ],
    ),
    "sdxl": dict(
        ck=lambda tag: f"{REPO}/models/sdxl-turbo/absorb_basis/sdxl_r3_{tag}.pt",
        old_ck=lambda lam: f"{REPO}/models/sdxl-turbo/absorb_basis/sdxl_absorb_damp{lam}_kernel.pt",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/sdxl/build_sdxl_kernel.py "
            f"--cov models/sdxl-turbo/basis/absorb_cov_sdxl_linear.pt "
            f"--cov-conv models/sdxl-turbo/basis/absorb_cov_sdxl_conv.pt "
            f"--out {out} --hsvd-damping {lam} {extra}"),
        smooth=f"{REPO}/models/sdxl-turbo/svdq_model_dump/smooth.pt",
        gains=f"{REPO}/models/sdxl-turbo/absorb_basis/smooth_gain.json",
        runner=f"{REPO}/absorb_basis/sdxl/run_sdxl_kernel_generate.py",
        yaml="configs/model/sdxl-turbo.yaml",
        qref=f"{DC}/runs/sdxl-qdiff-ref/samples/YAML/qdiff-128",
        old_qdiff=lambda lam: f"{DC}/runs/sdxl-qdiff-damp{lam}/samples/YAML/qdiff-128",
        current_final="damp0.1+S@1.0",
        official=dict(n=2500, bench="MJHQ",
                      ref=f"{DC}/runs/sdxl-ref/samples/MJHQ/MJHQ-2500",
                      gt=f"{DC}/benchmarks/MJHQ-GT-2500",
                      svdq=f"{DC}/runs/sdxl-svdq-final-kernel/samples/MJHQ/MJHQ-2500"),
        fetch=[], official_fetch=[],
    ),
    "sana": dict(
        ck=lambda tag: f"{REPO}/models/sana-1.6b/absorb_basis/sana_r3_{tag}.pt",
        old_ck=lambda lam: f"{REPO}/models/sana-1.6b/absorb_basis/sana_absorb_damp{lam}_kernel.pt",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/sana/build_sana_kernel.py "
            f"--cov models/sana-1.6b/basis/absorb_cov_sana.pt "
            f"--out {out} --hsvd-damping {lam} {extra}"),
        smooth=f"{REPO}/models/sana-1.6b/svdq_model_dump/smooth.pt",
        gains=f"{REPO}/models/sana-1.6b/absorb_basis/smooth_gain.json",
        runner=f"{REPO}/absorb_basis/sana/run_sana_kernel_generate.py",
        yaml="configs/model/sana-1.6b.yaml",
        qref=f"{DC}/runs/sana-qdiff-ref/samples/YAML/qdiff-128",
        old_qdiff=lambda lam: f"{DC}/runs/sana-qdiff-damp{lam}/samples/YAML/qdiff-128",
        current_final="damp0.003",
        official=dict(n=2500, bench="MJHQ",
                      ref=f"{DC}/runs/sana-ref/samples/MJHQ/MJHQ-2500",
                      gt=f"{DC}/benchmarks/MJHQ-GT-2500",
                      svdq=f"{DC}/runs/sana-svdq-final-kernel/samples/MJHQ/MJHQ-2500"),
        fetch=[
            (f"{VAULT}/DiRotQ/models/sana-1.6b/basis", f"{REPO}/models/sana-1.6b/basis", False),
            (f"{VAULT}/DiRotQ/models/sana-1.6b/absorb_basis", f"{REPO}/models/sana-1.6b/absorb_basis", False),
            (f"{VAULT}/DiRotQ/models/sana-1.6b/svdq_model_dump", f"{REPO}/models/sana-1.6b/svdq_model_dump", True),
            (f"{VAULT}/deepcompressor/runs/sana-qdiff-ref", f"{DC}/runs/sana-qdiff-ref", False),
        ] + [(f"{VAULT}/deepcompressor/runs/sana-qdiff-damp{lam}",
              f"{DC}/runs/sana-qdiff-damp{lam}", False) for lam in LAMBDAS_OLD],
        official_fetch=[
            (f"{VAULT}/deepcompressor/runs/sana-ref", f"{DC}/runs/sana-ref", False),
            (f"{VAULT}/deepcompressor/runs/sana-svdq-final-kernel",
             f"{DC}/runs/sana-svdq-final-kernel", False),
        ],
    ),
}[M] if sys.argv[1] != "flux" else None

if M == "flux":
    # FLUX: safetensors builds via build_checkpoint (v1 C++ loader);
    # stage B (per-channel-top) skipped — loader support unverified.
    MF = f"{REPO}/models/flux-schnell/absorb_basis"
    CFG = dict(
        ck=lambda tag: f"{MF}/dirotq-r3-{tag}-fp4_r32-flux.1-schnell.safetensors",
        old_ck=lambda lam: (
            f"{MF}/dirotq-absorb-hsvd-down-fp4_r32-flux.1-schnell.safetensors"
            if lam == "0.01" else
            f"{MF}/dirotq-absorb-hsvd-down-damp{lam}-fp4_r32-flux.1-schnell.safetensors"),
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/build_checkpoint.py --basis hsvd --down-absorb "
            f"--hsvd-damping {lam} --out {out} {extra}"),
        smooth=None,
        gains=f"{MF}/smooth_gain.json",
        runner=None,
        yaml=None,
        qref=f"{DC}/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/YAML/qdiff-128",
        old_qdiff=lambda lam: {
            "0.003": f"{DC}/runs/dirotq-absorb-hsvd-down-damp0.003-flux.1-schnell/samples/YAML/qdiff-128",
            "0.01": f"{DC}/runs/dirotq-absorb-hsvd-down-flux.1-schnell/samples/YAML/qdiff-128",
            "0.1": f"{DC}/runs/dirotq-absorb-hsvd-down-damp0.1-flux.1-schnell/samples/YAML/qdiff-128",
        }[lam],
        current_final="damp0.01+S@1.0",
        official=dict(n=1000, bench="MJHQ",
                      ref=f"{DC}/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1000",
                      gt=f"{DC}/benchmarks/MJHQ-GT-1000",
                      svdq=f"{DC}/runs/nvfp4-nunchaku-flux.1-schnell/samples/MJHQ/MJHQ-1000"),
        fetch=[
            (f"{VAULT}/DiRotQ/models/flux-schnell/basis", f"{REPO}/models/flux-schnell/basis", False),
            (f"{VAULT}/DiRotQ/models/flux-schnell/absorb_basis", MF, False),
            (f"{VAULT}/deepcompressor/baselines/torch.bfloat16",
             f"{DC}/baselines/torch.bfloat16", False),
        ] + [(f"{VAULT}/deepcompressor/runs/" + {
                "0.003": "dirotq-absorb-hsvd-down-damp0.003-flux.1-schnell",
                "0.01": "dirotq-absorb-hsvd-down-flux.1-schnell",
                "0.1": "dirotq-absorb-hsvd-down-damp0.1-flux.1-schnell"}[lam],
              f"{DC}/runs/" + {
                "0.003": "dirotq-absorb-hsvd-down-damp0.003-flux.1-schnell",
                "0.01": "dirotq-absorb-hsvd-down-flux.1-schnell",
                "0.1": "dirotq-absorb-hsvd-down-damp0.1-flux.1-schnell"}[lam], False)
             for lam in LAMBDAS_OLD],
        official_fetch=[
            (f"{VAULT}/deepcompressor/runs/nvfp4-nunchaku-flux.1-schnell",
             f"{DC}/runs/nvfp4-nunchaku-flux.1-schnell", False),
        ],
    )


def gen_qdiff(ckpt, root):
    out = f"{DC}/runs/{root}/samples/YAML/qdiff-128"
    if count_png(out) == 128:
        print(f"gen {root}: cached", flush=True)
        return out
    disk_guard()
    if M == "flux":
        rc = sh(f"{PY} -u {REPO}/absorb_basis/flux_gen_nunchaku.py --num-samples 128 "
                f"--benchmark prompts/qdiff.yaml --weight-path '{ckpt}' "
                f"--out-root '{DC}/runs/{root}' --stats-out '{DC}/runs/{root}/stats.json'",
                cwd=DC)
    else:
        rc = sh(f"{PY} {CFG['runner']} {CFG['yaml']} --kernel-weights '{ckpt}' "
                f"--gen-root 'runs/{root}' --stats-out '{DC}/runs/{root}/stats.json' "
                f"--eval-benchmarks prompts/qdiff.yaml --eval-num-samples 128 "
                f"--eval-num-gpus 1 --eval-batch-size 1 --skip-eval", cwd=DC)
    assert rc == 0 and count_png(out) == 128, f"gen {root} failed"
    return out


def build(out, lam, extra=""):
    if os.path.exists(out):
        print(f"build {os.path.basename(out)}: cached", flush=True)
        return
    disk_guard()
    rc = sh(CFG["build"](out, lam, extra), cwd=REPO)
    assert rc == 0 and os.path.exists(out), f"build {out} failed"


def smooth_extra(alpha):
    if M == "flux":
        return (f"--select-smooth-gains {CFG['gains']} "
                f"--select-smooth-alpha {alpha}")
    return (f"--smooth-pt {CFG['smooth']} --gains {CFG['gains']} "
            f"--smooth-alpha {alpha}")


state = {}

# ---- fetch prerequisites ----------------------------------------------------
step(f"{M}-fetch")
for src, dst, deref in CFG["fetch"]:
    fetch(src, dst, deref)
print(f"STEP {M}-fetch exit 0", flush=True)

# ---- Stage A: lambda 6-point grid -------------------------------------------
step(f"{M}-stageA")
cand = {}
for lam in LAMBDAS_OLD:
    cand[f"damp{lam}"] = (CFG["old_ck"](lam), CFG["old_qdiff"](lam))
for lam in LAMBDAS_NEW:
    ck = CFG["ck"](f"damp{lam}")
    build(ck, lam)
    d = gen_qdiff(ck, f"{M}-r3-qdiff-damp{lam}")
    cand[f"damp{lam}"] = (ck, d)
res = {}
for tag, (ck, d) in cand.items():
    assert count_png(d) == 128, f"missing qdiff images for {tag}: {d}"
    res[tag] = stats_vs_ref(CFG["qref"], d)
    print(f"METRICS {M}-{tag}: " + json.dumps(res[tag]), flush=True)
best = max(res, key=lambda k: sum(wins(res[k], res[o]) for o in res if o != k))
lam_star = best.replace("damp", "")
print(f"R3_{M.upper()}_LAMBDA lambda={lam_star}", flush=True)
state["A"] = {"lambda": lam_star, "metrics": res}
cur_tag, cur_ck, cur_stats = best, cand[best][0], res[best]
print(f"STEP {M}-stageA exit 0", flush=True)

# ---- Stage B: per-channel top (not for flux) --------------------------------
if M != "flux":
    step(f"{M}-stageB")
    ck = CFG["ck"](f"damp{lam_star}_pct")
    build(ck, lam_star, "--per-channel-top")
    d = gen_qdiff(ck, f"{M}-r3-qdiff-pct")
    s = stats_vs_ref(CFG["qref"], d)
    print(f"METRICS {M}-pct: " + json.dumps(s), flush=True)
    if wins(s, cur_stats) >= 3:
        cur_tag, cur_ck, cur_stats = f"damp{lam_star}+pct", ck, s
    print(f"R3_{M.upper()}_PCT gate={'accept' if 'pct' in cur_tag else 'reject'}",
          flush=True)
    print(f"STEP {M}-stageB exit 0", flush=True)

# ---- Stage C: selective smoothing strength alpha ----------------------------
if os.path.exists(CFG["gains"]):
    step(f"{M}-stageC")
    pct_extra = "--per-channel-top " if "pct" in cur_tag and M != "flux" else ""
    best_alpha = None
    for alpha in ALPHAS:
        ck = CFG["ck"](f"damp{lam_star}{'_pct' if pct_extra else ''}_S{alpha}")
        build(ck, lam_star, pct_extra + smooth_extra(alpha))
        d = gen_qdiff(ck, f"{M}-r3-qdiff-S{alpha}{'p' if pct_extra else ''}")
        s = stats_vs_ref(CFG["qref"], d)
        print(f"METRICS {M}-S@{alpha}: " + json.dumps(s), flush=True)
        if wins(s, cur_stats) >= 3:
            cur_tag, cur_ck, cur_stats = (
                f"damp{lam_star}{'+pct' if pct_extra else ''}+S@{alpha}", ck, s)
            best_alpha = alpha
    print(f"R3_{M.upper()}_S alpha={best_alpha}", flush=True)
    print(f"STEP {M}-stageC exit 0", flush=True)

json.dump({"final": cur_tag, "ckpt": cur_ck, "stats": cur_stats, "state": state},
          open(f"{RESULTS}/{M}_round3_selection.json", "w"), indent=2, default=str)
print(f"R3_{M.upper()}_FINAL config={cur_tag} (was {CFG['current_final']})", flush=True)

# ---- Officials if config changed --------------------------------------------
if cur_tag != CFG["current_final"]:
    step(f"{M}-official")
    for src, dst, deref in CFG["official_fetch"]:
        fetch(src, dst, deref)
    o = CFG["official"]
    root = f"{M}-r3-final-kernel"
    out = f"{DC}/runs/{root}/samples/{o['bench']}/{o['bench']}-{o['n']}"
    if count_png(out) != o["n"]:
        disk_guard()
        if M == "flux":
            rc = sh(f"{PY} -u {REPO}/absorb_basis/flux_gen_nunchaku.py "
                    f"--num-samples {o['n']} --weight-path '{cur_ck}' "
                    f"--out-root '{DC}/runs/{root}' "
                    f"--stats-out '{DC}/runs/{root}/stats.json'", cwd=DC)
        else:
            rc = sh(f"{PY} {CFG['runner']} {CFG['yaml']} --kernel-weights '{cur_ck}' "
                    f"--gen-root 'runs/{root}' --stats-out '{DC}/runs/{root}/stats.json' "
                    f"--eval-benchmarks {o['bench']} --eval-num-samples {o['n']} "
                    f"--eval-num-gpus 1 --eval-batch-size 1 --skip-eval", cwd=DC)
        assert rc == 0 and count_png(out) == o["n"], "official gen failed"
    fin = {}
    for tag, d in [(f"absorb-{cur_tag}", out), ("svdquant", o["svdq"])]:
        m = compute_image_similarity_metrics(o["ref"], d,
                                             metrics=("psnr", "lpips", "ssim"),
                                             num_workers=0)
        fin[tag] = {k: float(v) for k, v in m.items()}
        fin[tag]["fid_vs_ref"] = cf.compute_fid(o["ref"], d, verbose=False)
        fin[tag]["fid_vs_gt"] = cf.compute_fid(o["gt"], d, verbose=False)
        print(f"FINAL {M}-{tag}: " + json.dumps(fin[tag]), flush=True)
    json.dump(fin, open(f"{RESULTS}/{M}_round3_test.json", "w"), indent=2)
    print(f"STEP {M}-official exit 0", flush=True)
else:
    print(f"{M}: config unchanged, no official rerun", flush=True)

sh("bash /vault/dirotq-absorb-backup/backup.sh >/dev/null 2>&1")
print(f"R3_{M.upper()}_DONE", flush=True)
