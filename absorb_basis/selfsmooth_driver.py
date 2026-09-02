"""PLAN_SELFSMOOTH driver: per-model S_closed menu with Algorithm-1 gating on
qdiff-128, fully self-contained (zero SVDQuant calibration artifacts: no
svdq_model_dump smooth, no official-checkpoint smooth unpack, no
cov_actq_smooth; the FLUX --official file is used ONLY as a tensor-layout
container and an audit asserts every calibration-derived tensor is replaced).

Menu per model: base = lambda* no-S (existing qdiff images), candidates =
S_closed(family)@alpha built from selfsmooth_{family}.pt +
selfsmooth_gain_{family}.json (our own calibration only), greedy >=3/4 gate
(round3 stage-C semantics, family-major, rms first).

Then, if the final config differs from the currently published official
config, rerun the official benchmark on the ours side (ref/GT/SVDQuant
images reused; svdq images fetched from vault if pruned) and emit the
5-metric table.

Run in the svdquant env from the repo root:
  python absorb_basis/selfsmooth_driver.py pixart|sana|sdxl|sdxlb|flux|fluxdev \
      [--families rms,amax] [--alphas 0.25,0.5,0.75,1.0] [--skip-official]
"""

import argparse
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


def fetch(src, dst):
    os.makedirs(os.path.dirname(dst.rstrip("/")), exist_ok=True)
    rc = sh(f"cp -au '{src}/.' '{dst}/'" if os.path.isdir(src)
            else f"cp -au '{src}' '{dst}'")
    print(f"fetched {src} -> {dst} rc={rc}", flush=True)


def disk_guard():
    st = os.statvfs("/")
    avail_g = st.f_bavail * st.f_frsize / 2**30
    assert avail_g > 45, f"disk low: {avail_g:.0f}G"


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
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return r


def wins(a, b):
    return sum([a["psnr"] > b["psnr"], a["lpips"] < b["lpips"],
                a["ssim"] > b["ssim"], a["fid_proxy_128"] < b["fid_proxy_128"]])


DEV_ID = "black-forest-labs/FLUX.1-dev"
MDEV = f"{REPO}/models/flux-dev"
CAL_SUFFIXES = (".qweight", ".wscales", ".smooth", ".smooth_orig",
                ".lora_down", ".lora_up", ".wtscale", ".wcscales")

CFGS = {
    "pixart": dict(
        lam="0.1", current_final="damp0.1+S@0.5", is_flux=False,
        vec_dir=f"{REPO}/models/pixart-sigma/absorb_basis",
        ck=lambda tag: f"{REPO}/models/pixart-sigma/absorb_basis/pixart_selfsmooth_{tag}.pt",
        base_ck=f"{REPO}/models/pixart-sigma/absorb_basis/pixart_absorb_damp0.1_kernel.pt",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/pixart/build_pixart_kernel.py "
            f"--cov models/pixart-sigma/basis/absorb_cov_pixart.pt "
            f"--out {out} --hsvd-damping {lam} {extra}"),
        runner=f"{REPO}/absorb_basis/pixart/run_pixart_kernel_generate.py",
        yaml="configs/model/pixart-sigma.yaml",
        qref=f"{DC}/runs/pixart-qdiff-ref/samples/YAML/qdiff-128",
        base_qdiff=f"{DC}/runs/pixart-qdiff-damp0.1/samples/YAML/qdiff-128",
        official=dict(n=2500,
                      ref=f"{DC}/baselines/torch.float16/pixart-sigma/dpm20-g4.5/samples/MJHQ/MJHQ-2500",
                      gt=f"{DC}/benchmarks/MJHQ-GT-2500",
                      svdq=f"{DC}/runs/pixart-svdq-final-kernel/samples/MJHQ/MJHQ-2500",
                      svdq_vault=f"{VAULT}/deepcompressor/runs/pixart-svdq-final-kernel",
                      base_dir=None),
    ),
    "sana": dict(
        lam="0.3", current_final="damp0.3+S@0.25", is_flux=False,
        vec_dir=f"{REPO}/models/sana-1.6b/absorb_basis",
        ck=lambda tag: f"{REPO}/models/sana-1.6b/absorb_basis/sana_selfsmooth_{tag}.pt",
        base_ck=f"{REPO}/models/sana-1.6b/absorb_basis/sana_r3_damp0.3.pt",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/sana/build_sana_kernel.py "
            f"--cov models/sana-1.6b/basis/absorb_cov_sana.pt "
            f"--out {out} --hsvd-damping {lam} {extra}"),
        runner=f"{REPO}/absorb_basis/sana/run_sana_kernel_generate.py",
        yaml="configs/model/sana-1.6b.yaml",
        qref=f"{DC}/runs/sana-qdiff-ref/samples/YAML/qdiff-128",
        base_qdiff=f"{DC}/runs/sana-r3-qdiff-damp0.3/samples/YAML/qdiff-128",
        official=dict(n=2500,
                      ref=f"{DC}/runs/sana-ref/samples/MJHQ/MJHQ-2500",
                      gt=f"{DC}/benchmarks/MJHQ-GT-2500",
                      svdq=f"{DC}/runs/sana-svdq-final-kernel/samples/MJHQ/MJHQ-2500",
                      svdq_vault=f"{VAULT}/deepcompressor/runs/sana-svdq-final-kernel",
                      base_dir=None),
    ),
    "sdxl": dict(
        lam="0.3", current_final="damp0.3", is_flux=False,
        vec_dir=f"{REPO}/models/sdxl-turbo/absorb_basis",
        ck=lambda tag: f"{REPO}/models/sdxl-turbo/absorb_basis/sdxl_selfsmooth_{tag}.pt",
        base_ck=f"{REPO}/models/sdxl-turbo/absorb_basis/sdxl_r3_damp0.3.pt",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/sdxl/build_sdxl_kernel.py "
            f"--cov models/sdxl-turbo/basis/absorb_cov_sdxl_linear.pt "
            f"--cov-conv models/sdxl-turbo/basis/absorb_cov_sdxl_conv.pt "
            f"--out {out} --hsvd-damping {lam} {extra}"),
        runner=f"{REPO}/absorb_basis/sdxl/run_sdxl_kernel_generate.py",
        yaml="configs/model/sdxl-turbo.yaml",
        qref=f"{DC}/runs/sdxl-qdiff-ref/samples/YAML/qdiff-128",
        base_qdiff=f"{DC}/runs/sdxl-r3-qdiff-damp0.3/samples/YAML/qdiff-128",
        official=dict(n=2500,
                      ref=f"{DC}/runs/sdxl-ref/samples/MJHQ/MJHQ-2500",
                      gt=f"{DC}/benchmarks/MJHQ-GT-2500",
                      svdq=f"{DC}/runs/sdxl-svdq-final-kernel/samples/MJHQ/MJHQ-2500",
                      svdq_vault=f"{VAULT}/deepcompressor/runs/sdxl-svdq-final-kernel",
                      base_dir=None),
    ),
    "sdxlb": dict(
        lam="0.001", current_final="damp0.001+S@0.5", is_flux=False,
        vec_dir=f"{REPO}/models/sdxl-base/absorb_basis",
        ck=lambda tag: f"{REPO}/models/sdxl-base/absorb_basis/sdxlb_selfsmooth_{tag}.pt",
        base_ck=f"{REPO}/models/sdxl-base/absorb_basis/sdxlb_damp0.001.pt",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/sdxl/build_sdxl_kernel.py "
            f"--model-id stabilityai/stable-diffusion-xl-base-1.0 "
            f"--cov models/sdxl-base/basis/absorb_cov_sdxl_linear.pt "
            f"--cov-conv models/sdxl-base/basis/absorb_cov_sdxl_conv.pt "
            f"--out {out} --hsvd-damping {lam} {extra}"),
        runner=f"{REPO}/absorb_basis/sdxl/run_sdxl_kernel_generate.py",
        yaml="configs/model/sdxl.yaml",
        qref=f"{DC}/runs/sdxlb-qdiff-ref/samples/YAML/qdiff-128",
        base_qdiff=f"{DC}/runs/sdxlb-qdiff-damp0.001/samples/YAML/qdiff-128",
        official=dict(n=1000,
                      ref=f"{DC}/runs/sdxlb-ref/samples/MJHQ/MJHQ-1000",
                      gt=f"{DC}/benchmarks/MJHQ-GT-1000",
                      svdq=f"{DC}/runs/sdxlb-svdq-final-kernel/samples/MJHQ/MJHQ-1000",
                      svdq_vault=f"{VAULT}/deepcompressor/runs/sdxlb-svdq-final-kernel",
                      base_dir=None),
    ),
    "flux": dict(
        lam="0.01", current_final="damp0.01+S@1.0", is_flux=True,
        # the published base checkpoint carries the container's
        # SVDQuant-quantized adanorm linears -> rebuild the no-S base with
        # the data-free adanorm requant and fresh qdiff images, and always
        # rerun the official benchmark (adanorm bits changed)
        base_build=True, always_official=True,
        vec_dir=f"{REPO}/models/flux-schnell/absorb_basis",
        ck=lambda tag: (f"{REPO}/models/flux-schnell/absorb_basis/"
                        f"dirotq-selfsmooth-{tag}-fp4_r32-flux.1-schnell.safetensors"),
        base_ck=(f"{REPO}/models/flux-schnell/absorb_basis/"
                 "dirotq-selfsmooth-base-fp4_r32-flux.1-schnell.safetensors"),
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/build_checkpoint.py --basis hsvd --down-absorb "
            f"--hsvd-damping {lam} --out {out} {extra}"),
        gen_extra="",
        official_file=None,  # resolved from HF cache at audit time
        qref=f"{DC}/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/YAML/qdiff-128",
        base_qdiff=f"{DC}/runs/flux-selfsmooth-qdiff-base/samples/YAML/qdiff-128",
        official=dict(n=1000,
                      ref=f"{DC}/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1000",
                      ref_vault=f"{VAULT}/deepcompressor/baselines/torch.bfloat16",
                      ref_vault_dst=f"{DC}/baselines/torch.bfloat16",
                      gt=f"{DC}/benchmarks/MJHQ-GT-1000",
                      svdq=f"{DC}/runs/nvfp4-nunchaku-flux.1-schnell/samples/MJHQ/MJHQ-1000",
                      svdq_vault=f"{VAULT}/deepcompressor/runs/nvfp4-nunchaku-flux.1-schnell",
                      base_dir=None),
    ),
    "fluxdev": dict(
        lam="0.3", current_final="damp0.3", is_flux=True,
        base_build=True, always_official=True,
        vec_dir=f"{MDEV}/absorb_basis",
        ck=lambda tag: f"{MDEV}/absorb_basis/fluxdev_selfsmooth_{tag}.safetensors",
        base_ck=f"{MDEV}/absorb_basis/fluxdev_selfsmooth_base.safetensors",
        build=lambda out, lam, extra: (
            f"{PY} absorb_basis/build_checkpoint.py "
            f"--model-id {DEV_ID} --official {MDEV}/svdq-fp4_r32-flux.1-dev.safetensors "
            f"--cov {MDEV}/basis/absorb_cov_basis.pt "
            f"--cov-down-dir {MDEV}/basis/absorb_cov_down "
            f"--basis hsvd --down-absorb --hsvd-damping {lam} --out {out} {extra}"),
        gen_extra=(f"--base-model {DEV_ID} --num-steps 50 --guidance-scale 3.5 "),
        official_file=f"{MDEV}/svdq-fp4_r32-flux.1-dev.safetensors",
        qref=f"{DC}/runs/fluxdev-qdiff-ref/samples/YAML/qdiff-128",
        base_qdiff=f"{DC}/runs/fluxdev-selfsmooth-qdiff-base/samples/YAML/qdiff-128",
        official=dict(n=500,
                      ref=f"{DC}/runs/fluxdev-ref/samples/MJHQ/MJHQ-500",
                      gt=f"{DC}/benchmarks/MJHQ-GT-500",
                      svdq=f"{DC}/runs/fluxdev-svdq-final/samples/MJHQ/MJHQ-500",
                      svdq_vault=f"{VAULT}/deepcompressor/runs/fluxdev-svdq-final",
                      base_dir=None),
    ),
}


def gen_qdiff(M, CFG, ckpt, root):
    out = f"{DC}/runs/{root}/samples/YAML/qdiff-128"
    if count_png(out) == 128:
        print(f"gen {root}: cached", flush=True)
        return out
    disk_guard()
    if CFG["is_flux"]:
        rc = sh(f"{PY} -u {REPO}/absorb_basis/flux_gen_nunchaku.py {CFG['gen_extra']}"
                f"--num-samples 128 --benchmark prompts/qdiff.yaml "
                f"--weight-path '{ckpt}' --out-root '{DC}/runs/{root}' "
                f"--stats-out '{DC}/runs/{root}/stats.json'", cwd=DC)
    else:
        rc = sh(f"{PY} {CFG['runner']} {CFG['yaml']} --kernel-weights '{ckpt}' "
                f"--gen-root 'runs/{root}' --stats-out '{DC}/runs/{root}/stats.json' "
                f"--eval-benchmarks prompts/qdiff.yaml --eval-num-samples 128 "
                f"--eval-num-gpus 1 --eval-batch-size 1 --skip-eval", cwd=DC)
    assert rc == 0 and count_png(out) == 128, f"gen {root} failed"
    return out


def audit_flux_container(built, official):
    """Assert every calibration-derived tensor in the built checkpoint was
    replaced (differs from the official container)."""
    from safetensors import safe_open
    n_checked, same = 0, []
    with safe_open(built, framework="pt") as fb, \
            safe_open(official, framework="pt") as fo:
        keys = set(fb.keys())
        for k in sorted(keys):
            if not k.endswith(CAL_SUFFIXES):
                continue
            n_checked += 1
            tb, to = fb.get_tensor(k), fo.get_tensor(k)
            if tb.shape == to.shape and torch.equal(tb, to):
                # identity smoothing (all ones) carries no calibration info —
                # SVDQuant leaves the 12288-dim down projections unsmoothed
                # and so do we, so equality there is benign
                if k.endswith((".smooth", ".smooth_orig")) and \
                        bool((tb.float() == 1).all()):
                    continue
                same.append(k)
    assert not same, f"container audit FAILED, unreplaced calibration tensors: {same[:10]}"
    print(f"container audit OK: {n_checked} calibration-derived tensors all replaced",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(CFGS))
    ap.add_argument("--families", default="rms,amax")
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--skip-official", action="store_true")
    args = ap.parse_args()
    M, CFG = args.model, CFGS[args.model]
    families = args.families.split(",")
    alphas = args.alphas.split(",")
    lam = CFG["lam"]

    # ---- prerequisites (all self-contained artifacts) -----------------------
    step(f"{M}-prereq")
    for fam in families:
        for f in (f"{CFG['vec_dir']}/selfsmooth_{fam}.pt",
                  f"{CFG['vec_dir']}/selfsmooth_gain_{fam}.json"):
            assert os.path.exists(f), f"missing {f}"
        cmd = CFG["build"]("/dev/null", lam, "")
        assert "svdq_model_dump" not in cmd and "cov_actq" not in cmd, cmd
    assert count_png(CFG["qref"]) == 128, f"missing qref {CFG['qref']}"
    if not CFG.get("base_build"):
        assert count_png(CFG["base_qdiff"]) == 128, f"missing base qdiff {CFG['base_qdiff']}"
        assert os.path.exists(CFG["base_ck"]), f"missing base ckpt {CFG['base_ck']}"
    print(f"STEP {M}-prereq exit 0", flush=True)

    step(f"{M}-menu")
    audited = False

    def audit(ck):
        nonlocal audited
        if not CFG["is_flux"] or audited:
            return
        off = CFG["official_file"]
        if off is None:
            from huggingface_hub import hf_hub_download
            off = hf_hub_download("mit-han-lab/nunchaku-flux.1-schnell",
                                  "svdq-fp4_r32-flux.1-schnell.safetensors")
        audit_flux_container(ck, off)
        audited = True

    if CFG.get("base_build"):
        # self-contained no-S base (data-free adanorm requant included)
        if not os.path.exists(CFG["base_ck"]):
            disk_guard()
            rc = sh(CFG["build"](CFG["base_ck"], lam, ""), cwd=REPO)
            assert rc == 0 and os.path.exists(CFG["base_ck"]), "base build failed"
        audit(CFG["base_ck"])
        gen_qdiff(M, CFG, CFG["base_ck"], f"{M}-selfsmooth-qdiff-base")
    base_stats = stats_vs_ref(CFG["qref"], CFG["base_qdiff"])
    print(f"METRICS {M}-base(damp{lam}): " + json.dumps(base_stats), flush=True)
    cur_tag, cur_ck, cur_stats = f"damp{lam}", CFG["base_ck"], base_stats
    results = {"base": base_stats, "candidates": {}}
    for fam in families:
        vec = f"{CFG['vec_dir']}/selfsmooth_{fam}.pt"
        gains = f"{CFG['vec_dir']}/selfsmooth_gain_{fam}.json"
        for a in alphas:
            tag = f"{fam}{a}"
            ck = CFG["ck"](f"damp{lam}_{tag}")
            if CFG["is_flux"]:
                extra = (f"--select-smooth-gains {gains} "
                         f"--select-smooth-vectors {vec} "
                         f"--select-smooth-alpha {a}")
            else:
                extra = (f"--smooth-pt {vec} --gains {gains} "
                         f"--smooth-alpha {a}")
            if not os.path.exists(ck):
                disk_guard()
                rc = sh(CFG["build"](ck, lam, extra), cwd=REPO)
                assert rc == 0 and os.path.exists(ck), f"build {ck} failed"
            audit(ck)
            d = gen_qdiff(M, CFG, ck, f"{M}-selfsmooth-qdiff-{tag}")
            s = stats_vs_ref(CFG["qref"], d)
            results["candidates"][f"S_{fam}@{a}"] = s
            print(f"METRICS {M}-S_{fam}@{a}: " + json.dumps(s), flush=True)
            accepted = wins(s, cur_stats) >= 3
            if accepted:
                prev_ck, prev_tag = cur_ck, cur_tag
                cur_tag, cur_ck, cur_stats = f"damp{lam}+S_{fam}@{a}", ck, s
                if CFG["is_flux"] and prev_ck != CFG["base_ck"]:
                    sh(f"rm -f '{prev_ck}'")
            elif CFG["is_flux"] and ck != cur_ck:
                sh(f"rm -f '{ck}'")  # 7.1G per loser
            print(f"SELFSMOOTH_{M.upper()} S_{fam}@{a} "
                  f"gate={'accept' if accepted else 'reject'}", flush=True)
    results.update({"final": cur_tag, "ckpt": cur_ck,
                    "stats": cur_stats, "was": CFG["current_final"]})
    json.dump(results, open(f"{RESULTS}/{M}_selfsmooth_qdiff128.json", "w"),
              indent=2, default=str)
    print(f"SELFSMOOTH_{M.upper()}_FINAL config={cur_tag} "
          f"(published {CFG['current_final']})", flush=True)
    print(f"STEP {M}-menu exit 0", flush=True)

    # ---- official rerun if the self-contained config differs ---------------
    if args.skip_official:
        print(f"{M}: official stage skipped by flag", flush=True)
        return
    if cur_tag == CFG["current_final"] and not CFG.get("always_official"):
        print(f"{M}: config unchanged ({cur_tag}), official numbers stand; "
              f"pipeline is already self-contained", flush=True)
        return
    step(f"{M}-official")
    o = CFG["official"]
    n = o["n"]
    if count_png(o["ref"]) != n and "ref_vault" in o:
        fetch(o["ref_vault"], o["ref_vault_dst"])
    if count_png(o["svdq"]) != n:
        fetch(o["svdq_vault"], os.path.dirname(os.path.dirname(os.path.dirname(o["svdq"]))))
    assert count_png(o["ref"]) == n and count_png(o["svdq"]) == n, "missing official refs"
    if cur_tag == f"damp{lam}" and o["base_dir"] and count_png(o["base_dir"]) == n:
        out = o["base_dir"]
        print(f"{M}: reusing cached base official images {out}", flush=True)
    else:
        root = f"{M}-selfsmooth-final"
        out = f"{DC}/runs/{root}/samples/MJHQ/MJHQ-{n}"
        if count_png(out) != n:
            disk_guard()
            if CFG["is_flux"]:
                rc = sh(f"{PY} -u {REPO}/absorb_basis/flux_gen_nunchaku.py "
                        f"{CFG['gen_extra']}--num-samples {n} --weight-path '{cur_ck}' "
                        f"--out-root '{DC}/runs/{root}' "
                        f"--stats-out '{DC}/runs/{root}/stats.json'", cwd=DC)
            else:
                rc = sh(f"{PY} {CFG['runner']} {CFG['yaml']} --kernel-weights '{cur_ck}' "
                        f"--gen-root 'runs/{root}' --stats-out '{DC}/runs/{root}/stats.json' "
                        f"--eval-benchmarks MJHQ --eval-num-samples {n} "
                        f"--eval-num-gpus 1 --eval-batch-size 1 --skip-eval", cwd=DC)
            assert rc == 0 and count_png(out) == n, "official gen failed"
    fin = {}
    for tag, d in [(f"absorb-{cur_tag}", out), ("svdquant", o["svdq"])]:
        m = compute_image_similarity_metrics(o["ref"], d,
                                             metrics=("psnr", "lpips", "ssim"),
                                             num_workers=0)
        fin[tag] = {k: float(v) for k, v in m.items()}
        fin[tag]["fid_vs_ref"] = cf.compute_fid(o["ref"], d, verbose=False)
        fin[tag]["fid_vs_gt"] = cf.compute_fid(o["gt"], d, verbose=False)
        print(f"FINAL {M}-{tag}: " + json.dumps(fin[tag]), flush=True)
    json.dump(fin, open(f"{RESULTS}/{M}_selfsmooth_test{n}.json", "w"), indent=2)
    print(f"STEP {M}-official exit 0", flush=True)


if __name__ == "__main__":
    main()
    print("SELFSMOOTH_MODEL_DONE", flush=True)
