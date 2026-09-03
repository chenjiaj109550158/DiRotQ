"""PLAN_MX end-game driver: ours Algorithm-1 menu in MXFP4e2 on PixArt,
then finals (ours 2500 + svdq-MX 2500) and the five official metrics.

Stages (each cache-aware):
  menu   : lambda grid {6 std + 1e6} -> S_rms(MX gains) x alpha gate
  final  : winner 2500 MJHQ images (MX Triton kernel)
  svdq   : convert svdq_mx_dump -> MX kernel .pt, 2500 MJHQ images
  metrics: PSNR/LPIPS/SSIM + FID vs ref-2500 + FID vs GT-2500

Run: python absorb_basis/mx_round_driver.py menu|final|svdq|metrics
Results: results/pixart_mx_qdiff128.json / pixart_mx_test2500.json
"""

import json
import os
import sys
import time

sys.path.insert(0, "/home/dev/DiRotQ")

from absorb_basis.selfsmooth_driver import (
    CFGS, DC, PY, REPO, RESULTS, count_png, disk_guard, sh, stats_vs_ref, wins,
)

LAMBDAS = ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1e6"]
ALPHAS = ["0.25", "0.5", "0.75", "1.0"]
MP = f"{REPO}/models/pixart-sigma"
CACHES = f"{DC}/datasets/torch.float16/pixart-sigma/dpm20-g4.5/qdiff/s128/caches/*.pt"
QREF = CFGS["pixart"]["qref"]
RUNNER = CFGS["pixart"]["runner"]
YAML = CFGS["pixart"]["yaml"]


def step(name):
    print(f"=== STEP {name} start {time.strftime('%H:%M:%S')} ===", flush=True)


def build(ck, lam, extra=""):
    if os.path.exists(ck):
        return
    disk_guard()
    rc = sh(f"{PY} absorb_basis/pixart/build_pixart_kernel.py "
            f"--cov {MP}/basis/absorb_cov_pixart.pt --grid mx "
            f"--out {ck} --hsvd-damping {lam} {extra}", cwd=REPO)
    assert rc == 0 and os.path.exists(ck), f"build {ck} failed"


def gen(ck, root, bench, n):
    sub = f"YAML/qdiff-{n}" if bench.endswith("yaml") else f"MJHQ/MJHQ-{n}"
    out = f"{DC}/runs/{root}/samples/{sub}"
    if count_png(out) != n:
        disk_guard()
        rc = sh(f"{PY} {RUNNER} {YAML} --kernel-weights '{ck}' "
                f"--gen-root 'runs/{root}' --stats-out '{DC}/runs/{root}/stats.json' "
                f"--eval-benchmarks {bench} --eval-num-samples {n} "
                f"--eval-num-gpus 1 --eval-batch-size 1 --skip-eval", cwd=DC)
        assert rc == 0 and count_png(out) == n, f"gen {root} failed"
    return out


def menu():
    step("mx-menu")
    res, cand = {}, {}
    for lam in LAMBDAS:
        ck = f"{MP}/absorb_basis/pixart_mx_damp{lam}.pt"
        build(ck, lam)
        d = gen(ck, f"pixart-mx-qdiff-damp{lam}", "prompts/qdiff.yaml", 128)
        res[f"damp{lam}"] = stats_vs_ref(QREF, d)
        cand[f"damp{lam}"] = ck
        print(f"METRICS mx-damp{lam}: " + json.dumps(res[f"damp{lam}"]), flush=True)
    best = max(res, key=lambda k: sum(wins(res[k], res[o]) for o in res if o != k))
    lam_star = best.replace("damp", "")
    print(f"MX_LAMBDA lambda={lam_star}", flush=True)
    out = {"stageA": res, "lambda_star": lam_star}
    cur_tag, cur_ck, cur_stats = best, cand[best], res[best]
    # S stage: MX-domain gains + rms vectors at lambda*
    gains_f = f"{MP}/absorb_basis/selfsmooth_gain_rms_mx.json"
    if not os.path.exists(gains_f):
        rc = sh(f"{PY} absorb_basis/selfsmooth_vectors_hook.py --family pixart "
                f"--lam {lam_star} --mx --caches '{CACHES}'", cwd=REPO)
        assert rc == 0 and os.path.exists(gains_f), "mx gains failed"
    n_pass = sum(1 for v in json.load(open(gains_f)).values() if v > 0.3)
    print(f"mx gains: {n_pass} layers above tau", flush=True)
    if n_pass > 0:
        vec_f = f"{MP}/absorb_basis/selfsmooth_rms_mx.pt"
        for a in ALPHAS:
            ck = f"{MP}/absorb_basis/pixart_mx_damp{lam_star}_S{a}.pt"
            build(ck, lam_star,
                  f"--smooth-pt {vec_f} --gains {gains_f} --smooth-alpha {a}")
            d = gen(ck, f"pixart-mx-qdiff-S{a}", "prompts/qdiff.yaml", 128)
            s = stats_vs_ref(QREF, d)
            out.setdefault("stageC", {})[f"S_rms@{a}"] = s
            print(f"METRICS mx-S_rms@{a}: " + json.dumps(s), flush=True)
            if wins(s, cur_stats) >= 3:
                cur_tag, cur_ck, cur_stats = f"damp{lam_star}+S_rms@{a}", ck, s
            print(f"MX_S@{a} gate={'accept' if cur_ck == ck else 'reject'}", flush=True)
    out.update({"final": cur_tag, "ckpt": cur_ck, "stats": cur_stats})
    json.dump(out, open(f"{RESULTS}/pixart_mx_qdiff128.json", "w"), indent=2)
    print(f"MX_FINAL config={cur_tag}", flush=True)


def final():
    step("mx-final-2500")
    sel = json.load(open(f"{RESULTS}/pixart_mx_qdiff128.json"))
    gen(sel["ckpt"], "pixart-mx-final", "MJHQ", 2500)


def svdq():
    step("mx-svdq")
    dump = f"{MP}/svdq_mx_dump"
    assert os.path.isdir(dump), "svdq MX dump missing (calibration not done?)"
    ck = f"{MP}/absorb_basis/pixart_mx_svdq_kernel.pt"
    if not os.path.exists(ck):
        rc = sh(f"{PY} absorb_basis/pixart/build_mx_kernel_from_svdq.py "
                f"--dump {dump} --out {ck}", cwd=REPO)
        assert rc == 0 and os.path.exists(ck), "svdq MX convert failed"
    gen(ck, "pixart-mx-svdq", "MJHQ", 2500)


def metrics():
    step("mx-metrics")
    from cleanfid import fid as cf
    from deepcompressor.app.diffusion.eval.metrics.similarity import (
        compute_image_similarity_metrics,
    )
    sel = json.load(open(f"{RESULTS}/pixart_mx_qdiff128.json"))
    o = CFGS["pixart"]["official"]
    fin = {}
    for tag, d in [(f"absorb-mx-{sel['final']}",
                    f"{DC}/runs/pixart-mx-final/samples/MJHQ/MJHQ-2500"),
                   ("svdquant-mx",
                    f"{DC}/runs/pixart-mx-svdq/samples/MJHQ/MJHQ-2500")]:
        m = compute_image_similarity_metrics(o["ref"], d,
                                             metrics=("psnr", "lpips", "ssim"),
                                             num_workers=0)
        fin[tag] = {k: float(v) for k, v in m.items()}
        fin[tag]["fid_vs_ref"] = cf.compute_fid(o["ref"], d, verbose=False)
        fin[tag]["fid_vs_gt"] = cf.compute_fid(o["gt"], d, verbose=False)
        print(f"FINAL {tag}: " + json.dumps(fin[tag]), flush=True)
    json.dump(fin, open(f"{RESULTS}/pixart_mx_test2500.json", "w"), indent=2)


if __name__ == "__main__":
    {"menu": menu, "final": final, "svdq": svdq, "metrics": metrics}[sys.argv[1]]()
    print(f"MX_{sys.argv[1].upper()}_DONE", flush=True)
