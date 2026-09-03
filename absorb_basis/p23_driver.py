"""PLAN_NEXTQ P3 (clip-search re-audit) + P2 (down-proj closed-form S)
driver. Candidates are gated with the standard >=3/4 four-criteria rule
against the CURRENT zero-dependency final config's qdiff-128 images;
officials rerun automatically on acceptance (config change convention).

Usage (svdquant env, repo root):
  python absorb_basis/p23_driver.py p3 pixart|flux
  python absorb_basis/p23_driver.py p2 flux|fluxdev
Results: results/p23_qdiff128.json (+ {model}_p23_test{n}.json on accept)
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, "/home/dev/DiRotQ")

from absorb_basis.selfsmooth_driver import (
    CFGS, DC, PY, REPO, RESULTS, count_png, disk_guard, gen_qdiff, sh,
    stats_vs_ref, wins,
)

MF = f"{REPO}/models/flux-schnell"
MD = f"{REPO}/models/flux-dev"
ACCEL_FLUX = (f"--factor-cache {MF}/basis/factor_cache "
              f"--adanorm-cache {MF}/absorb_basis/adanorm_cache.pt ")
ACCEL_DEV = (f"--factor-cache {MD}/basis/factor_cache "
             f"--adanorm-cache {MD}/absorb_basis/adanorm_cache.pt ")

# current zero-dependency final configs: qdiff images + build extras
FINAL = {
    "pixart": dict(
        imgs=f"{DC}/runs/pixart-selfsmooth-qdiff-rms0.5/samples/YAML/qdiff-128",
        tag="damp0.1+S_rms@0.5",
        extra=(f"--smooth-pt {REPO}/models/pixart-sigma/absorb_basis/selfsmooth_rms.pt "
               f"--gains {REPO}/models/pixart-sigma/absorb_basis/selfsmooth_gain_rms.json "
               f"--smooth-alpha 0.5 ")),
    "flux": dict(
        imgs=f"{DC}/runs/flux-selfsmooth-qdiff-base/samples/YAML/qdiff-128",
        tag="damp0.01", extra=ACCEL_FLUX),
    "fluxdev": dict(
        imgs=f"{DC}/runs/fluxdev-selfsmooth-qdiff-rms0.25/samples/YAML/qdiff-128",
        tag="damp0.3+S_rms@0.25", extra=ACCEL_DEV),
}


def step(name):
    print(f"=== STEP {name} start {time.strftime('%H:%M:%S')} ===", flush=True)


def record(key, entry):
    path = f"{RESULTS}/p23_qdiff128.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[key] = entry
    json.dump(res, open(path, "w"), indent=2)


def evaluate(M, CFG, ck, root, fin):
    imgs = f"{DC}/runs/{root}/samples/YAML/qdiff-128"
    if count_png(imgs) != 128:
        gen_qdiff(M, CFG, ck, root)
    s = stats_vs_ref(CFG["qref"], imgs)
    base = stats_vs_ref(CFG["qref"], fin["imgs"])
    w_c, w_b = wins(s, base), wins(base, s)
    return s, base, w_c, w_b


def official(M, CFG, cur_tag, cur_ck, suffix):
    o = CFG["official"]
    n = o["n"]
    assert count_png(o["ref"]) == n and count_png(o["svdq"]) == n, \
        "official refs missing"
    root = f"{M}-p23-final"
    outd = f"{DC}/runs/{root}/samples/MJHQ/MJHQ-{n}"
    if count_png(outd) != n:
        disk_guard()
        rc = sh(f"{PY} -u {REPO}/absorb_basis/flux_gen_nunchaku.py "
                f"{CFG['gen_extra']}--num-samples {n} --weight-path '{cur_ck}' "
                f"--out-root '{DC}/runs/{root}' "
                f"--stats-out '{DC}/runs/{root}/stats.json'", cwd=DC) \
            if CFG["is_flux"] else \
            sh(f"{PY} {CFG['runner']} {CFG['yaml']} --kernel-weights '{cur_ck}' "
               f"--gen-root 'runs/{root}' --stats-out '{DC}/runs/{root}/stats.json' "
               f"--eval-benchmarks MJHQ --eval-num-samples {n} "
               f"--eval-num-gpus 1 --eval-batch-size 1 --skip-eval", cwd=DC)
        assert rc == 0 and count_png(outd) == n, "official gen failed"
    from cleanfid import fid as cf
    from deepcompressor.app.diffusion.eval.metrics.similarity import (
        compute_image_similarity_metrics,
    )
    fin = {}
    for tag, d in [(f"absorb-{cur_tag}", outd), ("svdquant", o["svdq"])]:
        m = compute_image_similarity_metrics(o["ref"], d,
                                             metrics=("psnr", "lpips", "ssim"),
                                             num_workers=0)
        fin[tag] = {k: float(v) for k, v in m.items()}
        fin[tag]["fid_vs_ref"] = cf.compute_fid(o["ref"], d, verbose=False)
        fin[tag]["fid_vs_gt"] = cf.compute_fid(o["gt"], d, verbose=False)
        print(f"FINAL {M}-{tag}: " + json.dumps(fin[tag]), flush=True)
    json.dump(fin, open(f"{RESULTS}/{M}_p23_test{n}.json", "w"), indent=2)


def run_p3(M):
    CFG, fin = CFGS[M], FINAL[M]
    step(f"p3-{M}")
    ck = CFG["ck"]("p3clip")
    if not os.path.exists(ck) and \
            count_png(f"{DC}/runs/{M}-p3clip-qdiff/samples/YAML/qdiff-128") != 128:
        disk_guard()
        rc = sh(CFG["build"](ck, CFG["lam"], fin["extra"] + "--clip-search"),
                cwd=REPO)
        assert rc == 0 and os.path.exists(ck), "p3 build failed"
    s, base, w_c, w_b = evaluate(M, CFG, ck, f"{M}-p3clip-qdiff", fin)
    accepted = w_c >= 3
    record(f"p3_{M}", {"final": base, "clip": s, "wins_clip_vs_final": w_c,
                       "wins_final_vs_clip": w_b, "accepted": accepted})
    print(f"METRICS {M}-p3clip: " + json.dumps(s), flush=True)
    print(f"P23_{M.upper()}_CLIP gate={'accept' if accepted else 'reject'} "
          f"({w_c}:{w_b})", flush=True)
    if accepted:
        official(M, CFG, fin["tag"] + "+clip", ck, "p3")
    elif CFG["is_flux"] and os.path.exists(ck):
        sh(f"rm -f '{ck}'")


def p2_files(M):
    """Build the merged gains/vectors files for the down-S candidate."""
    if M == "flux":  # base has no S: down-only files, global alpha applies
        d = f"{MF}/absorb_basis"
        return (f"{d}/selfsmooth_down_gain_rms.json",
                f"{d}/selfsmooth_down_rms.pt", "ALPHA")
    d = f"{MD}/absorb_basis"  # dev: keep non-down S at 0.25, down at alpha_eff
    gains = json.load(open(f"{d}/selfsmooth_gain_rms_lam0.3.json"))
    gains.update(json.load(open(f"{d}/selfsmooth_down_gain_rms.json")))
    gp = f"{d}/p2_merged_gains.json"
    json.dump(gains, open(gp, "w"), indent=2)
    vec = torch.load(f"{d}/selfsmooth_rms_lam0.3.pt", map_location="cpu",
                     weights_only=False)
    down = torch.load(f"{d}/selfsmooth_down_rms.pt", map_location="cpu",
                      weights_only=False)
    return gp, vec, down


def run_p2(M):
    CFG, fin = CFGS[M], FINAL[M]
    step(f"p2-{M}")
    gains_dn = json.load(open(
        f"{(MF if M == 'flux' else MD)}/absorb_basis/selfsmooth_down_gain_rms.json"))
    n_pass = sum(1 for v in gains_dn.values() if v > 0.3)
    print(f"{M}: down gains — {n_pass}/{len(gains_dn)} layers above tau", flush=True)
    if n_pass == 0:
        record(f"p2_{M}", {"down_gains_above_tau": 0, "accepted": False})
        print(f"P23_{M.upper()}_DOWNS gate=reject (no layer passes tau)", flush=True)
        return
    alphas_eff = ["0.5", "0.25"]  # screen 0.5 first, fall back to 0.25
    for a_eff in alphas_eff:
        if M == "flux":
            gp = f"{MF}/absorb_basis/selfsmooth_down_gain_rms.json"
            vp = f"{MF}/absorb_basis/selfsmooth_down_rms.pt"
            extra = (fin["extra"] +
                     f"--select-smooth-gains {gp} --select-smooth-vectors {vp} "
                     f"--select-smooth-alpha {a_eff} "
                     f"--reuse-from {CFG['base_ck']} ")
        else:
            d = f"{MD}/absorb_basis"
            gp, vec, down = p2_files(M)
            power = float(a_eff) / 0.25  # global alpha stays 0.25
            merged = dict(vec)
            merged.update({k: v.float().pow(power) for k, v in down.items()})
            vp = f"{d}/p2_merged_vectors_a{a_eff}.pt"
            torch.save(merged, vp)
            extra = (fin["extra"] +
                     f"--select-smooth-gains {gp} --select-smooth-vectors {vp} "
                     f"--select-smooth-alpha 0.25 "
                     f"--reuse-from {CFG['base_ck']} ")
        tag = f"downS{a_eff}"
        ck = CFG["ck"](f"p2_{tag}")
        if not os.path.exists(ck) and \
                count_png(f"{DC}/runs/{M}-p2-qdiff-{tag}/samples/YAML/qdiff-128") != 128:
            disk_guard()
            rc = sh(CFG["build"](ck, CFG["lam"], extra), cwd=REPO)
            assert rc == 0 and os.path.exists(ck), "p2 build failed"
        s, base, w_c, w_b = evaluate(M, CFG, ck, f"{M}-p2-qdiff-{tag}", fin)
        accepted = w_c >= 3
        record(f"p2_{M}_{tag}", {"final": base, "downS": s,
                                 "wins_downS_vs_final": w_c,
                                 "wins_final_vs_downS": w_b,
                                 "accepted": accepted})
        print(f"METRICS {M}-p2-{tag}: " + json.dumps(s), flush=True)
        print(f"P23_{M.upper()}_DOWNS@{a_eff} "
              f"gate={'accept' if accepted else 'reject'} ({w_c}:{w_b})",
              flush=True)
        if accepted:
            official(M, CFG, fin["tag"] + f"+downS@{a_eff}", ck, "p2")
            return
        if CFG["is_flux"] and os.path.exists(ck):
            sh(f"rm -f '{ck}'")


if __name__ == "__main__":
    mode, M = sys.argv[1], sys.argv[2]
    (run_p3 if mode == "p3" else run_p2)(M)
    print(f"P23_{mode.upper()}_{M.upper()}_DONE", flush=True)
