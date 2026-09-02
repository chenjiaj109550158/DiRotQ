"""PLAN_LAMBDAEXT: Algorithm-1 lambda-grid top-end extension for the two
grid-edge models (sdxl-turbo, flux-dev).

Stage A': rank {incumbent lambda* base} + {1, 10, 1e6} by pairwise wins on
the four qdiff-128 criteria (the 1e6 images are reused from the svd-basis
ablation; all candidates on the dev side are temb-adanorm builds).
Stage C: if lambda* changed, recompute the closed-form vectors/gains at the
new lambda* (s depends on W_res(lambda*)) and run the S_rms x alpha greedy
gate on the new base. Officials rerun iff the final config differs from the
current zero-dependency config.

Run in the svdquant env from the repo root:
  python absorb_basis/lambdaext_driver.py sdxl|fluxdev
"""

import json
import os
import sys
import time

sys.path.insert(0, "/home/dev/DiRotQ")

from absorb_basis.selfsmooth_driver import (
    CFGS, DC, PY, REPO, RESULTS, audit_flux_container, count_png,
    disk_guard, fetch, gen_qdiff, sh, stats_vs_ref, wins,
)

NEW_LAMBDAS = ["1", "10", "1e6"]
ALPHAS = ["0.25", "0.5", "0.75", "1.0"]

EXT = {
    "sdxl": dict(
        cur_final="damp0.3",  # current zero-dep config (S all-rejected)
        cur_tag="damp0.3",
        ck=lambda tag: f"{REPO}/models/sdxl-turbo/absorb_basis/sdxl_lext_{tag}.pt",
        vectors=lambda lam: (
            f"{PY} absorb_basis/selfsmooth_vectors_hook.py "
            f"--family sdxl-turbo --lam {lam}"),
        n_official=2500,
    ),
    "fluxdev": dict(
        cur_final="damp0.3+S_rms@0.25",
        cur_tag="damp0.3",
        ck=lambda tag: f"{REPO}/models/flux-dev/absorb_basis/fluxdev_lext_{tag}.safetensors",
        vectors=lambda lam: (
            f"{PY} absorb_basis/selfsmooth_vectors_flux.py --lam {lam} "
            f"--model-id black-forest-labs/FLUX.1-dev "
            f"--cov models/flux-dev/basis/absorb_cov_basis.pt "
            f"--act-samples models/flux-dev/basis/absorb_act_samples.pt "
            f"--act-amax models/flux-dev/basis/absorb_act_amax.pt "
            f"--out-dir models/flux-dev/absorb_basis"),
        n_official=500,
    ),
}


def step(name):
    print(f"=== STEP {name} start {time.strftime('%H:%M:%S')} ===", flush=True)


def main():
    M = sys.argv[1]
    CFG, E = CFGS[M], EXT[M]
    lam_old = CFG["lam"]

    def build(ck, lam, extra=""):
        if os.path.exists(ck):
            return
        disk_guard()
        rc = sh(CFG["build"](ck, lam, extra), cwd=REPO)
        assert rc == 0 and os.path.exists(ck), f"build {ck} failed"
        if CFG["is_flux"]:
            audit_flux_container(ck, CFG["official_file"])

    # ---- Stage A': top-end lambda ranking -----------------------------------
    step(f"{M}-lext-stageA")
    cand = {f"damp{lam_old}": CFG["base_qdiff"]}
    for lam in NEW_LAMBDAS:
        root = (f"{M}-svdinf-qdiff" if lam == "1e6"
                else f"{M}-lext-qdiff-damp{lam}")
        imgs = f"{DC}/runs/{root}/samples/YAML/qdiff-128"
        if count_png(imgs) != 128:
            ck = E["ck"](f"damp{lam}")
            build(ck, lam)
            gen_qdiff(M, CFG, ck, root)
        cand[f"damp{lam}"] = imgs
    res = {}
    for tag, d in cand.items():
        assert count_png(d) == 128, f"missing imgs {tag}: {d}"
        res[tag] = stats_vs_ref(CFG["qref"], d)
        print(f"METRICS {M}-lext-{tag}: " + json.dumps(res[tag]), flush=True)
    best = max(res, key=lambda k: sum(wins(res[k], res[o]) for o in res if o != k))
    lam_star = best.replace("damp", "")
    print(f"LEXT_{M.upper()}_LAMBDA lambda={lam_star} (was {lam_old})", flush=True)
    out = {"stageA": res, "lambda_star": lam_star, "lambda_old": lam_old}
    cur_tag, cur_stats = best, res[best]
    if lam_star == lam_old:
        cur_ck = CFG["base_ck"]
    else:
        cur_ck = E["ck"](f"damp{lam_star}")
        build(cur_ck, lam_star)  # winner ckpt (1e6 images were ckpt-less)

    # ---- Stage C: S_rms x alpha on the (possibly new) base ------------------
    step(f"{M}-lext-stageC")
    if lam_star != lam_old:
        vec_dir = CFG["vec_dir"]
        rc = sh(E["vectors"](lam_star), cwd=REPO)
        assert rc == 0, "vectors failed"
    vec = f"{CFG['vec_dir']}/selfsmooth_rms.pt"
    gains = f"{CFG['vec_dir']}/selfsmooth_gain_rms.json"
    for a in ALPHAS:
        tag = f"damp{lam_star}_rms{a}"
        ck = E["ck"](tag)
        if CFG["is_flux"]:
            extra = (f"--select-smooth-gains {gains} "
                     f"--select-smooth-vectors {vec} --select-smooth-alpha {a}")
        else:
            extra = f"--smooth-pt {vec} --gains {gains} --smooth-alpha {a}"
        imgs_root = f"{M}-lext-qdiff-S{a}"
        imgs = f"{DC}/runs/{imgs_root}/samples/YAML/qdiff-128"
        if count_png(imgs) != 128:
            build(ck, lam_star, extra)
            gen_qdiff(M, CFG, ck, imgs_root)
        s = stats_vs_ref(CFG["qref"], imgs)
        out.setdefault("stageC", {})[f"S_rms@{a}"] = s
        print(f"METRICS {M}-lext-S_rms@{a}: " + json.dumps(s), flush=True)
        accepted = wins(s, cur_stats) >= 3
        if accepted:
            prev = None if cur_tag == best else E["ck"](cur_tag.replace("+S_rms@", "_rms"))
            cur_tag, cur_stats = f"damp{lam_star}+S_rms@{a}", s
            cur_ck = ck
            if CFG["is_flux"] and prev and os.path.exists(prev):
                sh(f"rm -f '{prev}'")
        elif CFG["is_flux"] and os.path.exists(ck) and ck != cur_ck:
            sh(f"rm -f '{ck}'")
        print(f"LEXT_{M.upper()} S_rms@{a} gate={'accept' if accepted else 'reject'}",
              flush=True)
    out.update({"final": cur_tag, "ckpt": cur_ck, "stats": cur_stats,
                "prev_zero_dep": E["cur_final"]})
    json.dump(out, open(f"{RESULTS}/{M}_lambdaext_qdiff128.json", "w"), indent=2)
    print(f"LEXT_{M.upper()}_FINAL config={cur_tag} (prev {E['cur_final']})", flush=True)

    # ---- Official if config changed -----------------------------------------
    if cur_tag == E["cur_final"]:
        print(f"{M}: config unchanged, officials stand", flush=True)
        print(f"LEXT_{M.upper()}_DONE", flush=True)
        return
    step(f"{M}-lext-official")
    o = CFG["official"]
    n = o["n"]
    if count_png(o["svdq"]) != n:
        fetch(o["svdq_vault"],
              os.path.dirname(os.path.dirname(os.path.dirname(o["svdq"]))))
    assert count_png(o["ref"]) == n and count_png(o["svdq"]) == n
    root = f"{M}-lext-final"
    outd = f"{DC}/runs/{root}/samples/MJHQ/MJHQ-{n}"
    if count_png(outd) != n:
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
        assert rc == 0 and count_png(outd) == n, "official gen failed"
    import scipy.linalg  # noqa: F401  (sqrtm patch active via selfsmooth_driver)
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
    json.dump(fin, open(f"{RESULTS}/{M}_lambdaext_test{n}.json", "w"), indent=2)
    print(f"LEXT_{M.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
