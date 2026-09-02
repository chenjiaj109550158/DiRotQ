"""THEORY.md experimental closure: the lambda->infinity ablation.

Per model, build ONE extra candidate whose basis is the plain
(unweighted) rank-32 SVD of W — SVDQuant's decomposition — inside our
otherwise-identical pipeline (same NVFP4 GPTQ on our H, same kernels,
no smoothing, flux adanorm/temb identical). Implemented faithfully as
the lambda->infinity endpoint of the damped H-SVD family:
--hsvd-damping 1e6 (H + 1e6*mean(diag)*I ~ scaled identity).

Then rank on qdiff-128 with the same four calibration criteria against
the lambda* no-S base of each model (direct end-to-end instantiation of
Corollary 2.1 / Prop 3.1 in THEORY.md).

Run in the svdquant env from the repo root:
  python absorb_basis/svdbasis_ablation.py
Output: results/svdbasis_qdiff128.json; flux-size checkpoints are
deleted after their images are generated.
"""

import json
import os
import sys
import time

sys.path.insert(0, "/home/dev/DiRotQ")

from absorb_basis.selfsmooth_driver import (
    CFGS, DC, REPO, RESULTS, count_png, disk_guard, gen_qdiff, sh,
    stats_vs_ref, wins,
)

LAM_INF = "1e6"
ORDER = ["sana", "pixart", "sdxl", "sdxlb", "flux", "fluxdev"]


def step(name):
    print(f"=== STEP {name} start {time.strftime('%H:%M:%S')} ===", flush=True)


def main():
    out_path = f"{RESULTS}/svdbasis_qdiff128.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for M in ORDER:
        if M in res:
            print(f"{M}: cached", flush=True)
            continue
        CFG = CFGS[M]
        step(f"svdinf-{M}")
        assert count_png(CFG["qref"]) == 128, f"missing qref {CFG['qref']}"
        assert count_png(CFG["base_qdiff"]) == 128, f"missing base {CFG['base_qdiff']}"
        ck = CFG["ck"]("svdinf")
        imgs = f"{DC}/runs/{M}-svdinf-qdiff/samples/YAML/qdiff-128"
        if count_png(imgs) != 128:
            if not os.path.exists(ck):
                disk_guard()
                rc = sh(CFG["build"](ck, LAM_INF, ""), cwd=REPO)
                assert rc == 0 and os.path.exists(ck), f"build {ck} failed"
            gen_qdiff(M, CFG, ck, f"{M}-svdinf-qdiff")
        s_inf = stats_vs_ref(CFG["qref"], imgs)
        s_star = stats_vs_ref(CFG["qref"], CFG["base_qdiff"])
        res[M] = {
            "lambda_star_base": s_star,
            "svd_inf": s_inf,
            "wins_star_vs_inf": wins(s_star, s_inf),
            "wins_inf_vs_star": wins(s_inf, s_star),
        }
        print(f"METRICS {M}-svdinf: " + json.dumps(s_inf), flush=True)
        print(f"SVDBASIS_{M.upper()} star_vs_inf={res[M]['wins_star_vs_inf']}:"
              f"{res[M]['wins_inf_vs_star']}", flush=True)
        json.dump(res, open(out_path, "w"), indent=2)
        if CFG["is_flux"] and os.path.exists(ck):
            sh(f"rm -f '{ck}'")  # 7.1G each; images + stats retained
    print("SVDBASIS_DONE", flush=True)


if __name__ == "__main__":
    main()
