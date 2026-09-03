"""THEORY Prop 4' middle-rung ablation: basis metric = diag(H_lambda)
(the smooth-then-SVD / ASVD / LQER class) on schnell + pixart, everything
else identical to the lambda* pipeline (full-H GPTQ, no S, temb adanorm).

Completes the metric hierarchy row: I (svd_inf, cached) -> diag (this run)
-> full H (lambda* base, cached), four qdiff-128 criteria.

Run: python absorb_basis/diagmetric_ablation.py
Output: results/diagmetric_qdiff128.json
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

JOBS = [("pixart", "--basis-diag"), ("flux", "--basis-metric diag")]


def step(name):
    print(f"=== STEP {name} start {time.strftime('%H:%M:%S')} ===", flush=True)


def main():
    out_path = f"{RESULTS}/diagmetric_qdiff128.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    svdinf = json.load(open(f"{RESULTS}/svdbasis_qdiff128.json"))
    for M, flag in JOBS:
        if M in res:
            print(f"{M}: cached", flush=True)
            continue
        CFG = CFGS[M]
        step(f"diag-{M}")
        ck = CFG["ck"]("diagmetric")
        imgs = f"{DC}/runs/{M}-diagmetric-qdiff/samples/YAML/qdiff-128"
        if count_png(imgs) != 128:
            if not os.path.exists(ck):
                disk_guard()
                rc = sh(CFG["build"](ck, CFG["lam"], flag), cwd=REPO)
                assert rc == 0 and os.path.exists(ck), f"build {ck} failed"
            gen_qdiff(M, CFG, ck, f"{M}-diagmetric-qdiff")
        s_diag = stats_vs_ref(CFG["qref"], imgs)
        s_full = stats_vs_ref(CFG["qref"], CFG["base_qdiff"])
        s_id = svdinf[M]["svd_inf"]
        res[M] = {
            "metric_I": s_id, "metric_diag": s_diag, "metric_fullH": s_full,
            "wins_full_vs_diag": wins(s_full, s_diag),
            "wins_diag_vs_full": wins(s_diag, s_full),
            "wins_diag_vs_I": wins(s_diag, s_id),
            "wins_I_vs_diag": wins(s_id, s_diag),
        }
        print(f"METRICS {M}-diag: " + json.dumps(s_diag), flush=True)
        print(f"DIAGMETRIC_{M.upper()} full_vs_diag="
              f"{res[M]['wins_full_vs_diag']}:{res[M]['wins_diag_vs_full']} "
              f"diag_vs_I={res[M]['wins_diag_vs_I']}:{res[M]['wins_I_vs_diag']}",
              flush=True)
        json.dump(res, open(out_path, "w"), indent=2)
        if CFG["is_flux"] and os.path.exists(ck):
            sh(f"rm -f '{ck}'")
    print("DIAGMETRIC_DONE", flush=True)


if __name__ == "__main__":
    main()
