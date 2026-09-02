"""PLAN_SELFSMOOTH: self-contained closed-form smoothing vectors + offline
gain measurement for FLUX (schnell/dev). Zero SVDQuant-calibration inputs:
uses only our cov (diag H), our act amax, our act samples, and W_res(lambda*).

Two families (full-strength s stored; the builder applies s^alpha):
  rms : s_k = rms_x,k / rms_w,k   (rms_x = sqrt(diag H), rms_w = col-RMS of W_res)
        -> alpha=0.5 is the exact minimizer of the separable error bound.
  amax: s_k = amax_x,k / amax_w,k (full-stream act amax / col-absmax of W_res)
Both geometric-mean normalized (global scale is quantization-neutral).

Gains use the same offline estimator as measure_smooth_gain_flux.py
(e0/e1 on act samples), per family, at alpha=1.

Usage (svdquant env, repo root):
  python absorb_basis/selfsmooth_vectors_flux.py --lam 0.01 \
      --model-id black-forest-labs/FLUX.1-schnell \
      --cov models/flux-schnell/basis/absorb_cov_basis.pt \
      --act-samples models/flux-schnell/basis/absorb_act_samples.pt \
      --act-amax models/flux-schnell/basis/absorb_act_amax.pt \
      --out-dir models/flux-schnell/absorb_basis
Outputs: <out-dir>/selfsmooth_{rms,amax}.pt   ({nk_prefix: s float32})
         <out-dir>/selfsmooth_gain_{rms,amax}.json
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from absorb_basis.build_checkpoint import (
    hsvd_basis, layer_table, load_transformer_state_dict,
)
from absorb_basis.pixart.run_pixart_sim_generate import act_fp4_sim


def geo_normalize(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp(min=1e-6)
    return s / s.log().mean().exp()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--cov", required=True)
    ap.add_argument("--act-samples", required=True)
    ap.add_argument("--act-amax", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--num-double", type=int, default=19)
    ap.add_argument("--num-single", type=int, default=38)
    args = ap.parse_args()

    sd = load_transformer_state_dict(args.model_id)
    cov = torch.load(args.cov, map_location="cpu", weights_only=False)
    xs = torch.load(args.act_samples, map_location="cpu", weights_only=False)
    am = torch.load(args.act_amax, map_location="cpu", weights_only=False)

    vecs = {"rms": {}, "amax": {}}
    gains = {"rms": {}, "amax": {}}
    for nk_prefix, w_keys, cov_key, kind, col_slice in tqdm(
            layer_table(args.num_double, args.num_single), dynamic_ncols=True):
        if cov_key not in xs or cov_key not in am:
            continue
        W = torch.cat([sd[k].float() for k in w_keys], dim=0)
        if col_slice is not None:
            W = W[:, col_slice[0]:col_slice[1]]
        W = W.cuda()
        H = cov[f"{cov_key}.H"]
        D, lu = hsvd_basis(W, H, args.rank, "cuda", damping=args.lam)
        Wres = (W - lu @ D).half()

        rms_x = H.diagonal().float().clamp(min=0).sqrt().cuda()
        rms_w = Wres.float().pow(2).mean(dim=0).sqrt()
        amax_x = am[cov_key].float().cuda()
        amax_w = Wres.float().abs().amax(dim=0)
        s_fam = {
            "rms": geo_normalize(rms_x.clamp(min=1e-6) / rms_w.clamp(min=1e-6)),
            "amax": geo_normalize(amax_x.clamp(min=1e-6) / amax_w.clamp(min=1e-6)),
        }

        X = xs[cov_key].to("cuda", torch.float16)
        yt = (X @ Wres.t()).float()
        e0 = ((act_fp4_sim(X) @ Wres.t()).float() - yt).pow(2).sum()
        for fam, s in s_fam.items():
            s = s.to(torch.bfloat16).float()  # match runtime storage precision
            vecs[fam][nk_prefix] = s.cpu()
            Wres_s = (Wres.float() * s.unsqueeze(0)).half()
            Xs = (X.float() / s).half()
            e1 = ((act_fp4_sim(Xs) @ Wres_s.t()).float() - yt).pow(2).sum()
            gains[fam][nk_prefix] = (10 * math.log10(float(e0) / float(e1))
                                     if e0 > 0 and e1 > 0 else 0.0)
            del Wres_s, Xs
        del W, D, lu, Wres, X, yt
        torch.cuda.empty_cache()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fam in ("rms", "amax"):
        torch.save(vecs[fam], out / f"selfsmooth_{fam}.pt")
        with open(out / f"selfsmooth_gain_{fam}.json", "w") as f:
            json.dump(gains[fam], f, indent=2)
        v = sorted(gains[fam].values())
        print(f"{fam}: {len(v)} layers, median {statistics.median(v):+.2f} dB, "
              f">+0.3: {sum(x > 0.3 for x in v)}, <-0.1: {sum(x < -0.1 for x in v)}")
    print("saved:", out)


if __name__ == "__main__":
    main()
