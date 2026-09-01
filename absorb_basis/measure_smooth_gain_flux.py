"""FLUX per-layer smoothing gain on the lambda* H-SVD residuals (PLAN_ROUND2 S
selection), offline variant: act samples + cov + the official checkpoint's
packed smooth factors — no model forward needed.

For each layer-table entry:
  Wres = W - lora_up @ D          (H-SVD with --lam damping, rank 32)
  s    = unpack(official.smooth)  (same source build_checkpoint uses)
  e0   = ||fp4(X) Wres^T - X Wres^T||^2
  e1   = ||fp4(X/s) (Wres*s)^T - X Wres^T||^2
  gain = 10 log10(e0/e1)   [dB]

Usage (from repo root, svdquant env):
  python absorb_basis/measure_smooth_gain_flux.py --lam 0.01 \
      --model-id black-forest-labs/FLUX.1-dev \
      --official models/flux-dev/svdq-fp4_r32-flux.1-dev.safetensors \
      --cov models/flux-dev/basis/absorb_cov_basis.pt \
      --act-samples models/flux-dev/basis/absorb_act_samples.pt \
      --out models/flux-dev/absorb_basis/smooth_gain.json
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from absorb_basis.build_checkpoint import (
    hsvd_basis, layer_table, load_transformer_state_dict, unpack_scale_vector,
)
from absorb_basis.pixart.run_pixart_sim_generate import act_fp4_sim


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--official", required=True)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--act-samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--num-double", type=int, default=19)
    ap.add_argument("--num-single", type=int, default=38)
    args = ap.parse_args()

    sd = load_transformer_state_dict(args.model_id)
    cov = torch.load(args.cov, map_location="cpu", weights_only=False)
    xs = torch.load(args.act_samples, map_location="cpu", weights_only=False)
    tensors = load_file(args.official)

    gains = {}
    for nk_prefix, w_keys, cov_key, kind, col_slice in tqdm(
            layer_table(args.num_double, args.num_single), dynamic_ncols=True):
        skey = f"{nk_prefix}.smooth"
        if skey not in tensors or cov_key not in xs:
            continue
        W = torch.cat([sd[k].float() for k in w_keys], dim=0)
        if col_slice is not None:
            W = W[:, col_slice[0]:col_slice[1]]
        W = W.cuda()
        # the flux cov file stores eigenvectors under cov_key and the raw
        # second-moment matrix under f"{cov_key}.H" (build_checkpoint.get_H)
        H = cov[f"{cov_key}.H"]
        D, lu = hsvd_basis(W, H, args.rank, "cuda", damping=args.lam)
        Wres = (W - lu @ D).half()
        s = unpack_scale_vector(tensors[skey]).float().cuda()
        X = xs[cov_key].to("cuda", torch.float16)
        yt = (X @ Wres.t()).float()
        e0 = ((act_fp4_sim(X) @ Wres.t()).float() - yt).pow(2).sum()
        Wres_s = (Wres.float() * s.unsqueeze(0)).half()
        Xs = (X.float() / s).half()
        e1 = ((act_fp4_sim(Xs) @ Wres_s.t()).float() - yt).pow(2).sum()
        gains[nk_prefix] = (10 * math.log10(float(e0) / float(e1))
                            if e0 > 0 and e1 > 0 else 0.0)
        del W, D, lu, Wres, Wres_s, X, Xs, yt
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(gains, f, indent=2)
    v = sorted(gains.values())
    print(f"flux: {len(gains)} layers, median {statistics.median(v):+.2f} dB, "
          f">+0.3: {sum(x > 0.3 for x in v)}, <-0.1: {sum(x < -0.1 for x in v)}")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
