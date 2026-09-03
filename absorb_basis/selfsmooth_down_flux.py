"""PLAN_NEXTQ P2: closed-form rms smoothing vectors + offline gains for the
FLUX K=12288 down-projection layers (both SVDQuant and our current configs
leave these unsmoothed; the inputs are post-GELU activations — the classic
outlier layer class).

Outputs in <out-dir>:
  selfsmooth_down_rms.pt        {nk_prefix: s [12288] float32, full strength}
  selfsmooth_down_gain_rms.json {nk_prefix: gain_dB at alpha=1}

Usage:
  python absorb_basis/selfsmooth_down_flux.py --lam 0.01 \
      --model-id black-forest-labs/FLUX.1-schnell \
      --cov-down-dir models/flux-schnell/basis/absorb_cov_down \
      --act-samples models/flux-schnell/basis/absorb_act_samples_down.pt \
      --out-dir models/flux-schnell/absorb_basis
"""

import argparse
import json
import math
import os
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
from absorb_basis.selfsmooth_vectors_flux import geo_normalize


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--cov-down-dir", required=True)
    ap.add_argument("--act-samples", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--num-double", type=int, default=19)
    ap.add_argument("--num-single", type=int, default=38)
    args = ap.parse_args()

    sd = load_transformer_state_dict(args.model_id)
    xs = torch.load(args.act_samples, map_location="cpu", weights_only=False)
    full = layer_table(args.num_double, args.num_single, down_proj=True)
    nondown = {e[0] for e in layer_table(args.num_double, args.num_single)}
    down = [e for e in full if e[0] not in nondown]

    vecs, gains = {}, {}
    for nk_prefix, w_keys, cov_key, kind, col_slice in tqdm(down, dynamic_ncols=True):
        if cov_key not in xs:
            continue
        W = torch.cat([sd[k].float() for k in w_keys], dim=0)
        if col_slice is not None:
            W = W[:, col_slice[0]:col_slice[1]]
        W = W.cuda()
        H = torch.load(os.path.join(args.cov_down_dir, f"{cov_key}.pt"),
                       map_location="cpu", weights_only=False)
        D, lu = hsvd_basis(W, H, args.rank, "cuda", damping=args.lam)
        Wres = (W - lu @ D).half()
        rms_x = H.diagonal().float().clamp(min=0).sqrt().cuda()
        rms_w = Wres.float().pow(2).mean(dim=0).sqrt()
        s = geo_normalize(rms_x.clamp(min=1e-6) / rms_w.clamp(min=1e-6))
        s = s.to(torch.bfloat16).float()
        vecs[nk_prefix] = s.cpu()
        X = xs[cov_key].to("cuda", torch.float16)
        yt = (X @ Wres.t()).float()
        e0 = ((act_fp4_sim(X) @ Wres.t()).float() - yt).pow(2).sum()
        Wres_s = (Wres.float() * s.unsqueeze(0)).half()
        Xs = (X.float() / s).half()
        e1 = ((act_fp4_sim(Xs) @ Wres_s.t()).float() - yt).pow(2).sum()
        gains[nk_prefix] = (10 * math.log10(float(e0) / float(e1))
                            if e0 > 0 and e1 > 0 else 0.0)
        del W, D, lu, Wres, Wres_s, X, Xs, yt, H
        torch.cuda.empty_cache()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(vecs, out / "selfsmooth_down_rms.pt")
    with open(out / "selfsmooth_down_gain_rms.json", "w") as f:
        json.dump(gains, f, indent=2)
    v = sorted(gains.values())
    print(f"down: {len(v)} layers, median {statistics.median(v):+.2f} dB, "
          f">+0.3: {sum(x > 0.3 for x in v)}, <-0.1: {sum(x < -0.1 for x in v)}")


if __name__ == "__main__":
    main()
