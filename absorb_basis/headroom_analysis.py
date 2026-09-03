"""Small-scale headroom analysis for the next quality push (FLUX-schnell,
all offline from stored cov / act samples / weights; no image generation).

A. Per-layer lambda heterogeneity: full hsvd+GPTQ at lambda in
   {0.01, 0.3, 1e6} on 12 sampled layers; report the ACT-SAMPLE weighted
   output error ||X(W_hat-W)^T|| (the T1 metric) and the unweighted qsnr,
   per layer -> does the best lambda differ per layer / per metric?
B. Rank waterfilling oracle: full sigma_i(WC) spectra for all 228 non-down
   layers at lambda*=0.01; compare uniform r=32 vs waterfilled ranks under
   the same total-rank budget (H-metric residual energy).
C. Per-layer alpha* for S_rms: offline e1(alpha) curves on act samples;
   gain of per-layer alpha* vs the best global alpha (restricted to the
   tau-gated layer set and to all layers).
D. Alternating refit headroom: after GPTQ at lambda*, refit the rank-32
   lora on (W - Q) in the H metric; report weighted-error reduction.
E. Bias-correction headroom: ||mu (W_hat - W)^T|| vs rms of X W^T.

Run: python absorb_basis/headroom_analysis.py > results/headroom_analysis.log
"""

import json
import math
import sys

import torch

sys.path.insert(0, "/home/dev/DiRotQ")

from absorb_basis.build_checkpoint import (
    hsvd_basis, layer_table, load_transformer_state_dict, quantize_residual,
)
from absorb_basis.pixart.run_pixart_sim_generate import act_fp4_sim

REPO = "/home/dev/DiRotQ"
LAM_STAR = 0.01


@torch.no_grad()
def quant_layer(W, H, lam, rank=32):
    D, lu = hsvd_basis(W, H, rank, "cuda", damping=lam)
    W_res = W - lu @ D
    W_q, _, _ = quantize_residual(W_res, H.to("cuda", torch.float32), "plain",
                                  16, "cuda", gptq=True, damp_pct=0.01,
                                  block_size=128)
    return W_q + lu @ D, D, lu, W_q


@torch.no_grad()
def main():
    sd = load_transformer_state_dict("black-forest-labs/FLUX.1-schnell")
    cov = torch.load(f"{REPO}/models/flux-schnell/basis/absorb_cov_basis.pt",
                     map_location="cpu", weights_only=False)
    xs = torch.load(f"{REPO}/models/flux-schnell/basis/absorb_act_samples.pt",
                    map_location="cpu", weights_only=False)
    table = [e for e in layer_table(19, 38) if e[2] in xs]

    # ---- A + D + E on 12 sampled layers -------------------------------------
    sample = [t for i, t in enumerate(table) if i % 19 == 3][:12]
    print("== A: per-layer lambda heterogeneity (act-weighted err, lower=better)")
    lam_grid = [0.01, 0.3, 1e6]
    winners_w, winners_u = [], []
    refit_gains, bias_ratio = [], []
    for nk, w_keys, ck, kind, col in sample:
        W = torch.cat([sd[k].float() for k in w_keys], dim=0)
        if col is not None:
            W = W[:, col[0]:col[1]]
        W = W.cuda()
        H = cov[f"{ck}.H"]
        X = xs[ck].to("cuda", torch.float32)
        errs_w, errs_u = {}, {}
        for lam in lam_grid:
            W_hat, D, lu, W_q = quant_layer(W, H, lam)
            dw = W_hat - W
            errs_w[lam] = float((X @ dw.t()).pow(2).sum())
            errs_u[lam] = float(dw.pow(2).sum())
            if lam == LAM_STAR:
                # D: refit lora on (W - W_q) in the H metric, then re-quantize
                C = torch.linalg.cholesky(
                    H.to("cuda", torch.float64)
                    + LAM_STAR * H.diagonal().mean().to("cuda") *
                    torch.eye(H.shape[0], device="cuda", dtype=torch.float64))
                T = (W - W_q).to(torch.float64) @ C
                U, S, Vh = torch.linalg.svd(T, full_matrices=False)
                Lr = (U[:, :32] * S[:32]) @ Vh[:32] @ torch.linalg.inv(C)
                W_res2 = W - Lr.float()
                W_q2, _, _ = quantize_residual(W_res2, H.to("cuda", torch.float32),
                                               "plain", 16, "cuda", gptq=True,
                                               damp_pct=0.01, block_size=128)
                dw2 = (W_q2 + Lr.float()) - W
                e2 = float((X @ dw2.t()).pow(2).sum())
                refit_gains.append(10 * math.log10(errs_w[lam] / max(e2, 1e-30)))
                # E: bias-correction headroom at lambda*
                mu = X.mean(0)
                num = float((mu @ dw.t()).pow(2).sum())
                den = float((X @ W.t()).pow(2).mean())
                bias_ratio.append(num / max(den, 1e-30))
        bw = min(errs_w, key=errs_w.get)
        bu = max(errs_u, key=lambda k: -errs_u[k])
        winners_w.append(bw)
        winners_u.append(min(errs_u, key=errs_u.get))
        rel = {l: errs_w[l] / errs_w[bw] for l in lam_grid}
        print(f"  {nk}: best_w lam={bw} rel_w={{"
              + ", ".join(f"{l}: {rel[l]:.3f}" for l in lam_grid) + "}")
    from collections import Counter
    print("A summary: act-weighted winners", dict(Counter(map(str, winners_w))),
          "| unweighted winners", dict(Counter(map(str, winners_u))))
    print(f"D summary: refit(+reGPTQ) gain dB on {len(refit_gains)} layers: "
          f"median {sorted(refit_gains)[len(refit_gains)//2]:+.3f}, "
          f"max {max(refit_gains):+.3f}, min {min(refit_gains):+.3f}")
    print(f"E summary: bias-term share of output power: median "
          f"{sorted(bias_ratio)[len(bias_ratio)//2]:.2e}, max {max(bias_ratio):.2e}")

    # ---- B: rank waterfilling oracle ---------------------------------------
    print("== B: rank waterfill oracle (lambda*, all layers, budget = 32/layer)")
    spectra = []
    for nk, w_keys, ck, kind, col in table:
        W = torch.cat([sd[k].float() for k in w_keys], dim=0)
        if col is not None:
            W = W[:, col[0]:col[1]]
        W = W.cuda()
        H = cov[f"{ck}.H"].to("cuda", torch.float64)
        n = H.shape[0]
        C = torch.linalg.cholesky(
            H + LAM_STAR * H.diagonal().mean() * torch.eye(n, device="cuda",
                                                           dtype=torch.float64))
        s = torch.linalg.svdvals(W.to(torch.float64) @ C)
        spectra.append(s.pow(2).cpu())
        del W, H, C
    budget = 32 * len(spectra)
    uniform = sum(float(s[32:].sum()) for s in spectra)
    # greedy waterfill: repeatedly give the next rank unit to the layer with
    # the largest current marginal sigma^2 (cap rank at 64 for kernel sanity)
    ranks = [0] * len(spectra)
    import heapq
    heap = [(-float(s[0]), i) for i, s in enumerate(spectra)]
    heapq.heapify(heap)
    for _ in range(budget):
        neg, i = heapq.heappop(heap)
        ranks[i] += 1
        if ranks[i] < min(64, len(spectra[i])):
            heapq.heappush(heap, (-float(spectra[i][ranks[i]]), i))
    water = sum(float(s[r:].sum()) for s, r in zip(spectra, ranks))
    print(f"B summary: weighted residual energy uniform={uniform:.4e} "
          f"waterfilled={water:.4e} reduction={100 * (1 - water / uniform):.1f}% "
          f"| rank range [{min(ranks)}, {max(ranks)}], "
          f"layers at cap64: {sum(r == 64 for r in ranks)}")

    # ---- C: per-layer alpha* for S_rms -------------------------------------
    print("== C: per-layer alpha* (S_rms, offline e1 on act samples)")
    gains = json.load(open(f"{REPO}/models/flux-schnell/absorb_basis/"
                           "selfsmooth_gain_rms_lam0.3.json")) \
        if False else json.load(open(f"{REPO}/models/flux-schnell/absorb_basis/"
                                     "selfsmooth_gain_rms.json"))
    vecs = torch.load(f"{REPO}/models/flux-schnell/absorb_basis/selfsmooth_rms.pt",
                      map_location="cpu", weights_only=False)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_by_layer, e_global, e_perlayer = {}, {a: 0.0 for a in alphas}, 0.0
    gated = 0
    for nk, w_keys, ck, kind, col in table:
        W = torch.cat([sd[k].float() for k in w_keys], dim=0)
        if col is not None:
            W = W[:, col[0]:col[1]]
        W = W.cuda()
        H = cov[f"{ck}.H"]
        D, lu = hsvd_basis(W, H, 32, "cuda", damping=LAM_STAR)
        Wres = (W - lu @ D).half()
        X = xs[ck].to("cuda", torch.float16)
        yt = (X @ Wres.t()).float()
        s_full = vecs[nk].to("cuda", torch.float32)
        e = {}
        for a in alphas:
            s = s_full.pow(a)
            e[a] = float(((act_fp4_sim((X.float() / s).half())
                           @ (Wres.float() * s.unsqueeze(0)).half().t()).float()
                          - yt).pow(2).sum())
            e_global[a] += e[a]
        ba = min(e, key=e.get)
        best_by_layer[nk] = ba
        e_perlayer += e[ba]
        if gains.get(nk, -9) > 0.3:
            gated += 1
        del W, D, lu, Wres, X, yt
        torch.cuda.empty_cache()
    from collections import Counter as Cnt
    ga = min(e_global, key=e_global.get)
    print(f"C summary: alpha* histogram {dict(Cnt(best_by_layer.values()))}")
    print(f"C summary: best-global-alpha={ga} err={e_global[ga]:.4e}; "
          f"per-layer-alpha* err={e_perlayer:.4e} "
          f"({10 * math.log10(e_global[ga] / e_perlayer):+.3f} dB); "
          f"current pipeline (gate tau=0.3, {gated} layers smoothed, alpha "
          f"global) vs per-layer continuum")
    print("HEADROOM_DONE")


if __name__ == "__main__":
    main()
