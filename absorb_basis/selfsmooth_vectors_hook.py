"""PLAN_SELFSMOOTH: self-contained closed-form smoothing vectors + hook-based
gain measurement for the kernel-.pt model families (pixart / sana /
sdxl-turbo / sdxl-base). Zero SVDQuant-calibration inputs: uses only our cov
(diag H), a full-stream act-amax pass over our qdiff-128 caches, and
W_res(lambda*).

Families (full-strength s stored per hook; builders apply s^alpha via
--smooth-pt/--smooth-alpha, keyed by the same skey rule as svdq smooth.pt):
  rms : s_k = rms_x,k / rms_w,k   (rms_x = sqrt(diag H), rms_w = col-RMS of
        the hook-concatenated W_res)  -> alpha=0.5 is the closed-form bound
        minimizer.
  amax: s_k = amax_x,k / amax_w,k (act absmax over all caches / col-absmax)
Both geometric-mean normalized.

Gains: per-layer e0/e1 on real forwards over 16 strided caches (the
measure_smooth_gain_sdxl precedent), both families in one pass, alpha=1.

Usage (svdquant env, repo root):
  python absorb_basis/selfsmooth_vectors_hook.py --family sana --lam 0.3
Outputs in models/<m>/absorb_basis/:
  selfsmooth_{rms,amax}.pt ({skey: s float32}), selfsmooth_gain_{rms,amax}.json
"""

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from deepcompressor.utils.common import tree_map

from absorb_basis.build_checkpoint import hsvd_basis
from absorb_basis.pixart.run_pixart_sim_generate import act_fp4_sim
from absorb_basis.sana.collect_cov_sana import rows

DC = "/home/dev/deepcompressor/examples/diffusion"


def load_pixart():
    from diffusers import PixArtTransformer2DModel

    from absorb_basis.pixart.build_pixart_sim import layer_table
    m = PixArtTransformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="transformer",
        torch_dtype=torch.float16).to("cuda")
    return m, layer_table(len(m.transformer_blocks))


def load_sana():
    from diffusers import SanaTransformer2DModel

    from absorb_basis.sana.build_sana_kernel import layer_table
    from absorb_basis.sana.collect_cov_sana import MODEL_ID
    m = SanaTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
        variant="bf16").to("cuda")
    return m, layer_table(len(m.transformer_blocks))


def load_sdxl(model_id):
    from diffusers import UNet2DConditionModel

    from absorb_basis.sdxl.build_sdxl_kernel import linear_table
    m = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16,
        variant="fp16").to("cuda")
    return m, linear_table(m)


FAMILIES = {
    "pixart": dict(
        load=load_pixart, dtype=torch.float16,
        cov=f"{REPO}/models/pixart-sigma/basis/absorb_cov_pixart.pt",
        caches=f"{REPO}/models/pixart-sigma/calibration_dataset/caches/*.pt",
        out=f"{REPO}/models/pixart-sigma/absorb_basis"),
    "sana": dict(
        load=load_sana, dtype=torch.bfloat16,
        cov=f"{REPO}/models/sana-1.6b/basis/absorb_cov_sana.pt",
        caches=f"{REPO}/models/sana-1.6b/calibration_dataset/caches/*.pt",
        out=f"{REPO}/models/sana-1.6b/absorb_basis"),
    "sdxl-turbo": dict(
        load=lambda: load_sdxl("stabilityai/sdxl-turbo"), dtype=torch.float16,
        cov=f"{REPO}/models/sdxl-turbo/basis/absorb_cov_sdxl_linear.pt",
        caches=f"{DC}/datasets/torch.float16/sdxl-turbo/eulera4-g0/qdiff/s128/caches/*.pt",
        out=f"{REPO}/models/sdxl-turbo/absorb_basis"),
    "sdxl-base": dict(
        load=lambda: load_sdxl("stabilityai/stable-diffusion-xl-base-1.0"),
        dtype=torch.float16,
        cov=f"{REPO}/models/sdxl-base/basis/absorb_cov_sdxl_linear.pt",
        caches=f"{DC}/datasets/torch.float16/sdxl/euler30-g5.0/qdiff/s128/caches/*.pt",
        out=f"{REPO}/models/sdxl-base/absorb_basis"),
}


def get(root, path):
    m = root
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def smooth_key(lp):
    if ".attn1.to_k" in lp or ".attn1.to_v" in lp:
        return lp.replace(".to_k", ".to_q").replace(".to_v", ".to_q")
    return lp


def geo_normalize(s):
    s = s.clamp(min=1e-6)
    return s / s.log().mean().exp()


def to_dev_fn(dtype):
    def to_dev(v):
        if isinstance(v, torch.Tensor):
            return v.to("cuda", dtype) if v.is_floating_point() else v.to("cuda")
        return v
    return to_dev


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--amax-files", type=int, default=-1,
                    help="cap the amax pass (-1 = all caches, full-stream)")
    ap.add_argument("--gain-files", type=int, default=16)
    args = ap.parse_args()
    cfg = FAMILIES[args.family]

    model, table = cfg["load"]()
    model.eval()
    model.requires_grad_(False)
    cov = torch.load(cfg["cov"], map_location="cpu", weights_only=False)
    to_dev = to_dev_fn(cfg["dtype"])

    files = sorted(glob.glob(cfg["caches"]))
    assert files, f"no caches: {cfg['caches']}"
    gain_files = files[:: max(1, len(files) // args.gain_files)][: args.gain_files]
    amax_files = files if args.amax_files < 0 else \
        files[:: max(1, len(files) // args.amax_files)][: args.amax_files]
    print(f"{args.family}: {len(table)} layers, {len(files)} caches "
          f"(amax pass {len(amax_files)}, gain pass {len(gain_files)})")

    def forward_files(fs, desc):
        for f in tqdm(fs, desc=desc, dynamic_ncols=True):
            d = torch.load(f, map_location="cpu", weights_only=False)
            model(*tree_map(to_dev, d["input_args"]),
                  **tree_map(to_dev, d["input_kwargs"]))

    # ---- pass 1: full-stream act amax, one hook per unique cov key ----------
    hook_mod, hook_layers = {}, {}
    for lp, ck in table:
        hook_layers.setdefault(ck, []).append(lp)
        hook_mod.setdefault(ck, get(model, lp))
    amax = {}
    handles = []
    for ck, mod in hook_mod.items():
        def mk(ck):
            def hook(m, a):
                v = rows(a[0]).abs().amax(dim=0).float()
                amax[ck] = v if ck not in amax else torch.maximum(amax[ck], v)
            return hook
        handles.append(mod.register_forward_pre_hook(mk(ck)))
    forward_files(amax_files, "amax")
    for h in handles:
        h.remove()

    # ---- W_res(lambda*) + closed-form s per hook ----------------------------
    sd = {k: v for k, v in model.state_dict().items()}
    wres = {}
    for lp, ck in tqdm(table, desc="hsvd", dynamic_ncols=True):
        W = sd[f"{lp}.weight"].to("cuda", torch.float32)
        if W.dim() == 4:  # 1x1 conv
            W = W.view(W.shape[0], W.shape[1])
        D, lu = hsvd_basis(W, cov[ck], args.rank, "cuda", damping=args.lam)
        wres[lp] = (W - lu @ D).to(cfg["dtype"])
        del W, D, lu
    del sd
    torch.cuda.empty_cache()

    vecs = {"rms": {}, "amax": {}}
    s_by_hook = {"rms": {}, "amax": {}}
    for ck, lps in hook_layers.items():
        Wcat = torch.cat([wres[lp].float() for lp in lps], dim=0)
        rms_x = cov[ck].diagonal().float().clamp(min=0).sqrt().cuda()
        rms_w = Wcat.pow(2).mean(dim=0).sqrt()
        amax_w = Wcat.abs().amax(dim=0)
        fam_s = {
            "rms": geo_normalize(rms_x.clamp(min=1e-6) / rms_w.clamp(min=1e-6)),
            "amax": geo_normalize(amax[ck].cuda().clamp(min=1e-6) / amax_w.clamp(min=1e-6)),
        }
        del Wcat
        for fam, s in fam_s.items():
            s = s.to(cfg["dtype"]).float()  # runtime storage precision
            s_by_hook[fam][ck] = s.to("cuda", cfg["dtype"])
            for lp in lps:
                vecs[fam][smooth_key(lp)] = s.cpu()

    # ---- pass 2: per-layer gains for both families over strided caches ------
    acc = {lp: [0.0, 0.0, 0.0] for lp, _ in table}  # e0, e1_rms, e1_amax
    handles = []
    for lp, ck in table:
        def mk(lp, ck):
            Wr = wres[lp]
            Wr_s = {f: Wr.float().mul(s_by_hook[f][ck].float().unsqueeze(0))
                    .to(cfg["dtype"]) for f in ("rms", "amax")}

            def hook(m, a):
                x = rows(a[0]).to(cfg["dtype"])
                yt = (x @ Wr.t()).float()
                acc[lp][0] += float(((act_fp4_sim(x) @ Wr.t()).float() - yt)
                                    .pow(2).sum())
                for i, f in enumerate(("rms", "amax")):
                    s = s_by_hook[f][ck]
                    e = ((act_fp4_sim(x / s) @ Wr_s[f].t()).float() - yt).pow(2).sum()
                    acc[lp][1 + i] += float(e)
            return hook
        handles.append(get(model, lp).register_forward_pre_hook(mk(lp, ck)))
    forward_files(gain_files, "gain")
    for h in handles:
        h.remove()

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    for i, fam in enumerate(("rms", "amax")):
        gains = {}
        for lp, _ in table:
            e0, *e1 = acc[lp]
            gains[lp] = (10 * math.log10(e0 / e1[i])
                         if e0 > 0 and e1[i] > 0 else 0.0)
        torch.save(vecs[fam], out / f"selfsmooth_{fam}.pt")
        with open(out / f"selfsmooth_gain_{fam}.json", "w") as f:
            json.dump(gains, f, indent=2)
        v = sorted(gains.values())
        print(f"{fam}: {len(v)} layers, median {statistics.median(v):+.2f} dB, "
              f">+0.3: {sum(x > 0.3 for x in v)}, <-0.1: {sum(x < -0.1 for x in v)}")
    print("saved:", out)


if __name__ == "__main__":
    main()
