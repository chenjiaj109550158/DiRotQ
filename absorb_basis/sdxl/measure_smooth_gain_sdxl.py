"""SDXL per-layer smoothing gain on the lambda* H-SVD residuals (PLAN_ROUND2
S selection). Usage: python measure_smooth_gain_sdxl.py <lambda>"""
import glob
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deepcompressor.utils.common import tree_map
from diffusers import UNet2DConditionModel

from absorb_basis.build_checkpoint import hsvd_basis
from absorb_basis.pixart.run_pixart_sim_generate import act_fp4_sim
from absorb_basis.sdxl.build_sdxl_kernel import linear_table

LAM = float(sys.argv[1])
unet = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet",
                                            torch_dtype=torch.float16, variant="fp16").to("cuda")
unet.eval()
unet.requires_grad_(False)
lt = linear_table(unet)
sd = {k: v for k, v in unet.state_dict().items()}
cov = torch.load("/home/dev/DiRotQ/models/sdxl-turbo/basis/absorb_cov_sdxl_linear.pt",
                 map_location="cpu", weights_only=False)
sm = torch.load("/home/dev/DiRotQ/models/sdxl-turbo/svdq_model_dump/smooth.pt",
                map_location="cpu", weights_only=False)


def smooth_key(lp):
    if ".attn1.to_k" in lp or ".attn1.to_v" in lp:
        return lp.replace(".to_k", ".to_q").replace(".to_v", ".to_q")
    return lp


layers = {}
for lp, ck in lt:
    skey = smooth_key(lp)
    if skey not in sm:
        continue
    W = sd[f"{lp}.weight"].to("cuda", torch.float32)
    D, lu = hsvd_basis(W, cov[ck], 32, "cuda", damping=LAM)
    Wres = (W - lu @ D).half()
    s = sm[skey].to("cuda", torch.float16)
    layers[lp] = {"Wres": Wres, "Wres_s": (Wres.float() * s.float().unsqueeze(0)).half(),
                  "s": s, "acc": [0.0, 0.0, 0.0]}
del sd


def get(root, path):
    m = root
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


hooks = []
for lp, info in layers.items():
    def mk(info):
        def hook(m, a):
            x = a[0].reshape(-1, a[0].shape[-1])
            with torch.no_grad():
                yt = (x @ info["Wres"].t()).float()
                e0 = ((act_fp4_sim(x) @ info["Wres"].t()).float() - yt).pow(2).sum()
                e1 = ((act_fp4_sim(x / info["s"]) @ info["Wres_s"].t()).float() - yt).pow(2).sum()
                info["acc"][0] += float(yt.pow(2).sum())
                info["acc"][1] += float(e0)
                info["acc"][2] += float(e1)
        return hook
    hooks.append(get(unet, lp).register_forward_pre_hook(mk(info)))

files = sorted(glob.glob("/home/dev/deepcompressor/examples/diffusion/datasets/"
                         "torch.float16/sdxl-turbo/eulera4-g0/qdiff/s128/caches/*.pt"))[100:116]


def to_dev(v):
    if isinstance(v, torch.Tensor):
        return v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda")
    return v


with torch.no_grad():
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        unet(*tree_map(to_dev, d["input_args"]), **tree_map(to_dev, d["input_kwargs"]))
for h in hooks:
    h.remove()
gains = {}
for lp, info in layers.items():
    y, e0, e1 = info["acc"]
    gains[lp] = 10 * math.log10(e0 / e1) if e0 > 0 and e1 > 0 else 0.0
json.dump(gains, open("/home/dev/DiRotQ/models/sdxl-turbo/absorb_basis/smooth_gain.json", "w"),
          indent=2)
import statistics
v = sorted(gains.values())
print(f"sdxl: {len(gains)} layers, median {statistics.median(v):+.2f} dB, "
      f">+0.3: {sum(x > 0.3 for x in v)}, <-0.1: {sum(x < -0.1 for x in v)}")
