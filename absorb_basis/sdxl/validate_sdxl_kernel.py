"""End-to-end kernel validation: whole-unet forward QSNR vs fp16 on calib
caches. Usage: python validate_sdxl_kernel.py <kernel.pt>"""
import glob
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deepcompressor.utils.common import tree_map
from diffusers import UNet2DConditionModel

from absorb_basis.sdxl.run_sdxl_kernel_generate import inject_kernel

ckpt = sys.argv[1]


def load_unet():
    m = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet",
                                             torch_dtype=torch.float16, variant="fp16").to("cuda")
    m.eval()
    m.requires_grad_(False)
    return m


files = sorted(glob.glob("/home/dev/deepcompressor/examples/diffusion/datasets/"
                         "torch.float16/sdxl-turbo/eulera4-g0/qdiff/s128/caches/*.pt"))[100:104]


def to_dev(v):
    if isinstance(v, torch.Tensor):
        return v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda")
    return v


def run(m):
    outs = []
    with torch.no_grad():
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            y = m(*tree_map(to_dev, d["input_args"]), **tree_map(to_dev, d["input_kwargs"]))[0]
            outs.append(y.float().cpu())
    return torch.cat(outs)


m = load_unet()
y_fp = run(m)
del m
torch.cuda.empty_cache()
m = load_unet()
pk = torch.load(ckpt, map_location="cpu", weights_only=False)
nl, nc = inject_kernel(m, pk, torch_dtype=torch.float16)
y_k = run(m)
err = (y_k - y_fp).pow(2).mean()
ref = y_fp.pow(2).mean()
print(json.dumps({"ckpt": ckpt.split("/")[-1], "linears": nl, "convs": nc,
                  "unet_output_qsnr_db": round(10 * math.log10(float(ref / err)), 2),
                  "nan": bool(torch.isnan(y_k).any())}))
