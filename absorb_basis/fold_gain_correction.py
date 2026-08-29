"""PLAN_ROUND2 C: folded gain correction (PTQD-style correlated-noise fix,
zero runtime cost).

Measured fact: each quantized layer's output is y_q ~= k*y + n with a
systematic k<1 (act-quant deadzone shrink) — the FID "softening" mechanism.
This tool estimates k per layer PER OUTPUT CHANNEL on the calibration caches
using the REAL kernel modules, then folds 1/k into the checkpoint:
  - wcscales  <- packed(1/k)   (scales the 4-bit main branch)
  - lora_up   <- lora_up / k   (scales the low-rank branch)
  - bias unchanged (exact, must not be rescaled)
so the deployed layer computes (y_q - b)/k + b with no extra work.

Usage (svdquant env, repo root):
  python absorb_basis/fold_gain_correction.py pixart <in.pt> <out.pt>
  python absorb_basis/fold_gain_correction.py sana   <in.pt> <out.pt>
"""

import glob
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from absorb_basis.build_checkpoint import pack_perm_vector

KIND = sys.argv[1]
CKPT_IN, CKPT_OUT = sys.argv[2], sys.argv[3]
NUM_CACHES = int(sys.argv[4]) if len(sys.argv) > 4 else 32
CLAMP = (0.95, 1.05)
EPS_FRAC = 0.05  # regularizer strength relative to mean channel energy

if KIND == "pixart":
    from diffusers import PixArtTransformer2DModel as TF

    from absorb_basis.pixart.run_pixart_kernel_generate import inject_kernel
    MODEL = ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", {}, torch.float16)
    CACHES = ("/home/dev/deepcompressor/examples/diffusion/datasets/torch.float16/"
              "pixart-sigma/dpm20-g4.5/qdiff/s128/caches")
else:
    from diffusers import SanaTransformer2DModel as TF

    from absorb_basis.sana.run_sana_kernel_generate import inject_kernel
    MODEL = ("Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632",
             {"variant": "bf16"}, torch.bfloat16)
    CACHES = ("/home/dev/deepcompressor/examples/diffusion/datasets/torch.bfloat16/"
              "sana-1.6b-1024px-bf16-ch5632/flowdpm20-g4.5/qdiff/s128/caches")

dt = MODEL[2]


def get(root, path):
    m = root
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def rows(x):
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return x.reshape(-1, x.shape[-1])


@torch.no_grad()
def main():
    from deepcompressor.utils.common import tree_map

    tf = TF.from_pretrained(MODEL[0], subfolder="transformer", torch_dtype=dt,
                            **MODEL[1]).to("cuda")
    tf.eval()
    tf.requires_grad_(False)
    ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    # keep the fp weights and biases of the quantized layers before injection
    fpw = {}
    for lp in ckpt:
        mod = get(tf, lp)
        W = mod.weight.detach()
        if W.dim() == 4:
            W = W.view(W.shape[0], W.shape[1])
        fpw[lp] = (W.clone(), mod.bias.detach().clone() if mod.bias is not None else None)
    inject_kernel(tf, ckpt)

    acc = {lp: [torch.zeros(fpw[lp][0].shape[0], dtype=torch.float64, device="cuda"),
                torch.zeros(fpw[lp][0].shape[0], dtype=torch.float64, device="cuda")]
           for lp in ckpt}
    hooks = []
    for lp in ckpt:
        mod = get(tf, lp)
        W_fp, b_fp = fpw[lp]

        def mk(lp, W_fp, b_fp):
            def hook(m, a, out):
                x = rows(a[0])
                y_k = rows(out).float()
                if b_fp is not None:
                    y_k = y_k - b_fp.float()
                y_f = (x @ W_fp.t()).float()
                acc[lp][0] += (y_k * y_f).sum(0).double()
                acc[lp][1] += (y_f * y_f).sum(0).double()
            return hook

        hooks.append(mod.register_forward_hook(mk(lp, W_fp, b_fp)))

    files = sorted(glob.glob(CACHES + "/*.pt"))
    step = max(1, len(files) // NUM_CACHES)
    files = files[::step][:NUM_CACHES]

    def to_dev(v):
        if isinstance(v, torch.Tensor):
            return v.to("cuda", dt) if v.is_floating_point() else v.to("cuda")
        return v

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        tf(*tree_map(to_dev, d["input_args"]), **tree_map(to_dev, d["input_kwargs"]))
    for h in hooks:
        h.remove()

    # NOTE: the packed lora_up rows are tile-interleaved (verified: neither
    # model-order nor perm-order row scaling reproduces per-channel output
    # scaling), so the lora half of the correction must be folded PRE-PACK.
    # This tool therefore only exports k (model channel order); the builders
    # consume it via --gain-k and fold 1/k into lora_up before packing +
    # wcscales after packing.
    ks, kout = [], {}
    for lp in ckpt:
        qy, yy = acc[lp]
        eps = float(yy.mean()) * EPS_FRAC
        k = ((qy + eps) / (yy + eps)).clamp(*CLAMP).float()   # [oc], model order
        ks.append(float(k.mean()))
        kout[lp] = k.cpu()
    torch.save(kout, CKPT_OUT)
    import statistics
    print(json.dumps({"k_file": CKPT_OUT, "layers": len(kout),
                      "mean_k_median": round(statistics.median(ks), 5),
                      "mean_k_min": round(min(ks), 5)}))


if __name__ == "__main__":
    main()
