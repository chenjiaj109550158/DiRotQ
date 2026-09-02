"""PLAN_SELFSMOOTH: collect the adaLN modulation input silu(temb) for FLUX
from OUR calibration prompts only (qdiff-128 CLIP-L pooled embeddings x a
timestep grid; guidance fixed for dev). No transformer forward needed: temb
depends only on (timestep[, guidance], pooled_projection), so we instantiate
just time_text_embed and replay it.

Usage (svdquant env, repo root):
  python absorb_basis/collect_temb_flux.py \
      --model-id black-forest-labs/FLUX.1-schnell \
      --prompts models/flux-schnell/calib_prompts.yaml \
      --out models/flux-schnell/basis/adanorm_temb.pt
  python absorb_basis/collect_temb_flux.py \
      --model-id black-forest-labs/FLUX.1-dev --guidance 3.5 \
      --prompts models/flux-schnell/calib_prompts.yaml \
      --out models/flux-dev/basis/adanorm_temb.pt
Output: {"x": [N,3072] bf16 silu(temb) samples, "diag": [3072] fp32 sum x^2}
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from absorb_basis.build_checkpoint import load_transformer_state_dict


def load_prompts(path, n=128):
    d = yaml.safe_load(open(path))
    if isinstance(d, dict):
        # deepcompressor benchmark yaml: {name: prompt} or {prompts: [...]}
        d = d.get("prompts", d)
        if isinstance(d, dict):
            d = [v if isinstance(v, str) else v.get("prompt")
                 for _, v in sorted(d.items())]
    assert isinstance(d, list) and len(d) >= n, f"bad prompts yaml {path}"
    return [p if isinstance(p, str) else p["prompt"] for p in d[:n]]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guidance", type=float, default=None,
                    help="guidance scale for guidance-distilled models (dev)")
    ap.add_argument("--num-timesteps", type=int, default=32,
                    help="timestep grid over [0,1] (superset coverage of any "
                         "scheduler; prompts stay the qdiff-128 calib set)")
    args = ap.parse_args()

    from diffusers.models.embeddings import (
        CombinedTimestepGuidanceTextProjEmbeddings,
        CombinedTimestepTextProjEmbeddings,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    prompts = load_prompts(args.prompts)
    print(f"{len(prompts)} calib prompts")

    tok = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder",
                                       torch_dtype=torch.bfloat16).to("cuda")
    te.eval()
    pooled = []
    for i in range(0, len(prompts), 16):
        batch = tok(prompts[i:i + 16], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt")
        out = te(batch.input_ids.to("cuda"), output_hidden_states=False)
        pooled.append(out.pooler_output)
    pooled = torch.cat(pooled)  # [128, 768] bf16
    del te
    torch.cuda.empty_cache()

    sd = load_transformer_state_dict(args.model_id)
    guided = any(k.startswith("time_text_embed.guidance_embedder") for k in sd)
    cls = (CombinedTimestepGuidanceTextProjEmbeddings if guided
           else CombinedTimestepTextProjEmbeddings)
    mod = cls(embedding_dim=3072, pooled_projection_dim=768)
    w = {k[len("time_text_embed."):]: v for k, v in sd.items()
         if k.startswith("time_text_embed.")}
    mod.load_state_dict(w)
    mod = mod.to("cuda", torch.bfloat16).eval()
    if guided:
        assert args.guidance is not None, "guidance-distilled model needs --guidance"

    # FluxTransformer2DModel.forward scales timestep (and guidance) by 1000
    ts = torch.linspace(0.0, 1.0, args.num_timesteps, device="cuda") * 1000
    xs = []
    for t in ts:
        tt = t.expand(pooled.shape[0]).to(torch.bfloat16)
        if guided:
            g = torch.full_like(tt, args.guidance * 1000)
            emb = mod(tt, g, pooled)
        else:
            emb = mod(tt, pooled)
        xs.append(torch.nn.functional.silu(emb))
    x = torch.cat(xs)  # [128*T, 3072]
    out = {"x": x.cpu(), "diag": x.float().pow(2).sum(0).cpu()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {tuple(x.shape)} silu(temb) samples (guided={guided}) -> {args.out}")
    print("diag range:", out["diag"].min().item(), out["diag"].max().item())


if __name__ == "__main__":
    main()
