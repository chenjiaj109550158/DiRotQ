"""Collect per-layer input covariances + top-r PCA bases for DiRotQ-absorb-basis
on FLUX.1-schnell.

For every W4A4 linear that DiRotQ-absorb-basis re-quantizes (all K=3072 inputs:
qkv/out_proj/mlp_fc1 in double and single blocks), accumulate the uncentered
input covariance H = (2/n) * X^T X over the calibration caches. H serves two
purposes:
  1. GPTQ Hessian for quantizing the residual weight Q4((I - U U^T)-projected W)
  2. PCA basis: top-r eigenvectors U_r of H form the 16-bit low-rank branch
     lora_down = U_r, lora_up = W @ U_r.

K=12288 down-projections (mlp_fc2) are NOT collected — those layers are copied
verbatim from the official SVDQuant checkpoint.

Two passes over the calibration caches keep VRAM in budget on a 32 GB card
(transformer 23.8 GiB bf16 + fp32 GPU accumulators):
  pass "double": 6 hooks x 19 blocks = 114 accumulators (4.3 GiB)
  pass "single": 2 hooks x 38 blocks =  76 accumulators (2.9 GiB)

Output: a single .pt with, per key:
  {key}                : evec top-r  [3072, r] float32 (ascending eig order,
                          i.e. columns are the r LARGEST-eigenvalue vectors)
  {key}.eigenvalues    : all 3072 eigenvalues float32 (ascending)
  {key}.H              : normalized covariance [3072, 3072] float32 (GPTQ Hessian)

Keys follow models/flux-schnell/basis_utils.py naming:
  layer.{i}.img_attn / txt_attn / img_attn.value / txt_attn.value / img_ffn / txt_ffn
  single.{i}.attn / attn_out.value
"""

import argparse
import glob
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_hook_specs(transformer, which: str):
    """Return list of (key, module, slice_end) to hook for the given pass.

    slice_end: if not None, only x[..., :slice_end] enters the covariance
    (used for the fused single-block proj_out: attn half is the first 3072).
    """
    specs = []
    if which == "double":
        for i, blk in enumerate(transformer.transformer_blocks):
            specs += [
                (f"layer.{i}.img_attn", blk.attn.to_q, None),
                (f"layer.{i}.txt_attn", blk.attn.add_q_proj, None),
                (f"layer.{i}.img_attn.value", blk.attn.to_out[0], None),
                (f"layer.{i}.txt_attn.value", blk.attn.to_add_out, None),
                (f"layer.{i}.img_ffn", blk.ff.net[0].proj, None),
                (f"layer.{i}.txt_ffn", blk.ff_context.net[0].proj, None),
            ]
    elif which == "single":
        for i, blk in enumerate(transformer.single_transformer_blocks):
            specs += [
                # qkv_proj and mlp_fc1 (proj_mlp) share this input
                (f"single.{i}.attn", blk.attn.to_q, None),
                # fused proj_out input = concat([attn_out (3072), mlp_act (12288)])
                (f"single.{i}.attn_out.value", blk.proj_out, 3072),
            ]
    else:
        raise ValueError(which)
    return specs


@torch.no_grad()
def run_pass(transformer, cache_files, which, batch_size, device):
    specs = build_hook_specs(transformer, which)
    H = {key: torch.zeros(3072, 3072, dtype=torch.float32, device=device)
         for key, _, _ in specs}
    cnt = {key: 0 for key, _, _ in specs}

    hooks = []

    def make_hook(key, slice_end):
        def hook(module, args):
            x = args[0]
            if slice_end is not None:
                x = x[..., :slice_end]
            x = x.reshape(-1, x.shape[-1]).float()
            H[key].addmm_(x.t(), x)
            cnt[key] += x.shape[0]
        return hook

    for key, mod, slice_end in specs:
        hooks.append(mod.register_forward_pre_hook(make_hook(key, slice_end)))

    model_dtype = next(transformer.parameters()).dtype
    file_batches = [cache_files[i:i + batch_size]
                    for i in range(0, len(cache_files), batch_size)]

    def load_batch(files):
        return [torch.load(f, map_location="cpu", weights_only=False) for f in files]

    def forward_batch(batch_data):
        latents = torch.cat(
            [d["input_args"][0].to(model_dtype) for d in batch_data], dim=0
        ).to(device)
        kwargs = {}
        for k, v in batch_data[0]["input_kwargs"].items():
            if isinstance(v, torch.Tensor):
                vals = [d["input_kwargs"][k] for d in batch_data]
                if v.ndim >= 1 and v.shape[0] == 1:
                    kwargs[k] = torch.cat(
                        [x.to(model_dtype) if x.is_floating_point() else x for x in vals],
                        dim=0,
                    ).to(device)
                else:
                    kwargs[k] = (v.to(model_dtype) if v.is_floating_point() else v).to(device)
            else:
                kwargs[k] = v
        transformer(latents, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        io_queue = deque()
        it = iter(file_batches)
        for _ in range(2):
            try:
                io_queue.append(pool.submit(load_batch, next(it)))
            except StopIteration:
                break
        for _ in tqdm(range(len(file_batches)), desc=f"pass:{which}", dynamic_ncols=True):
            try:
                io_queue.append(pool.submit(load_batch, next(it)))
            except StopIteration:
                pass
            forward_batch(io_queue.popleft().result())

    for h in hooks:
        h.remove()

    out = {}
    for key in H:
        assert cnt[key] > 0, f"no data for {key}"
        out[key] = (H[key] * (2.0 / cnt[key])).cpu()
    del H
    torch.cuda.empty_cache()
    return out


def eigh_topr(H_cpu: torch.Tensor, rank: int, damping: float = 0.01, device="cuda"):
    """fp64 GPU eigendecomposition, mirroring basis_utils._eigh."""
    H = H_cpu.to(device=device, dtype=torch.float64)
    H = H + damping * H.diagonal().mean() * torch.eye(
        H.shape[0], dtype=H.dtype, device=H.device
    )
    evals, evec = torch.linalg.eigh(H)
    return evec[:, -rank:].float().cpu(), evals.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--calib-dir", default="models/flux-schnell/calibration_dataset/caches")
    ap.add_argument("--out", default="models/flux-schnell/basis/absorb_cov_basis.pt")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-files", type=int, default=-1)
    args = ap.parse_args()

    cache_files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    assert cache_files, f"no caches in {args.calib_dir}"
    if args.max_files > 0:
        cache_files = cache_files[: args.max_files]
    print(f"{len(cache_files)} calibration caches")

    from diffusers import FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    transformer.eval()
    transformer.requires_grad_(False)

    result = {}
    for which in ("double", "single"):
        covs = run_pass(transformer, cache_files, which, args.batch_size, "cuda")
        for key, Hc in covs.items():
            result[f"{key}.H"] = Hc
        del covs

    del transformer
    torch.cuda.empty_cache()

    print("Eigendecompositions (fp64, GPU)...")
    keys = [k[: -len(".H")] for k in result if k.endswith(".H")]
    for key in tqdm(keys, dynamic_ncols=True):
        evec_top, evals = eigh_topr(result[f"{key}.H"], args.rank)
        result[key] = evec_top
        result[f"{key}.eigenvalues"] = evals

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(result, args.out)
    print(f"saved {len(result)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
