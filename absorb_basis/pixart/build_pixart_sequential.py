"""Sequential (quantization-error-propagating) calibration for
DiRotQ-absorb-basis on PixArt-Sigma (PLAN item A).

Instead of calibrating every layer on the full-precision model's activations,
blocks are processed in order: block i's input covariances are computed from
the activation stream that has already passed through the QUANTIZED blocks
0..i-1, so H-SVD and GPTQ compensate upstream quantization error.

Implementation:
  1. One pass over all calibration caches, running the model front-end and
     capturing the inputs of transformer_blocks[0] (hidden + shared kwargs).
     PixArt passes identical kwargs to every block, so the captured kwargs
     serve all blocks. The hidden stream ([N, 4096, 1152] fp16, ~48 GB for
     the full 5120-cache set) does not fit in RAM, so it lives in two
     alternating disk memmaps under --scratch-dir; the small kwargs stay in
     RAM.
  2. For each block: (a) forward the stored stream through the fp16 block
     with covariance hooks; (b) hsvd+GPTQ its 8 quantized linears (cross-attn
     KV stays fp16, aligned with SVDQuant's attn_add skip) and swap them for
     SimW4A4Linear (weight AND activation quantization simulated, nunchaku
     semantics); (c) forward the stream through the quantized block into the
     other memmap to produce the next block's input.
  3. Save the same sim-weights dict format as build_pixart_sim.py.

Run in the svdquant env:
  python build_pixart_sequential.py --calib-dir <caches> --out <sim.pt> \
      --scratch-dir <dir with ~100 GB free>
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from absorb_basis.build_checkpoint import hsvd_basis, quantize_residual
from absorb_basis.pixart.run_pixart_sim_generate import SimW4A4Linear

BLOCK_LAYERS = [  # (module path within block, cov key)
    ("attn1.to_q", "attn1_qkv"),
    ("attn1.to_k", "attn1_qkv"),
    ("attn1.to_v", "attn1_qkv"),
    ("attn1.to_out.0", "attn1_out"),
    ("attn2.to_q", "attn2_q"),
    ("attn2.to_out.0", "attn2_out"),
    ("ff.net.0.proj", "ffn_up"),
    ("ff.net.2", "ffn_down"),
]
HOOK_POINTS = [  # (cov key, module path of the linear whose input we tap, dim)
    ("attn1_qkv", "attn1.to_q", 1152),
    ("attn1_out", "attn1.to_out.0", 1152),
    ("attn2_q", "attn2.to_q", 1152),
    ("attn2_out", "attn2.to_out.0", 1152),
    ("ffn_up", "ff.net.0.proj", 1152),
    ("ffn_down", "ff.net.2", 4608),
]


class _StopForward(Exception):
    pass


def get_submodule(root, path):
    m = root
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def set_submodule(root, path, mod):
    parts = path.split(".")
    parent = get_submodule(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    if parts[-1].isdigit():
        parent[int(parts[-1])] = mod
    else:
        setattr(parent, parts[-1], mod)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch-dir", required=True,
                    help="holds two [N,tokens,1152] fp16 memmaps (~2x9.4MB per cache)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--hsvd-damping", type=float, default=0.01)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16, help="caches per block-forward chunk")
    ap.add_argument("--max-files", type=int, default=-1)
    args = ap.parse_args()

    from deepcompressor.utils.common import tree_map
    from diffusers import PixArtTransformer2DModel

    transformer = PixArtTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float16
    ).to("cuda")
    transformer.eval()
    transformer.requires_grad_(False)

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    if args.max_files > 0:
        files = files[: args.max_files]
    assert files
    n = len(files)
    print(f"{n} calibration caches")

    # ---- stage 1: capture block-0 inputs (hidden -> memmap, kwargs -> RAM) --
    os.makedirs(args.scratch_dir, exist_ok=True)
    stream_paths = [os.path.join(args.scratch_dir, f"seq_stream_{i}.f16") for i in (0, 1)]
    state = {"stream": None, "enc": None, "mask": None, "temb": None, "i": 0}

    def cap_hook(module, hargs, hkwargs):
        assert not hargs[1:] and "encoder_hidden_states" in hkwargs, (
            f"unexpected block call convention: args={len(hargs)}, kwargs={list(hkwargs)}"
        )
        i = state["i"]
        h = hargs[0]
        if state["stream"] is None:  # first file: allocate from observed shapes
            state["tok"], state["dim"] = h.shape[1], h.shape[2]
            state["stream"] = np.memmap(stream_paths[0], dtype=np.float16, mode="w+",
                                        shape=(n, h.shape[1], h.shape[2]))
            for k in ("encoder_hidden_states", "encoder_attention_mask", "timestep"):
                v = hkwargs.get(k)
                if v is not None:
                    state[{"encoder_hidden_states": "enc", "encoder_attention_mask": "mask",
                           "timestep": "temb"}[k]] = torch.empty(n, *v.shape[1:], dtype=torch.float16)
            extras = {k: v for k, v in hkwargs.items()
                      if k not in ("encoder_hidden_states", "encoder_attention_mask", "timestep")
                      and v is not None and not (isinstance(v, dict) and not v)}
            assert not extras, f"unhandled non-None block kwargs: {list(extras)}"
        assert h.shape[0] == 1 and h.shape[1] == state["tok"]
        state["stream"][i] = h[0].half().cpu().numpy()
        for k, s in (("encoder_hidden_states", "enc"), ("encoder_attention_mask", "mask"),
                     ("timestep", "temb")):
            v = hkwargs.get(k)
            if v is not None:
                state[s][i] = v[0].half().cpu()
        raise _StopForward

    hcap = transformer.transformer_blocks[0].register_forward_pre_hook(cap_hook, with_kwargs=True)

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to("cuda", torch.float16) if x.is_floating_point() else x.to("cuda")
        return x

    for f in tqdm(files, desc="capture", dynamic_ncols=True):
        data = torch.load(f, map_location="cpu", weights_only=False)
        try:
            transformer(*tree_map(to_dev, data["input_args"]),
                        **tree_map(to_dev, data["input_kwargs"]))
        except _StopForward:
            pass
        state["i"] += 1
    hcap.remove()
    assert state["i"] == n and state["stream"] is not None
    state["stream"].flush()
    cur = state["stream"]
    print(f"captured {n} block-0 inputs: tokens={state['tok']}, "
          f"enc={None if state['enc'] is None else tuple(state['enc'].shape)}, "
          f"mask={None if state['mask'] is None else tuple(state['mask'].shape)}, "
          f"temb={None if state['temb'] is None else tuple(state['temb'].shape)}")

    # ---- stage 2: sequential per-block calibrate -> quantize -> propagate --
    sim_out, qsnrs = {}, {}
    num_blocks = len(transformer.transformer_blocks)
    t0 = time.time()
    for bi in tqdm(range(num_blocks), desc="blocks", dynamic_ncols=True):
        blk = transformer.transformer_blocks[bi]

        # (a) covariances from the current (quantized-prefix) stream
        H = {key: torch.zeros(d, d, dtype=torch.float32, device="cuda")
             for key, _, d in HOOK_POINTS}
        cnt = {key: 0 for key, _, _ in HOOK_POINTS}
        hooks = []

        def mk(key):
            def hook(module, hargs):
                x = hargs[0].reshape(-1, hargs[0].shape[-1]).float()
                H[key].addmm_(x.t(), x)
                cnt[key] += x.shape[0]
            return hook

        for key, mpath, _ in HOOK_POINTS:
            hooks.append(get_submodule(blk, mpath).register_forward_pre_hook(mk(key)))

        def block_forward_all(dst):
            for c0 in range(0, n, args.chunk):
                c1 = min(c0 + args.chunk, n)
                hs = torch.from_numpy(np.ascontiguousarray(cur[c0:c1])).to("cuda")
                kw = {"encoder_hidden_states": state["enc"][c0:c1].to("cuda")}
                if state["mask"] is not None:
                    kw["encoder_attention_mask"] = state["mask"][c0:c1].to("cuda")
                if state["temb"] is not None:
                    kw["timestep"] = state["temb"][c0:c1].to("cuda")
                y = blk(hs, **kw)
                y = y[0] if isinstance(y, tuple) else y
                if dst is not None:
                    dst[c0:c1] = y.half().cpu().numpy()

        block_forward_all(dst=None)
        for hk in hooks:
            hk.remove()

        # (b) quantize the block's 8 linears on the propagated statistics
        for mpath, ckey in BLOCK_LAYERS:
            lin = get_submodule(blk, mpath)
            W = lin.weight.to(torch.float32)
            Hc = (H[ckey] * (2.0 / cnt[ckey])).cpu()
            D, lora_up = hsvd_basis(W, Hc, args.rank, "cuda", damping=args.hsvd_damping)
            W_res = W - lora_up @ D
            W_q, _, _ = quantize_residual(
                W_res, Hc.to("cuda"), "plain", args.group_size, "cuda",
                gptq=True, damp_pct=args.damp, block_size=args.block_size,
            )
            full = f"transformer_blocks.{bi}.{mpath}"
            sim_out[full] = {
                "W_q": W_q.half().cpu(),
                "lora_down": D.t().half().cpu(),
                "lora_up": lora_up.half().cpu(),
            }
            W_hat = W_q + lora_up @ D
            err = (W_hat - W).pow(2).sum()
            qsnrs[full] = (10.0 * torch.log10(W.pow(2).sum() / err.clamp(min=1e-20))).item()
            t = sim_out[full]
            set_submodule(
                blk, mpath,
                SimW4A4Linear(
                    t["W_q"].to("cuda", torch.float16),
                    t["lora_down"].to("cuda", torch.float16),
                    t["lora_up"].to("cuda", torch.float16),
                    lin.bias.detach().clone() if lin.bias is not None else None,
                ),
            )
        del H

        # (c) propagate the stream through the quantized block
        if bi < num_blocks - 1:
            nxt = np.memmap(stream_paths[(bi + 1) % 2], dtype=np.float16, mode="w+",
                            shape=(n, state["tok"], state["dim"]))
            block_forward_all(dst=nxt)
            cur = nxt
        torch.cuda.empty_cache()

    print(f"sequential quantization done in {time.time()-t0:.0f}s")
    worst = sorted(qsnrs.items(), key=lambda kv: kv[1])
    for k, v in worst[:5]:
        print(f"  {v:6.2f} dB  {k}")
    print(f"median weight-QSNR: {worst[len(worst)//2][1]:.2f} dB")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(sim_out, args.out)
    with open(args.out + ".qsnr.json", "w") as f:
        json.dump(qsnrs, f, indent=2)
    print("saved:", args.out)
    for p in stream_paths:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
