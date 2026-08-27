"""Build a nunchaku-format DiRotQ-absorb-basis checkpoint for FLUX.1-schnell.

Method (DiRotQ-absorb-basis):
  Original DiRotQ main branch:  Q4(X U_l R) @ Q4(R^T U_l^T W^T)
  Absorbed form:                Q4(X) @ Q4(W_res^T) + (X U_r)(W U_r)^T
  where U_r = top-r PCA eigenvectors of the layer-input covariance,
        W_res = W - (W U_r) U_r^T   (weight projected off the top-r subspace,
                                     GPTQ-quantized offline to NVFP4),
        the 16-bit low-rank branch (rank r=32, same as SVDQuant) is
        lora_down = U_r, lora_up = W U_r,
        and there is NO online rotation and NO smoothing (smooth = 1).

Checkpoint assembly:
  - Start from the official SVDQuant svdq-fp4_r32-flux.1-schnell.safetensors.
  - Replace, for every K=3072 W4A4 layer (double: qkv_proj, qkv_proj_context,
    out_proj, out_proj_context, mlp_fc1, mlp_context_fc1; single: qkv_proj,
    mlp_fc1, out_proj):
      qweight, wscales, wtscale (or wcscales for qkv), lora_down, lora_up,
      smooth, smooth_orig
  - Keep verbatim (SVDQuant method, per user spec): mlp_fc2 / mlp_context_fc2
    (down projections, K=12288), the W4A16 int4-g64 adaptive-norm linears,
    biases, and every unquantized tensor.

NVFP4 scales are two-level, mirroring deepcompressor:
  top scale  = amax / (6 * 448)     (per-channel for fused qkv -> wcscales,
                                     per-tensor otherwise -> wtscale, bf16)
  micro scale = e4m3(group16_amax / (6 * top))  -> wscales (fp8_e4m3)
GPTQ runs against the exact effective grid scale = micro(e4m3) * top.

Run in the dirotq env (no nunchaku needed):
  python absorb_basis/build_checkpoint.py \
      --cov models/flux-schnell/basis/absorb_cov_basis.pt \
      --out models/flux-schnell/absorb_basis/dirotq-absorb-basis-fp4_r32-flux.1-schnell.safetensors
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from safetensors import safe_open
from safetensors.torch import save_file

from speedup.nunchaku_pack import NunchakuWeightPacker, convert_to_nunchaku_w4x4y16
from utils.gptq_utils import _gptq_quantize_layer

E2M1_MAX = 6.0
E4M3_MAX = 448.0


# --------------------------------------------------------------------------
# Layer table: (nunchaku key prefix, diffusers weight sources, cov key, kind)
# kind: "qkv" -> per-channel top scale (wcscales); "plain" -> wtscale.
# For "slice" sources, take W[:, :3072] of the named weight.
# --------------------------------------------------------------------------

def layer_table(num_double: int, num_single: int):
    table = []
    for i in range(num_double):
        p = f"transformer_blocks.{i}"
        d = f"transformer_blocks.{i}"
        table += [
            (f"{p}.qkv_proj",
             [f"{d}.attn.to_q.weight", f"{d}.attn.to_k.weight", f"{d}.attn.to_v.weight"],
             f"layer.{i}.img_attn", "qkv", None),
            (f"{p}.qkv_proj_context",
             [f"{d}.attn.add_q_proj.weight", f"{d}.attn.add_k_proj.weight", f"{d}.attn.add_v_proj.weight"],
             f"layer.{i}.txt_attn", "qkv", None),
            (f"{p}.out_proj",
             [f"{d}.attn.to_out.0.weight"],
             f"layer.{i}.img_attn.value", "plain", None),
            (f"{p}.out_proj_context",
             [f"{d}.attn.to_add_out.weight"],
             f"layer.{i}.txt_attn.value", "plain", None),
            (f"{p}.mlp_fc1",
             [f"{d}.ff.net.0.proj.weight"],
             f"layer.{i}.img_ffn", "plain", None),
            (f"{p}.mlp_context_fc1",
             [f"{d}.ff_context.net.0.proj.weight"],
             f"layer.{i}.txt_ffn", "plain", None),
        ]
    for i in range(num_single):
        p = f"single_transformer_blocks.{i}"
        d = f"single_transformer_blocks.{i}"
        table += [
            (f"{p}.qkv_proj",
             [f"{d}.attn.to_q.weight", f"{d}.attn.to_k.weight", f"{d}.attn.to_v.weight"],
             f"single.{i}.attn", "qkv", None),
            (f"{p}.mlp_fc1",
             [f"{d}.proj_mlp.weight"],
             f"single.{i}.attn", "plain", None),  # shares input (and basis) with qkv
            (f"{p}.out_proj",
             [f"{d}.proj_out.weight"],
             f"single.{i}.attn_out.value", "plain", 3072),  # attn half of fused proj_out
        ]
    return table


def load_transformer_state_dict(model_id: str) -> dict:
    """Load the diffusers transformer weights (bf16) straight from safetensors."""
    from huggingface_hub import snapshot_download

    snap = snapshot_download(model_id, allow_patterns=["transformer/*"])
    sd = {}
    for f in sorted(Path(snap, "transformer").glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                sd[k] = fh.get_tensor(k)
    assert sd, f"no transformer weights under {snap}"
    return sd


def two_level_scales(W_res: torch.Tensor, group_size: int, per_channel: bool):
    """Compute (top, micro_e4m3, effective) scales for NVFP4.

    top:   [oc] (per_channel) or scalar tensor  — bf16-storable float32
    micro: [oc, ng] float32 holding exact e4m3 values
    eff:   [oc, ng] float32 = micro * top      — the actual dequant grid
    """
    oc, ic = W_res.shape
    ng = ic // group_size
    Wg = W_res.abs().reshape(oc, ng, group_size)
    gmax = Wg.amax(dim=-1)  # [oc, ng]
    if per_channel:
        top = (gmax.amax(dim=1) / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-8)  # [oc]
        top = top.to(torch.bfloat16).float()  # round to storable bf16 first
        micro = gmax / (E2M1_MAX * top.unsqueeze(1))
    else:
        top = (gmax.amax() / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-8).reshape(())
        top = top.to(torch.bfloat16).float()
        micro = gmax / (E2M1_MAX * top)
    micro = micro.clamp(max=E4M3_MAX)
    micro = micro.to(torch.float8_e4m3fn).float().clamp(min=2.0 ** -9)  # e4m3 grid, no zeros
    eff = micro * (top.unsqueeze(1) if per_channel else top)
    return top, micro, eff


def pack_top_scale_per_channel(packer: NunchakuWeightPacker, top: torch.Tensor):
    """Pack a per-channel top scale into the wcscales layout ([oc] bf16)."""
    t = top.to(torch.bfloat16).view(-1, 1, 1, 1)
    t = packer.pad_scale(t, group_size=-1)
    return packer.pack_scale(t, group_size=-1).view(-1)


@torch.no_grad()
def build_layer(W: torch.Tensor, H: torch.Tensor, U: torch.Tensor, kind: str,
                group_size: int, device: str, gptq: bool,
                damp_pct: float, block_size: int,
                valref: dict | None = None, valref_key: str = ""):
    """Returns dict of replacement tensors for one nunchaku layer."""
    W = W.to(device=device, dtype=torch.float32)
    U = U.to(device=device, dtype=torch.float32)  # [ic, r]
    oc, ic = W.shape

    lora_up = W @ U                      # [oc, r]
    W_res = W - lora_up @ U.t()          # [oc, ic]

    top, micro, eff = two_level_scales(W_res, group_size, per_channel=(kind == "qkv"))

    if gptq:
        W_q = _gptq_quantize_layer(
            W_res, H.to(device=device, dtype=torch.float32),
            bits=4, groupsize=group_size, sym=True,
            damp_pct=damp_pct, block_size=block_size, num_inv_tries=8,
            device=device, nvfp4=True, scales_override=eff,
        )
        assert W_q is not None, "GPTQ failed after retries"
    else:  # RTN on the same grid
        ng = ic // group_size
        Wg = W_res.reshape(oc, ng, group_size) / eff.unsqueeze(-1)
        from utils.quant_utils import round_to_nf4_codebook
        W_q = (round_to_nf4_codebook(Wg) * eff.unsqueeze(-1)).reshape(oc, ic)

    # normalize by top so the packer sees codes * micro (micro is what wscales stores)
    if kind == "qkv":
        W_n = (W_q / top.unsqueeze(1)).to(torch.bfloat16)
    else:
        W_n = (W_q / top).to(torch.bfloat16)

    packer = NunchakuWeightPacker(bits=4)
    qweight, wscales, _bias, smooth_packed, (ld, lu) = convert_to_nunchaku_w4x4y16(
        weight=W_n,
        scale=micro.to(torch.bfloat16),          # [oc, ng] -> micro-scale e4m3 path
        bias=None,
        smooth=torch.ones(ic, dtype=torch.bfloat16, device=device),
        lora=(U.t().to(torch.bfloat16),          # lora_down pre-pack [r, ic]
              lora_up.to(torch.bfloat16)),       # lora_up  pre-pack [oc, r]
        float_point=True,
    )

    out = {
        "qweight": qweight.cpu(),
        "wscales": wscales.cpu(),
        "lora_down": ld.cpu(),
        "lora_up": lu.cpu(),
        "smooth": torch.ones(ic, dtype=torch.bfloat16),
        "smooth_orig": torch.ones(ic, dtype=torch.bfloat16),
    }
    if kind == "qkv":
        out["wcscales"] = pack_top_scale_per_channel(packer, top).cpu()
    else:
        out["wtscale"] = top.reshape(1).to(torch.bfloat16).cpu()

    if valref is not None:
        valref[valref_key] = {
            "W_q": W_q.half().cpu(),          # dequantized residual on the kernel grid
            "U": U.half().cpu(),              # [ic, r] lora_down (unpacked)
            "lora_up": lora_up.half().cpu(),  # [oc, r] (unpacked)
        }

    # quantization SNR of the full layer (lora + quantized residual) vs W
    W_hat = W_q + lora_up @ U.t()
    err = (W_hat - W).pow(2).sum()
    qsnr = 10.0 * torch.log10(W.pow(2).sum() / err.clamp(min=1e-20))
    return out, qsnr.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--official", default=None,
                    help="path to svdq-fp4_r32-flux.1-schnell.safetensors "
                         "(default: resolve from HF cache)")
    ap.add_argument("--cov", default="models/flux-schnell/basis/absorb_cov_basis.pt")
    ap.add_argument("--out", default="models/flux-schnell/absorb_basis/"
                                     "dirotq-absorb-basis-fp4_r32-flux.1-schnell.safetensors")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--rtn", action="store_true", help="RTN instead of GPTQ for the residual")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--num-double", type=int, default=19)
    ap.add_argument("--num-single", type=int, default=38)
    args = ap.parse_args()

    if args.official is None:
        from huggingface_hub import hf_hub_download
        args.official = hf_hub_download(
            "mit-han-lab/nunchaku-flux.1-schnell", "svdq-fp4_r32-flux.1-schnell.safetensors"
        )

    print("loading official checkpoint:", args.official)
    tensors, metadata = {}, None
    with safe_open(args.official, framework="pt") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    print("loading bf16 transformer weights:", args.model_id)
    sd = load_transformer_state_dict(args.model_id)

    print("loading covariances/basis:", args.cov)
    cov = torch.load(args.cov, map_location="cpu", weights_only=False)

    VALREF_LAYERS = {
        "transformer_blocks.0.qkv_proj",
        "transformer_blocks.7.mlp_fc1",
        "transformer_blocks.18.out_proj",
        "single_transformer_blocks.0.qkv_proj",
        "single_transformer_blocks.20.out_proj",
        "single_transformer_blocks.37.mlp_fc1",
    }
    valrefs = {}

    table = layer_table(args.num_double, args.num_single)
    qsnrs = {}
    t0 = time.time()
    for nk_prefix, w_keys, cov_key, kind, slice_end in tqdm(table, dynamic_ncols=True):
        Ws = [sd[k].float() for k in w_keys]
        W = torch.cat(Ws, dim=0)
        if slice_end is not None:
            W = W[:, :slice_end]
        U = cov[cov_key][:, -args.rank:]           # top-r eigenvectors
        H = cov[f"{cov_key}.H"]
        repl, qsnr = build_layer(
            W, H, U, kind, args.group_size, "cuda",
            gptq=not args.rtn, damp_pct=args.damp, block_size=args.block_size,
            valref=valrefs if nk_prefix in VALREF_LAYERS else None,
            valref_key=nk_prefix,
        )
        qsnrs[nk_prefix] = qsnr
        for name, t in repl.items():
            full = f"{nk_prefix}.{name}"
            assert full in tensors, f"unexpected key {full} (not in official checkpoint)"
            assert tensors[full].shape == t.shape, \
                f"{full}: shape {tuple(t.shape)} != official {tuple(tensors[full].shape)}"
            assert tensors[full].dtype == t.dtype, \
                f"{full}: dtype {t.dtype} != official {tensors[full].dtype}"
            tensors[full] = t

    print(f"quantized {len(table)} layers in {time.time() - t0:.0f}s")
    w_qsnr = sorted(qsnrs.items(), key=lambda kv: kv[1])
    print("lowest weight-QSNR layers:")
    for k, v in w_qsnr[:5]:
        print(f"  {v:6.2f} dB  {k}")
    print(f"median weight-QSNR: {w_qsnr[len(w_qsnr)//2][1]:.2f} dB")

    metadata["method"] = "dirotq-absorb-basis"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)
    print("saved:", args.out)
    with open(args.out + ".qsnr.json", "w") as f:
        json.dump(qsnrs, f, indent=2)
    torch.save(valrefs, args.out + ".valref.pt")
    print("saved validation refs:", args.out + ".valref.pt")


if __name__ == "__main__":
    main()
