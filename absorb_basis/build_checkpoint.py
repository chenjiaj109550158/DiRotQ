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


def pack_perm_vector(n: int, warp_n: int = 128) -> torch.Tensor:
    """Index permutation applied by NunchakuWeightPacker.pack_scale(group_size=-1)
    to a [n] vector: packed[i] = orig[perm[i]]."""
    s_pack_size = min(max(warp_n // 32, 2), 8)
    num_s_lanes = min(32, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    assert warp_s == warp_n and n % warp_s == 0
    idx = torch.arange(n).reshape(
        n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1
    )
    return idx.permute(0, 6, 1, 2, 4, 3, 5).contiguous().view(-1)


def unpack_scale_vector(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_scale(group_size=-1) for a flat [n] vector."""
    perm = pack_perm_vector(packed.shape[0])
    out = torch.empty_like(packed)
    out[perm] = packed
    return out


def _selftest_pack_perm():
    packer = NunchakuWeightPacker(bits=4)
    v = torch.randn(3072).to(torch.bfloat16)
    ref = packer.pack_scale(
        packer.pad_scale(v.view(-1, 1, 1, 1), group_size=-1), group_size=-1
    ).view(-1)
    mine = v[pack_perm_vector(3072)]
    assert torch.equal(ref, mine), "pack_perm_vector mismatch with NunchakuWeightPacker"
    assert torch.equal(unpack_scale_vector(ref), v), "unpack_scale_vector roundtrip failed"


def smoothquant_factors(act_amax: torch.Tensor, W: torch.Tensor, alpha: float) -> torch.Tensor:
    """Classic SmoothQuant: s_j = amax|X_j|^a / amax|W_:,j|^(1-a). Returns float32 [ic]."""
    ax = act_amax.float().clamp(min=1e-5)
    aw = W.abs().amax(dim=0).float().clamp(min=1e-5)
    s = ax.pow(alpha) / aw.pow(1.0 - alpha)
    return s.clamp(min=1e-5, max=1e4)


def two_level_scales(W_res: torch.Tensor, group_size: int, per_channel: bool,
                     hdiag: torch.Tensor | None = None,
                     clip_ratios: torch.Tensor | None = None):
    """Compute (top, micro_e4m3, effective) scales for NVFP4.

    top:   [oc] (per_channel) or scalar tensor  — bf16-storable float32
    micro: [oc, ng] float32 holding exact e4m3 values
    eff:   [oc, ng] float32 = micro * top      — the actual dequant grid

    With clip_ratios (and hdiag = diagonal of the input Hessian), each group's
    micro-scale is chosen from {r * amax/6 : r in clip_ratios} minimizing the
    Hessian-diagonal-weighted RTN error of that group (mirrors deepcompressor's
    calib_range search, at per-group granularity).
    """
    from utils.quant_utils import round_to_nf4_codebook

    oc, ic = W_res.shape
    ng = ic // group_size
    Wg = W_res.reshape(oc, ng, group_size)
    gmax = Wg.abs().amax(dim=-1)  # [oc, ng]
    if per_channel:
        top = (gmax.amax(dim=1) / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-8)  # [oc]
        top = top.to(torch.bfloat16).float()  # round to storable bf16 first
        top_b = top.unsqueeze(1)
    else:
        top = (gmax.amax() / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-8).reshape(())
        top = top.to(torch.bfloat16).float()
        top_b = top

    def to_e4m3(m):
        m = m.clamp(max=E4M3_MAX)
        return m.to(torch.float8_e4m3fn).float().clamp(min=2.0 ** -9)

    if clip_ratios is None:
        micro = to_e4m3(gmax / (E2M1_MAX * top_b))
    else:
        assert hdiag is not None
        hg = hdiag.to(W_res.device, torch.float32).clamp(min=0).reshape(1, ng, group_size)
        best_err = torch.full((oc, ng), float("inf"), device=W_res.device)
        micro = torch.empty((oc, ng), device=W_res.device)
        CHUNK = 2048
        for r in clip_ratios:
            micro_r = to_e4m3(float(r) * gmax / (E2M1_MAX * top_b))  # [oc, ng]
            eff_r = micro_r * top_b
            for c0 in range(0, oc, CHUNK):
                c1 = min(c0 + CHUNK, oc)
                q = round_to_nf4_codebook(Wg[c0:c1] / eff_r[c0:c1].unsqueeze(-1))
                q = q * eff_r[c0:c1].unsqueeze(-1)
                err = ((q - Wg[c0:c1]).pow(2) * hg).sum(dim=-1)  # [chunk, ng]
                take = err < best_err[c0:c1]
                best_err[c0:c1][take] = err[take]
                micro[c0:c1][take] = micro_r[c0:c1][take]

    eff = micro * top_b
    return top, micro, eff


def pack_top_scale_per_channel(packer: NunchakuWeightPacker, top: torch.Tensor):
    """Pack a per-channel top scale into the wcscales layout ([oc] bf16)."""
    t = top.to(torch.bfloat16).view(-1, 1, 1, 1)
    t = packer.pad_scale(t, group_size=-1)
    return packer.pack_scale(t, group_size=-1).view(-1)


@torch.no_grad()
def quantize_residual(W_res: torch.Tensor, H: torch.Tensor, kind: str,
                      group_size: int, device: str, gptq: bool,
                      damp_pct: float, block_size: int,
                      hdiag: torch.Tensor | None = None,
                      clip_ratios: torch.Tensor | None = None):
    """NVFP4-quantize a residual weight (in its final/smoothed domain) on the
    exact two-level kernel grid. Returns (W_q dequantized, top, micro)."""
    oc, ic = W_res.shape
    top, micro, eff = two_level_scales(W_res, group_size, per_channel=(kind == "qkv"),
                                       hdiag=hdiag, clip_ratios=clip_ratios)
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
    return W_q, top, micro


@torch.no_grad()
def pack_layer(W_q, top, micro, lora_down_prepack, lora_up, s, kind, device):
    """Pack quantized residual + lora + smooth into nunchaku replacement tensors."""
    ic = lora_down_prepack.shape[1]
    if kind == "qkv":
        W_n = (W_q / top.unsqueeze(1)).to(torch.bfloat16)
    else:
        W_n = (W_q / top).to(torch.bfloat16)
    packer = NunchakuWeightPacker(bits=4)
    qweight, wscales, _bias, smooth_packed, (ld, lu) = convert_to_nunchaku_w4x4y16(
        weight=W_n,
        scale=micro.to(torch.bfloat16),
        bias=None,
        smooth=s.to(torch.bfloat16),
        lora=(lora_down_prepack.to(torch.bfloat16), lora_up.to(torch.bfloat16)),
        float_point=True,
    )
    out = {
        "qweight": qweight.cpu(),
        "wscales": wscales.cpu(),
        "lora_down": ld.cpu(),
        "lora_up": lu.cpu(),
        "smooth": smooth_packed.view(-1).to(torch.bfloat16).cpu(),
        "smooth_orig": smooth_packed.view(-1).to(torch.bfloat16).cpu(),
    }
    if kind == "qkv":
        out["wcscales"] = pack_top_scale_per_channel(packer, top).cpu()
    else:
        out["wtscale"] = top.reshape(1).to(torch.bfloat16).cpu()
    return out


@torch.no_grad()
def build_layer_v2(W: torch.Tensor, H: torch.Tensor, D: torch.Tensor,
                   lora_up: torch.Tensor, kind: str, group_size: int, device: str,
                   gptq: bool, damp_pct: float, block_size: int,
                   alt_iters: int = 1, refit: bool = False,
                   clip_ratios: torch.Tensor | None = None,
                   mu: torch.Tensor | None = None,
                   valref: dict | None = None, valref_key: str = ""):
    """Raw-domain (no-smooth) layer build with optional clip search, alternating
    lora refit, and bias correction.

    D:        [r, ic] lora_down (pre-pack); low-rank branch = x @ D^T @ lora_up^T
    lora_up:  [oc, r] initial value; refit updates it in the H metric.
    mu:       [ic] calibration input mean for bias correction (None to skip).

    Returns (out_tensors, qsnr, bias_delta_or_None).
    """
    W = W.to(device=device, dtype=torch.float32)
    D = D.to(device=device, dtype=torch.float32)
    lora_up = lora_up.to(device=device, dtype=torch.float32)
    oc, ic = W.shape
    Hg = H.to(device=device, dtype=torch.float32)
    hdiag = Hg.diagonal() if clip_ratios is not None else None

    W_q = top = micro = None
    for _ in range(max(1, alt_iters)):
        W_res = W - lora_up @ D
        W_q, top, micro = quantize_residual(
            W_res, Hg, kind, group_size, device, gptq, damp_pct, block_size,
            hdiag=hdiag, clip_ratios=clip_ratios,
        )
        if refit:
            E = W - W_q
            G = D @ Hg @ D.t()
            G = G + 1e-6 * G.diagonal().mean() * torch.eye(G.shape[0], device=device)
            lora_up = (E @ Hg @ D.t()) @ torch.linalg.inv(G)

    s = torch.ones(ic, device=device, dtype=torch.float32)
    out = pack_layer(W_q, top, micro, D, lora_up, s, kind, device)

    W_hat = W_q + lora_up @ D
    bias_delta = None
    if mu is not None:
        bias_delta = (mu.to(device=device, dtype=torch.float32) @ (W - W_hat).t()).cpu()

    if valref is not None:
        valref[valref_key] = {
            "W_q": W_q.half().cpu(),
            "U_eff": D.t().half().cpu(),
            "lora_up": lora_up.half().cpu(),
            "s": s.cpu(),
        }

    err = (W_hat - W).pow(2).sum()
    qsnr = 10.0 * torch.log10(W.pow(2).sum() / err.clamp(min=1e-20))
    return out, qsnr.item(), bias_delta


@torch.no_grad()
def hsvd_basis(W: torch.Tensor, H: torch.Tensor, rank: int, device: str,
               damping: float = 0.01):
    """Activation-weighted (H-metric) rank-r branch: minimize ||X (W - L)^T||_F.

    H = C C^T (eigen square root, damped). Best rank-r L = SVD_r(W C) C^{-1}.
    Returns (D [r, ic], lora_up [oc, r]) with L = lora_up @ D.
    """
    Hd = H.to(device=device, dtype=torch.float64)
    Hd = Hd + damping * Hd.diagonal().mean() * torch.eye(
        Hd.shape[0], dtype=Hd.dtype, device=device
    )
    evals, Q = torch.linalg.eigh(Hd)
    evals = evals.clamp(min=evals.max() * 1e-12)
    C = Q * evals.sqrt().unsqueeze(0)              # [ic, ic]
    Cinv = (Q / evals.sqrt().unsqueeze(0)).t()     # [ic, ic] = Lambda^-1/2 Q^T
    M = (W.to(device=device, dtype=torch.float64) @ C).float()  # fp32 SVD (top-r only)
    Us, Ss, Vh = torch.linalg.svd(M, full_matrices=False)
    lora_up = Us[:, :rank] * Ss[:rank].unsqueeze(0)             # [oc, r]
    D = (Vh[:rank].double() @ Cinv).float()                     # [r, ic]
    return D, lora_up


@torch.no_grad()
def build_layer(W: torch.Tensor, H: torch.Tensor, U: torch.Tensor, kind: str,
                group_size: int, device: str, gptq: bool,
                damp_pct: float, block_size: int,
                valref: dict | None = None, valref_key: str = "",
                s: torch.Tensor | None = None, decouple: bool = False):
    """Returns dict of replacement tensors for one nunchaku layer.

    Smoothed-domain mode (decouple=False): W and H must ALREADY be in the
    smoothed domain (W = W_raw * s, H = D^-1 H_raw D^-1) and U the top-r
    eigenvectors of the smoothed covariance; stored lora_down = U / s.

    Decoupled mode (decouple=True): W is the RAW weight, U the RAW-domain
    basis, and smoothing applies to the main (4-bit) branch only:
        Y = (X U)(W U)^T + Q4(X/s) Q4(W_res * s)^T,  W_res = W - (W U) U^T.
    H must be the smoothed Hessian D^-1 H_raw D^-1 (input of the main branch
    is X/s). Stored lora_down = U (kernel applies low-rank on the raw input).
    """
    W = W.to(device=device, dtype=torch.float32)
    U = U.to(device=device, dtype=torch.float32)  # [ic, r]
    oc, ic = W.shape
    if s is None:
        s = torch.ones(ic, device=device, dtype=torch.float32)
    else:
        s = s.to(device=device, dtype=torch.float32)

    lora_up = W @ U                      # [oc, r]
    W_res = W - lora_up @ U.t()          # [oc, ic]
    if decouple:
        W_res = W_res * s.unsqueeze(0)   # main branch moves to the smoothed domain

    W_q, top, micro = quantize_residual(
        W_res, H, kind, group_size, device, gptq, damp_pct, block_size
    )

    # normalize by top so the packer sees codes * micro (micro is what wscales stores)
    if kind == "qkv":
        W_n = (W_q / top.unsqueeze(1)).to(torch.bfloat16)
    else:
        W_n = (W_q / top).to(torch.bfloat16)

    packer = NunchakuWeightPacker(bits=4)
    if decouple:
        lora_down_prepack = U.t().to(torch.bfloat16)                 # [r, ic], raw basis
    else:
        lora_down_prepack = (U.t() / s.unsqueeze(0)).to(torch.bfloat16)  # [r, ic], absorbs 1/s
    qweight, wscales, _bias, smooth_packed, (ld, lu) = convert_to_nunchaku_w4x4y16(
        weight=W_n,
        scale=micro.to(torch.bfloat16),          # [oc, ng] -> micro-scale e4m3 path
        bias=None,
        smooth=s.to(torch.bfloat16),
        lora=(lora_down_prepack,                 # lora_down pre-pack [r, ic]
              lora_up.to(torch.bfloat16)),       # lora_up  pre-pack [oc, r]
        float_point=True,
    )

    out = {
        "qweight": qweight.cpu(),
        "wscales": wscales.cpu(),
        "lora_down": ld.cpu(),
        "lora_up": lu.cpu(),
        "smooth": smooth_packed.view(-1).to(torch.bfloat16).cpu(),
        "smooth_orig": smooth_packed.view(-1).to(torch.bfloat16).cpu(),
    }
    if kind == "qkv":
        out["wcscales"] = pack_top_scale_per_channel(packer, top).cpu()
    else:
        out["wtscale"] = top.reshape(1).to(torch.bfloat16).cpu()

    if valref is not None:
        # U_eff: raw-domain effective lora_down (kernel applies lora on raw x)
        U_eff = U if decouple else U / s.unsqueeze(1)
        valref[valref_key] = {
            "W_q": W_q.half().cpu(),          # dequantized residual on the kernel grid (smoothed domain)
            "U_eff": U_eff.half().cpu(),      # [ic, r] raw-domain lora_down (unpacked)
            "lora_up": lora_up.half().cpu(),  # [oc, r] (unpacked)
            "s": s.float().cpu(),             # smooth factor (bf16-exact values)
        }

    # quantization SNR of the full layer (lora + quantized residual) vs W,
    # compared in the domain of the passed-in W (raw for decouple).
    if decouple:
        W_hat = W_q / s.unsqueeze(0) + lora_up @ U.t()
    else:
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
    ap.add_argument("--smooth", choices=["none", "a05", "svdq", "main-a05", "main-search"],
                    default="none",
                    help="none | smooth-then-PCA: a05 (classic alpha=0.5) / svdq (official "
                         "factors) | decoupled (PCA raw, smooth main branch only): "
                         "main-a05 (fixed alpha) / main-search (per-layer alpha search)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="SmoothQuant alpha for --smooth a05 / main-a05")
    ap.add_argument("--act-amax", default="models/flux-schnell/basis/absorb_act_amax.pt")
    ap.add_argument("--act-samples", default="models/flux-schnell/basis/absorb_act_samples.pt",
                    help="raw activation samples for --smooth main-search")
    ap.add_argument("--search-alphas", type=float, nargs="*",
                    default=[0.25, 0.4, 0.5, 0.6, 0.75],
                    help="alpha grid for --smooth main-search (no-smooth is always a candidate)")
    ap.add_argument("--basis", choices=["pca", "hsvd"], default="pca",
                    help="low-rank branch: pca (top-r input PCA) or hsvd "
                         "(activation-weighted SVD of W in the H metric). "
                         "hsvd requires --smooth none")
    ap.add_argument("--clip-search", action="store_true",
                    help="per-group weight-scale clip search (Hessian-diagonal-weighted), "
                         "requires --smooth none")
    ap.add_argument("--clip-grid", type=float, nargs=3, default=[0.8, 1.0, 21],
                    metavar=("LO", "HI", "N"), help="clip ratio grid")
    ap.add_argument("--refit-lora", action="store_true",
                    help="closed-form H-metric refit of lora_up after each GPTQ pass "
                         "(requires --smooth none)")
    ap.add_argument("--alt-iters", type=int, default=1,
                    help="number of (quantize -> refit) passes (>=2 alternates)")
    ap.add_argument("--bias-correct", action="store_true",
                    help="calibration-mean bias correction (requires --smooth none "
                         "and --act-samples)")
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

    v2_features = args.basis == "hsvd" or args.clip_search or args.refit_lora \
        or args.bias_correct or args.alt_iters > 1
    if v2_features:
        assert args.smooth == "none", \
            "--basis hsvd / --clip-search / --refit-lora / --bias-correct / --alt-iters " \
            "are only supported with --smooth none"
    clip_ratios = None
    if args.clip_search:
        lo, hi, n = args.clip_grid
        clip_ratios = torch.linspace(lo, hi, int(n))

    act_amax = None
    if args.smooth in ("a05", "main-a05", "main-search"):
        act_amax = torch.load(args.act_amax, map_location="cpu", weights_only=False)
    if args.smooth == "svdq":
        _selftest_pack_perm()
    act_samples = None
    if args.smooth == "main-search" or args.bias_correct:
        act_samples = torch.load(args.act_samples, map_location="cpu", weights_only=False)
    if args.smooth == "main-search":
        from absorb_basis.validate_kernel import simulate_act_fp4

    from absorb_basis.collect_cov import eigh_topr

    table = layer_table(args.num_double, args.num_single)
    qsnrs = {}
    alphas = {}
    t0 = time.time()
    for nk_prefix, w_keys, cov_key, kind, slice_end in tqdm(table, dynamic_ncols=True):
        Ws = [sd[k].float() for k in w_keys]
        W = torch.cat(Ws, dim=0)
        if slice_end is not None:
            W = W[:, :slice_end]
        H = cov[f"{cov_key}.H"]
        decouple = args.smooth in ("main-a05", "main-search")

        if args.smooth == "none":
            Wg = W.to("cuda", torch.float32)
            if args.basis == "hsvd":
                D, lu0 = hsvd_basis(Wg, H, args.rank, "cuda")
            else:
                Ug = cov[cov_key][:, -args.rank:].to("cuda", torch.float32)
                D, lu0 = Ug.t(), Wg @ Ug
            mu = None
            if args.bias_correct:
                mu = act_samples[cov_key].float().mean(dim=0)
            repl, qsnr, bias_delta = build_layer_v2(
                Wg, H, D, lu0, kind, args.group_size, "cuda",
                gptq=not args.rtn, damp_pct=args.damp, block_size=args.block_size,
                alt_iters=args.alt_iters, refit=args.refit_lora,
                clip_ratios=clip_ratios, mu=mu,
                valref=valrefs if nk_prefix in VALREF_LAYERS else None,
                valref_key=nk_prefix,
            )
            if bias_delta is not None:
                bkey = f"{nk_prefix}.bias"
                perm = pack_perm_vector(bias_delta.shape[0])
                tensors[bkey] = (
                    tensors[bkey].float() + bias_delta[perm].float()
                ).to(tensors[bkey].dtype)
            qsnrs[nk_prefix] = qsnr
            for name, t in repl.items():
                full = f"{nk_prefix}.{name}"
                assert full in tensors, f"unexpected key {full}"
                assert tensors[full].shape == t.shape and tensors[full].dtype == t.dtype, full
                tensors[full] = t
            del Wg
            continue

        if args.smooth in ("a05", "svdq"):         # smooth-then-PCA (smoothed domain)
            if args.smooth == "a05":
                s = smoothquant_factors(act_amax[cov_key], W, args.alpha)
            else:  # svdq: official per-layer smooth factors (stored packed)
                s = unpack_scale_vector(tensors[f"{nk_prefix}.smooth"]).float()
            s = s.to(torch.bfloat16).float()       # bf16-exact, matches runtime storage
            W = W * s.unsqueeze(0)                 # smoothed weight
            H = H / (s.unsqueeze(1) * s.unsqueeze(0))  # cov of X/s
            U, _ = eigh_topr(H, args.rank)         # PCA in the smoothed domain
        else:                                      # decoupled: PCA raw, smooth main branch
            U = cov[cov_key][:, -args.rank:]
            Wg = W.to("cuda", torch.float32)
            Ug = U.to("cuda", torch.float32)
            W_res_raw = Wg - (Wg @ Ug) @ Ug.t()
            ax = act_amax[cov_key].to("cuda")
            if args.smooth == "main-a05":
                s = smoothquant_factors(ax, W_res_raw, args.alpha).to(torch.bfloat16).float()
                alphas[nk_prefix] = args.alpha
            else:  # main-search: per-layer alpha grid, no-smooth always a candidate
                X = act_samples[cov_key].to("cuda", torch.float32)
                ref = X @ W_res_raw.t()
                Hc = H.to("cuda", torch.float32)
                best = (None, None, float("inf"))
                for cand in [None] + list(args.search_alphas):
                    if cand is None:
                        s_c = torch.ones(W.shape[1], device="cuda")
                    else:
                        s_c = smoothquant_factors(ax, W_res_raw, cand).to(torch.bfloat16).float()
                    H_c = Hc / (s_c.unsqueeze(1) * s_c.unsqueeze(0))
                    W_q_c, _, _ = quantize_residual(
                        W_res_raw * s_c.unsqueeze(0), H_c, kind, args.group_size,
                        "cuda", not args.rtn, args.damp, args.block_size,
                    )
                    est = simulate_act_fp4((X / s_c).to(torch.bfloat16)).float() @ W_q_c.t()
                    err = (est - ref).pow(2).sum().item()
                    if err < best[2]:
                        best = (cand, s_c, err)
                    del W_q_c, est, H_c
                alphas[nk_prefix] = best[0]
                s = best[1]
                del X, ref, Hc
            H = H.to("cuda", torch.float32) / (s.unsqueeze(1) * s.unsqueeze(0))
            del Wg, Ug, W_res_raw
            torch.cuda.empty_cache()

        repl, qsnr = build_layer(
            W, H, U, kind, args.group_size, "cuda",
            gptq=not args.rtn, damp_pct=args.damp, block_size=args.block_size,
            valref=valrefs if nk_prefix in VALREF_LAYERS else None,
            valref_key=nk_prefix,
            s=s, decouple=decouple,
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

    metadata["method"] = "dirotq-absorb-basis" + (
        "" if args.smooth == "none" else f"-smooth-{args.smooth}"
    ) + (f"-{args.basis}" if args.basis != "pca" else "") \
      + ("-clip" if args.clip_search else "") \
      + ("-refit" if args.refit_lora else "") \
      + (f"-alt{args.alt_iters}" if args.alt_iters > 1 else "") \
      + ("-bc" if args.bias_correct else "")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)
    print("saved:", args.out)
    with open(args.out + ".qsnr.json", "w") as f:
        json.dump({"qsnr": qsnrs, "alphas": alphas} if alphas else qsnrs, f, indent=2)
    if alphas:
        from collections import Counter
        print("alpha histogram:", dict(Counter(str(a) for a in alphas.values())))
    torch.save(valrefs, args.out + ".valref.pt")
    print("saved validation refs:", args.out + ".valref.pt")


if __name__ == "__main__":
    main()
