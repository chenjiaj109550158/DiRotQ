"""
speedup/convert_to_int4_ckpt.py

Convert the bf16 fake-quantized DiRotQ checkpoint into a true int4 checkpoint.

DiRotQ's RTN/GPTQ pipeline stores weights as bf16/fp16 with the low-region
values living on the symmetric int4 grid (codes ∈ [-8, 7], per-group scales
= max(|w|) / 7). The high tail region is true bf16. This script:

  1. Loads the existing fake-quant checkpoint.
  2. For each ActQuantWrapper-wrapped layer:
       - Splits W into [W_low (n_low cols, on int4 grid) | W_tail (hlen cols, fp)].
       - Re-extracts the per-group scale and signed int4 codes from W_low.
       - Packs codes into uint8 (two per byte, low nibble first) and stores
         scales as bf16 [N, n_low / gs].
       - Keeps W_tail and bias as bf16.
  3. Drops redundant fp16 weight aliases that ActQuantWrapper saves
     (`<layer>.weight`, `<layer>.bias` are duplicates of `<layer>.module.*`).
  4. Saves a manifest dict containing the packed weights, fp16 tails,
     biases, all non-quantized weights (norms, embeddings, ...) and
     per-layer metadata.
  5. Verifies that dequantizing the int4 manifest reproduces the original
     fake-quant fp16 weights exactly (modulo bf16 noise).
  6. Reports size savings on disk and per-section breakdown.

Layer-shape rules used (flux-dev / flux-schnell / pixart):
  - in_features == hidden     -> hlen = high_len_hidden
  - in_features == intermediate -> hlen = high_len_down
  - everything else           -> hlen = 0 (no split, full int4 quantization).

The resulting manifest format:

  {
      "format": "dirotq-int4-v1",
      "model": "flux-dev",
      "groupsize": 64,
      "high_len_hidden": 384,
      "high_len_down": 1536,
      "high_len_head": 16,
      "weights": {
          # Quantized layer: split into packed/scales/tail/bias
          "<layer>.module._w_packed":  uint8 [N, n_low/2],
          "<layer>.module._w_scales":  bf16  [N, n_low/gs],
          "<layer>.module._w_tail":    bf16  [N, hlen],     # optional (hlen>0)
          "<layer>.module.bias":       bf16  [N],            # optional
          # Non-quantized layer: kept as-is
          "<other.layer>.weight":      bf16  [...],
          "<other.layer>.bias":        bf16  [...],
      },
      "layer_meta": {
          "<layer>": {"kind", "in_features", "out_features",
                      "n_low", "hlen", "groupsize", "dtype"},
      },
  }

Usage:
    python -m speedup.convert_to_int4_ckpt --model flux-dev --gptq
    python -m speedup.convert_to_int4_ckpt --model flux-dev --gptq --no-verify
    python -m speedup.convert_to_int4_ckpt --model flux-dev --gptq -o my_int4.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

_HERE = Path(__file__).resolve().parent
_DIROTQ_ROOT = _HERE.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))


# ---------------------------------------------------------------------------
# Pack / unpack helpers (matching DiRotQ's symmetric int4 RTN convention)
# ---------------------------------------------------------------------------

def pack_int4_sym(W_low_fp: torch.Tensor, group_size: int):
    """Re-extract per-group symmetric int4 from fake-quantized fp/bf16 weights.

    DiRotQ's quantization produces W_q = code * scale where:
      - RTN  : codes in [-7, 7]  (scale = max(|W_orig|)/7, no -8 values).
      - GPTQ : codes in [-8, 7]  (loss-aware adjustment can push to -8).

    For RTN the recovery scale is max(|W_q|)/7 — exact.
    For GPTQ the most-extreme element may be at code=-8, so
    max(|W_q|) = 8*scale_orig. To handle both cases we try BOTH candidates
    (max/7 and max/8) per group and pick the one with smaller reconstruction
    residual. This recovers the original scale bit-for-bit (modulo bf16 noise).

    Returns:
        packed:  uint8 [N, K/2]   — two int4 codes per byte, low nibble first.
        scales:  W_low_fp.dtype [N, K/gs]
    """
    assert W_low_fp.dim() == 2
    N, K = W_low_fp.shape
    assert K % group_size == 0, f"K={K} not divisible by gs={group_size}"
    assert K % 2 == 0, "K must be even for two-int4-per-byte packing"

    Wg = W_low_fp.float().reshape(N, K // group_size, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    s7 = max_abs / 7.0
    s8 = max_abs / 8.0

    codes_7 = (Wg / s7).round().clamp(-8, 7)
    codes_8 = (Wg / s8).round().clamp(-8, 7)
    err_7 = (Wg - codes_7 * s7).abs().sum(dim=-1, keepdim=True)
    err_8 = (Wg - codes_8 * s8).abs().sum(dim=-1, keepdim=True)

    use_8 = err_8 < err_7
    scale = torch.where(use_8, s8, s7)
    codes = torch.where(use_8, codes_8, codes_7).to(torch.int8).reshape(N, K)

    even = codes[:, 0::2].to(torch.int32) & 0xF
    odd = codes[:, 1::2].to(torch.int32) & 0xF
    packed = (even | (odd << 4)).to(torch.uint8).contiguous()

    return packed, scale.squeeze(-1).to(W_low_fp.dtype).contiguous()


def unpack_int4_sym(packed: torch.Tensor, scales: torch.Tensor,
                    group_size: int, K: int) -> torch.Tensor:
    """Inverse of pack_int4_sym. Returns a [N, K] tensor of dtype scales.dtype."""
    N, half = packed.shape
    assert 2 * half == K
    p32 = packed.to(torch.int32)
    even = p32 & 0xF
    odd = (p32 >> 4) & 0xF
    even = torch.where(even >= 8, even - 16, even).to(torch.int8)
    odd = torch.where(odd >= 8, odd - 16, odd).to(torch.int8)

    codes = torch.empty(N, K, dtype=torch.int8, device=packed.device)
    codes[:, 0::2] = even
    codes[:, 1::2] = odd

    scales_e = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    out = (codes.to(torch.float32) * scales_e).to(scales.dtype)
    return out


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

def _hlen_for_layer(in_features: int, cfg: dict, *,
                    high_len_hidden: int, high_len_down: int,
                    act_quantized: bool = True) -> int:
    """Map input dim → high_bits_length for GLOBAL-rotation layers.

    Per-head layers are handled separately by `_per_head_info` and use
    `high_len_head` instead — don't call this for them.
    """
    if not act_quantized:
        return 0
    if in_features == cfg["dims"]["hidden"]:
        return high_len_hidden
    if in_features == cfg["dims"]["intermediate"]:
        return high_len_down
    return 0


def _per_head_info(model: str, layer_prefix: str, cfg: dict, *,
                   high_len_head: int) -> dict | None:
    """Return per-head info {num_heads, head_dim, d_q, hlen_per_head} if the
    layer uses per-head PCA rotation, else None.

    Per-head layers in DiRotQ:
      - flux-dev / flux-schnell: NONE (all attn-out uses global rotation in
            the schnell model_utils — the per-head field is unused).
      - pixart-sigma: `attn1.to_out.0`, `attn2.to_out.0` (self-attn + cross-attn
            output projections).
    """
    if model in ("flux-dev", "flux-schnell"):
        return None
    if model == "pixart-sigma":
        if layer_prefix.endswith(".attn1.to_out.0") or layer_prefix.endswith(".attn2.to_out.0"):
            H = int(cfg["dims"]["num_heads"])
            d = int(cfg["dims"]["head"])
            d_q = d - high_len_head
            assert d_q > 0, f"head_dim={d} <= high_len_head={high_len_head}"
            return {"num_heads": H, "head_dim": d, "d_q": d_q,
                    "hlen_per_head": high_len_head}
    return None


def pack_per_head_int4(W: torch.Tensor, num_heads: int, head_dim: int,
                       hlen_per_head: int) -> tuple:
    """Pack a per-head weight [N, H*d] into (low_packed, low_scales, tail).

    The fake-quant cache stores W in interleaved per-head layout — within each
    head, the first d_q channels are on the int4 grid (low) and the last
    `hlen_per_head` channels are bf16 tail.

    Returns:
      packed:  uint8 [N, H*d_q / 2]    int4-packed low region
      scales:  W.dtype [N, H]           one scale per row per head (gs = d_q)
      tail:    W.dtype [N, H*hlen_per_head]  bf16 passthrough
    """
    H, d = num_heads, head_dim
    d_q = d - hlen_per_head
    assert W.shape[1] == H * d, f"W shape {W.shape} != [N, H*d]={W.shape[0], H*d}"
    if (H * d_q) % 2 != 0:
        raise ValueError(
            f"H*d_q = {H * d_q} must be even for two-int4-per-byte packing")

    W_3d = W.reshape(W.shape[0], H, d)
    W_low_3d = W_3d[:, :, :d_q].contiguous()         # [N, H, d_q]
    W_tail_3d = W_3d[:, :, d_q:].contiguous()        # [N, H, hlen]
    W_low = W_low_3d.reshape(W.shape[0], H * d_q)    # [N, H*d_q]
    W_tail = W_tail_3d.reshape(W.shape[0], H * hlen_per_head)

    # GPTQ for per-head uses groupsize = d_q (each head = one scale group).
    packed, scales = pack_int4_sym(W_low, group_size=d_q)
    return packed, scales, W_tail


def unpack_per_head_int4(packed: torch.Tensor, scales: torch.Tensor,
                         tail: torch.Tensor, num_heads: int, head_dim: int,
                         hlen_per_head: int) -> torch.Tensor:
    """Inverse of pack_per_head_int4."""
    H, d = num_heads, head_dim
    d_q = d - hlen_per_head
    N = packed.shape[0]

    W_low_2d = unpack_int4_sym(packed, scales, d_q, H * d_q)  # [N, H*d_q]
    W_low_3d = W_low_2d.reshape(N, H, d_q)
    W_tail_3d = tail.reshape(N, H, hlen_per_head)
    W_3d = torch.cat([W_low_3d, W_tail_3d], dim=2)            # [N, H, d]
    return W_3d.reshape(N, H * d)


def _is_act_quantized(state: dict, layer_prefix: str) -> bool:
    """A layer is act-quantized iff its quantizer.maxq is small (e.g., 7 or 15
    for sym/asym int4). bits=16 (act skipped via --skip-quant-layers) yields
    maxq=32767. Returns True if act-quantized at <= 8 bits."""
    maxq_key = f"{layer_prefix}.quantizer.maxq"
    if maxq_key not in state:
        return True  # default: assume act-quantized
    maxq = int(state[maxq_key].item())
    return maxq <= 255


def convert_checkpoint(*, model: str, src_ckpt: Path, dst_ckpt: Path,
                       gs_override: int | None = None,
                       progress: bool = True) -> dict:
    """Convert one fake-quant cache → int4 manifest. Returns size summary."""
    cfg_path = _DIROTQ_ROOT / "models" / model / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    rotation_path = _DIROTQ_ROOT / cfg["rotation"]["output_path"]
    print(f"Loading rotation file (for high_len_*): {rotation_path}")
    rot = torch.load(rotation_path, map_location="cpu", weights_only=False)
    high_len_hidden = int(rot["high_len_hidden"])
    high_len_down = int(rot.get("high_len_down", 0))
    high_len_head = int(rot["high_len_head"])

    gs = gs_override or cfg["quantization"]["w_groupsize"]
    # Align to gs (matches configure_quantizers_by_name).
    high_len_hidden = ((high_len_hidden + gs - 1) // gs) * gs
    high_len_down = ((high_len_down + gs - 1) // gs) * gs
    print(f"  high_len_hidden = {high_len_hidden}")
    print(f"  high_len_down   = {high_len_down}")
    print(f"  high_len_head   = {high_len_head}  (used for per-head layers)")
    print(f"  groupsize       = {gs}")
    del rot  # release the 1.3GB rotation file before loading the cache

    print(f"\nLoading fake-quant cache: {src_ckpt}")
    state = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    print(f"  cache keys: {len(state)}")

    qw_keys = [k for k in state if k.endswith(".module.weight")]
    print(f"  quantized linear weights (.module.weight): {len(qw_keys)}")

    weights_out: dict[str, torch.Tensor] = {}
    layer_meta: dict[str, dict] = {}

    bytes_packed = 0
    bytes_scales = 0
    bytes_tail = 0
    bytes_bias = 0
    bytes_other = 0

    n_split = 0
    n_full = 0
    n_perhead = 0

    # ---- Process quantized layers ----
    handled_keys: set[str] = set()
    for i, qw_key in enumerate(qw_keys):
        prefix = qw_key[: -len(".module.weight")]  # e.g. transformer_blocks.0.attn.to_q
        W = state[qw_key]                           # bf16 [N, K]
        bias_key = f"{prefix}.module.bias"
        bias = state.get(bias_key)
        N, K = W.shape

        act_q = _is_act_quantized(state, prefix)
        ph_info = _per_head_info(model, prefix, cfg,
                                  high_len_head=high_len_head) if act_q else None

        if ph_info:
            # ---- Per-head layer (pixart-sigma attn out) ----
            H = ph_info["num_heads"]
            d = ph_info["head_dim"]
            d_q = ph_info["d_q"]
            hlen_per_head = ph_info["hlen_per_head"]
            assert K == H * d, f"{prefix}: K={K} != H*d={H*d}"

            packed, scales, W_tail = pack_per_head_int4(
                W, num_heads=H, head_dim=d, hlen_per_head=hlen_per_head)
            n_low = H * d_q
            hlen = H * hlen_per_head
            kind = "per-head-int4"
            n_perhead += 1

            weights_out[f"{prefix}.module._w_packed"] = packed
            weights_out[f"{prefix}.module._w_scales"] = scales
            weights_out[f"{prefix}.module._w_tail"] = W_tail
            bytes_packed += packed.numel() * packed.element_size()
            bytes_scales += scales.numel() * scales.element_size()
            bytes_tail += W_tail.numel() * W_tail.element_size()

            layer_meta[prefix] = {
                "kind": kind,
                "in_features": int(K),
                "out_features": int(N),
                "n_low": int(n_low),
                "hlen": int(hlen),
                "num_heads": int(H),
                "head_dim": int(d),
                "d_q": int(d_q),
                "hlen_per_head": int(hlen_per_head),
                "groupsize": int(d_q),  # one scale per head
                "dtype": str(W.dtype),
            }
        else:
            # ---- Global / full-int4 layer ----
            hlen = _hlen_for_layer(K, cfg, high_len_hidden=high_len_hidden,
                                    high_len_down=high_len_down,
                                    act_quantized=act_q)
            n_low = K - hlen
            if n_low <= 0:
                raise RuntimeError(f"{prefix}: n_low={n_low} non-positive")
            if n_low % gs != 0 or n_low % 2 != 0:
                print(f"WARNING: {prefix}: n_low={n_low} not divisible by gs/2; skipping")
                continue

            if hlen > 0:
                W_low = W[:, :n_low].contiguous()
                W_tail = W[:, n_low:].contiguous()
                kind = "split-int4"
                n_split += 1
            else:
                W_low = W.contiguous()
                W_tail = None
                kind = "full-int4"
                n_full += 1

            packed, scales = pack_int4_sym(W_low, gs)

            weights_out[f"{prefix}.module._w_packed"] = packed
            weights_out[f"{prefix}.module._w_scales"] = scales
            if W_tail is not None:
                weights_out[f"{prefix}.module._w_tail"] = W_tail
                bytes_tail += W_tail.numel() * W_tail.element_size()

            bytes_packed += packed.numel() * packed.element_size()
            bytes_scales += scales.numel() * scales.element_size()

            layer_meta[prefix] = {
                "kind": kind,
                "in_features": int(K),
                "out_features": int(N),
                "n_low": int(n_low),
                "hlen": int(hlen),
                "groupsize": int(gs),
                "dtype": str(W.dtype),
            }

        if bias is not None:
            weights_out[f"{prefix}.module.bias"] = bias
            bytes_bias += bias.numel() * bias.element_size()

        handled_keys.update([
            qw_key, bias_key,
            f"{prefix}.weight", f"{prefix}.bias",
            f"{prefix}.quantizer.maxq",
            f"{prefix}.quantizer.scale", f"{prefix}.quantizer.zero",
        ])

        if progress and (i + 1) % 50 == 0:
            print(f"  packed {i+1}/{len(qw_keys)} layers")

    # ---- Carry over non-quantized weights ----
    for k, v in state.items():
        if k in handled_keys:
            continue
        weights_out[k] = v
        if hasattr(v, "numel"):
            bytes_other += v.numel() * v.element_size()

    print(f"\nPacked layers: split-int4={n_split}, full-int4={n_full}, "
          f"per-head-int4={n_perhead}")
    # Source bf16 equivalent for the same layers (low region is 4x bigger as bf16
    # than as packed int4; scales don't exist in source; everything else identical).
    src_low_bf16 = bytes_packed * 4
    src_total_bf16 = src_low_bf16 + bytes_tail + bytes_bias + bytes_other
    new_total = bytes_packed + bytes_scales + bytes_tail + bytes_bias + bytes_other
    print(f"\nByte-level breakdown:")
    print(f"  {'Section':<22}{'Source (bf16)':>18}{'New (int4)':>18}")
    print(f"  {'-'*22}{'-'*18}{'-'*18}")
    print(f"  {'low region':<22}{src_low_bf16/1e9:>15.3f} GB{bytes_packed/1e9:>15.3f} GB")
    print(f"  {'group scales':<22}{0:>15.3f} GB{bytes_scales/1e9:>15.3f} GB")
    print(f"  {'fp tails':<22}{bytes_tail/1e9:>15.3f} GB{bytes_tail/1e9:>15.3f} GB")
    print(f"  {'biases':<22}{bytes_bias/1e9:>15.3f} GB{bytes_bias/1e9:>15.3f} GB")
    print(f"  {'non-quantized':<22}{bytes_other/1e9:>15.3f} GB{bytes_other/1e9:>15.3f} GB")
    print(f"  {'TOTAL':<22}{src_total_bf16/1e9:>15.3f} GB{new_total/1e9:>15.3f} GB")
    print(f"  ratio (TOTAL)       : {src_total_bf16/max(new_total,1):.2f}x smaller")

    manifest = {
        "format": "dirotq-int4-v1",
        "model": model,
        "groupsize": gs,
        "high_len_hidden": high_len_hidden,
        "high_len_down": high_len_down,
        "high_len_head": high_len_head,
        "dtype": str(state[qw_keys[0]].dtype),
        "weights": weights_out,
        "layer_meta": layer_meta,
    }

    print(f"\nSaving int4 manifest to: {dst_ckpt}")
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, dst_ckpt)

    # Disk size
    src_disk = src_ckpt.stat().st_size
    dst_disk = dst_ckpt.stat().st_size
    print(f"\n=== File-size comparison (on disk) ===")
    print(f"  source (fake-quant bf16) : {src_disk/1e9:7.3f} GB  ({src_ckpt.name})")
    print(f"  destination (int4)        : {dst_disk/1e9:7.3f} GB  ({dst_ckpt.name})")
    saving = (src_disk - dst_disk) / src_disk * 100
    print(f"  saving                    : {saving:.1f}%   ({(src_disk-dst_disk)/1e9:.3f} GB)")
    print(f"  ratio                     : {src_disk/dst_disk:.2f}x smaller")

    return {
        "src_disk_bytes": src_disk,
        "dst_disk_bytes": dst_disk,
        "n_split": n_split,
        "n_full": n_full,
        "bytes_packed": bytes_packed,
        "bytes_scales": bytes_scales,
        "bytes_tail": bytes_tail,
        "bytes_bias": bytes_bias,
        "bytes_other": bytes_other,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_equivalence(src_ckpt: Path, dst_ckpt: Path, *, sample: int = 0,
                        matmul_check: bool = True) -> dict:
    """Reload both checkpoints, dequantize int4 → bf16, compare to original.

    Numerical equivalence is *up to bf16 round-off*. The original fake-quant
    cache stores W as bf16(scale_fp32 * code). When we re-extract codes from
    bf16 W, a small fraction of elements sit on a rounding boundary where
    bf16 truncation causes the recovered code to differ by ±1, producing a
    per-element diff of one `scale` unit on those boundaries. The MEAN diff
    is the relevant metric for inference: it is well below bf16 LSB.

    For inference-level equivalence we also do a `y = x @ W` matmul check
    on a sample of layers, which is what actually matters downstream.

    sample=0 → check ALL layers; otherwise check the first `sample`.
    """
    print(f"\n=== Verification (numerical equivalence) ===")
    print(f"  Loading {src_ckpt} ...")
    state = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    print(f"  Loading {dst_ckpt} ...")
    manifest = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
    weights = manifest["weights"]
    layer_meta = manifest["layer_meta"]

    layer_names = list(layer_meta.keys())
    if sample > 0:
        layer_names = layer_names[:sample]

    layer_stats = []
    n_total_elems = 0
    n_total_shifted = 0
    matmul_diffs = []  # (name, rel_err) for sampled layers
    matmul_sample = max(1, len(layer_names) // 30)  # ~30 layers sampled

    for i, prefix in enumerate(layer_names):
        meta = layer_meta[prefix]
        K = meta["in_features"]
        n_low = meta["n_low"]
        hlen = meta["hlen"]
        gs = meta["groupsize"]

        packed = weights[f"{prefix}.module._w_packed"]
        scales = weights[f"{prefix}.module._w_scales"]
        tail = weights.get(f"{prefix}.module._w_tail")

        if meta.get("kind") == "per-head-int4":
            W_recon = unpack_per_head_int4(
                packed, scales, tail,
                num_heads=meta["num_heads"],
                head_dim=meta["head_dim"],
                hlen_per_head=meta["hlen_per_head"],
            )
        else:
            W_low_recon = unpack_int4_sym(packed, scales, gs, n_low)
            if tail is not None:
                W_recon = torch.cat([W_low_recon, tail], dim=1)
            else:
                W_recon = W_low_recon

        W_orig = state[f"{prefix}.module.weight"]
        diff = (W_orig.float() - W_recon.float()).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        n_shifted = (diff > 1e-4).sum().item()
        n_total_shifted += n_shifted
        n_total_elems += diff.numel()
        layer_stats.append({"name": prefix, "max": max_d, "mean": mean_d,
                            "n_shifted": n_shifted, "n_elems": diff.numel()})

        # Matmul-level check on a sample of layers — what really matters for inference.
        if matmul_check and i % matmul_sample == 0:
            torch.manual_seed(i)
            x = torch.randn(64, K, dtype=W_orig.dtype) * 0.5
            y_orig = (x @ W_orig.t()).float()
            y_recon = (x @ W_recon.t()).float()
            y_diff = (y_orig - y_recon).abs()
            rel = y_diff.mean().item() / max(y_orig.abs().mean().item(), 1e-9)
            matmul_diffs.append((prefix, rel,
                                 y_diff.mean().item(), y_orig.abs().mean().item()))

    if layer_stats:
        max_diff_overall = max(s["max"] for s in layer_stats)
        mean_diff_overall = sum(s["mean"] * s["n_elems"]
                                 for s in layer_stats) / n_total_elems
        # Estimate bf16 LSB at typical layer-weight magnitude
        avg_abs = sum(state[f"{s['name']}.module.weight"].abs().mean().item() * s["n_elems"]
                       for s in layer_stats) / n_total_elems
        bf16_lsb = avg_abs / 128  # ~7-bit mantissa

        print(f"  Layers checked            : {len(layer_stats)}")
        print(f"  Total weight elements     : {n_total_elems:,}")
        print(f"  Per-elem MEAN  abs diff   : {mean_diff_overall:.3e}  (the metric for inference)")
        print(f"  Per-elem MAX   abs diff   : {max_diff_overall:.3e}  (worst single boundary case)")
        print(f"  Avg |W|                   : {avg_abs:.3e}")
        print(f"  bf16 LSB at avg |W|       : ~{bf16_lsb:.3e}")
        ratio = mean_diff_overall / bf16_lsb if bf16_lsb > 0 else 0
        print(f"  MEAN diff / bf16 LSB      : {ratio:.2f}x  "
              f"({'within bf16 noise' if ratio < 2 else 'larger than bf16 LSB'})")
        print(f"  Elements with diff > 1e-4 : {n_total_shifted:,} "
              f"({n_total_shifted/n_total_elems*100:.3f}%) — bf16 boundary cases")

    if matmul_diffs:
        max_rel = max(d[1] for d in matmul_diffs)
        mean_rel = sum(d[1] for d in matmul_diffs) / len(matmul_diffs)
        print(f"\n  Matmul-level (y = x @ W) on {len(matmul_diffs)} sampled layers:")
        print(f"    Max  relative err on y : {max_rel:.3e}")
        print(f"    Mean relative err on y : {mean_rel:.3e}")
        worst = max(matmul_diffs, key=lambda d: d[1])
        print(f"    Worst layer            : {worst[0]} (rel_err={worst[1]:.3e})")
        print(f"  → Inference-level equivalence: {'✅ YES' if max_rel < 1e-2 else '⚠ check'} "
              "(matmul output sees the MEAN weight diff, not the max)")

    return {
        "max_diff": max(s["max"] for s in layer_stats) if layer_stats else 0.0,
        "mean_diff_per_elem": (sum(s["mean"] * s["n_elems"] for s in layer_stats)
                                / max(n_total_elems, 1)),
        "n_layers": len(layer_stats),
        "total_elems": n_total_elems,
        "shifted_elems": n_total_shifted,
        "matmul_max_rel": max(d[1] for d in matmul_diffs) if matmul_diffs else 0.0,
        "matmul_mean_rel": sum(d[1] for d in matmul_diffs) / max(len(matmul_diffs), 1) if matmul_diffs else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiRotQ fake-quant → int4 checkpoint")
    p.add_argument("--model", default="flux-dev")
    p.add_argument("--gptq", action="store_true",
                   help="Use the GPTQ cache filename (default: RTN).")
    p.add_argument("--nvfp4", action="store_true")
    p.add_argument("--src", default=None,
                   help="Source fake-quant checkpoint (default: derived from model).")
    p.add_argument("-o", "--output", default=None,
                   help="Destination int4 manifest path (default: same dir, "
                        "filename with .int4 suffix).")
    p.add_argument("--no-verify", dest="verify", action="store_false", default=True,
                   help="Skip numerical equivalence verification.")
    p.add_argument("--verify-sample", type=int, default=0,
                   help="Only verify the first N layers (0 = all). Use to "
                        "speed up sanity checks.")
    return p.parse_args()


def _default_src(model: str, *, gptq: bool, nvfp4: bool) -> Path:
    method = "gptq" if gptq else "rtn"
    fmt = "nvfp4" if nvfp4 else "int4"
    return _DIROTQ_ROOT / "models" / model / "quantized_cache" / \
        f"{fmt}_g64_{method}_model.pt"


def main() -> None:
    args = _parse_args()
    src = Path(args.src) if args.src else _default_src(
        args.model, gptq=args.gptq, nvfp4=args.nvfp4)
    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src}")

    if args.output is None:
        dst = src.with_name(src.stem + "_int4packed" + src.suffix)
    else:
        dst = Path(args.output)

    print(f"=== DiRotQ int4 checkpoint conversion ===")
    print(f"  model: {args.model}")
    print(f"  src:   {src}")
    print(f"  dst:   {dst}")
    print()

    summary = convert_checkpoint(model=args.model, src_ckpt=src, dst_ckpt=dst)

    if args.verify:
        verify_equivalence(src, dst, sample=args.verify_sample)


if __name__ == "__main__":
    main()
