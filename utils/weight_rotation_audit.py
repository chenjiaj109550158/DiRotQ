"""Read-only SANA residual-weight distribution audit helpers.

This module compares PCA-only (identity residual R) and PCA plus cached random
R.  It never runs denoising, GPTQ, or cache writes, and never retains raw
weights beyond the current layer.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from .hardware_weight_fp4 import (
    E0_VALUES,
    E2_VALUES,
    frozen_block_scales,
    hardware_global_scale,
    payload_indices,
    quantize_with_frozen_scales,
)
from .weight_mixfp4 import hessian_trace_loss


BLOCK = 16
TILE_K = 64
TILE_N = 8
CREST_EDGES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
MAG_HIST_BINS = 32
CREST_FINE_BINS = 3000
KURTOSIS_FINE_BINS = 4096


def file_snapshot(paths: list[Path]) -> dict[str, dict]:
    output = {}
    for path in paths:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        output[str(path)] = {
            "sha256": digest.hexdigest(), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return output


def assert_files_unchanged(before: dict[str, dict]) -> None:
    after = file_snapshot([Path(path) for path in before])
    if before != after:
        changed = [path for path in before if before[path] != after.get(path)]
        raise RuntimeError(f"read-only audit changed input files: {changed}")


def layer_kind(name: str) -> tuple[str, str]:
    suffix = name.split("transformer_blocks.", 1)[-1].split(".", 1)[-1]
    if suffix in {"attn1.to_q", "attn1.to_k", "attn1.to_v"}:
        return "self_attention", "input_projection"
    if suffix == "attn1.to_out.0":
        return "self_attention", "output_projection"
    if suffix == "attn2.to_q":
        return "cross_attention", "input_projection"
    if suffix == "attn2.to_out.0":
        return "cross_attention", "output_projection"
    raise ValueError(f"unsupported SANA active layer name: {name}")


def basis_for_layer(name: str, basis: dict, rotation: dict):
    parts = name.split(".")
    block_index = int(parts[parts.index("transformer_blocks") + 1])
    suffix = ".".join(parts[parts.index("transformer_blocks") + 2:])
    if suffix in {"attn1.to_q", "attn1.to_k", "attn1.to_v"}:
        return basis[f"layer.{block_index}.self_attn"], rotation["R1"], False
    if suffix == "attn1.to_out.0":
        return basis[f"layer.{block_index}.self_attn.value"], rotation["R2"], True
    if suffix == "attn2.to_q":
        return basis[f"layer.{block_index}.cross_attn"], rotation["R1"], False
    if suffix == "attn2.to_out.0":
        return basis[f"layer.{block_index}.cross_attn.value"], rotation["R2"], True
    raise ValueError(f"unsupported SANA active layer name: {name}")


def orthogonality_error(rotation_low: torch.Tensor) -> float:
    identity = torch.eye(
        rotation_low.shape[-1], device=rotation_low.device, dtype=torch.float32
    )
    return float((rotation_low.float().T @ rotation_low.float() - identity).abs().max())


@torch.no_grad()
def transform_weight_hessian_pair(
    stored_weight: torch.Tensor,
    hessian_pre: torch.Tensor,
    pca_basis: torch.Tensor,
    residual_rotation: torch.Tensor,
    high_length: int,
    *,
    per_head: bool,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Build stored low weights and matching full-precision Hessians."""
    weight = stored_weight.float()
    hpre = hessian_pre.float()
    if hpre.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("pre-wrapper Hessian/weight input shape mismatch")
    if per_head:
        heads, dim, _ = pca_basis.shape
        low = dim - high_length
        if weight.shape[1] != heads * dim or residual_rotation.shape != (dim, dim):
            raise ValueError("per-head basis/weight/rotation shape mismatch")
        u_low = pca_basis.float()[..., :low]
        r_low = residual_rotation.float()[:low, :low]
        weight_3d = weight.reshape(weight.shape[0], heads, dim)
        wid_3d = torch.einsum("ohd,hdk->ohk", weight_3d, u_low)
        wid = wid_3d.reshape(weight.shape[0], heads * low).contiguous()
        wrand = torch.einsum("ohk,kl->ohl", wid_3d, r_low).reshape_as(wid).contiguous()

        h4 = hpre.reshape(heads, dim, heads, dim)
        temp = torch.einsum("adi,adbe->aibe", u_low, h4)
        hid4 = torch.einsum("aibe,bej->aibj", temp, u_low)
        hid = hid4.reshape(heads * low, heads * low).contiguous()
        identity_presymmetry = float(
            (hid-hid.T).abs().max() / hid.abs().max().clamp_min(1e-30)
        )
        # H_pre is exactly symmetric.  The antisymmetric residue introduced by
        # the two FP32 projection GEMMs is numerical noise, so canonicalize the
        # mathematically equivalent quadratic form before rotating it again.
        hid = ((hid + hid.T) * .5).contiguous()
        hid4 = hid.reshape(heads, low, heads, low)
        temp = torch.einsum("ki,akbl->aibl", r_low, hid4)
        hrand = torch.einsum("aibl,lj->aibj", temp, r_low).reshape_as(hid).contiguous()
        random_presymmetry = float(
            (hrand-hrand.T).abs().max() / hrand.abs().max().clamp_min(1e-30)
        )
        hrand = ((hrand + hrand.T) * .5).contiguous()
        del h4, temp, hid4

        gram = torch.bmm(u_low.transpose(1, 2), u_low)
        pca_orthogonality = float(
            (gram-torch.eye(low, device=gram.device).expand_as(gram)).abs().max()
        )

        generator = torch.Generator(device=weight.device).manual_seed(20260812)
        x = torch.randn(4, heads, dim, generator=generator, device=weight.device)
        xid = torch.einsum("mhd,hdk->mhk", x, u_low).reshape(4, heads * low)
        xrand = torch.einsum(
            "mhk,kl->mhl", xid.reshape(4, heads, low), r_low
        ).reshape_as(xid)
    else:
        dim = pca_basis.shape[0]
        low = dim - high_length
        if pca_basis.shape != (dim, dim) or residual_rotation.shape != (dim, dim):
            raise ValueError("hidden basis/rotation shape mismatch")
        u_low = pca_basis.float()[:, :low]
        r_low = residual_rotation.float()[:low, :low]
        wid = (weight @ u_low).contiguous()
        wrand = (wid @ r_low).contiguous()
        hid = (u_low.T @ hpre @ u_low).contiguous()
        identity_presymmetry = float(
            (hid-hid.T).abs().max() / hid.abs().max().clamp_min(1e-30)
        )
        hid = ((hid + hid.T) * .5).contiguous()
        hrand = (r_low.T @ hid @ r_low).contiguous()
        random_presymmetry = float(
            (hrand-hrand.T).abs().max() / hrand.abs().max().clamp_min(1e-30)
        )
        hrand = ((hrand + hrand.T) * .5).contiguous()
        pca_orthogonality = float(
            (u_low.T@u_low-torch.eye(low, device=u_low.device)).abs().max()
        )
        generator = torch.Generator(device=weight.device).manual_seed(20260812)
        x = torch.randn(4, dim, generator=generator, device=weight.device)
        xid = x @ u_low
        xrand = xid @ r_low

    output_id = xid @ wid.T
    output_rand = xrand @ wrand.T
    output_error = float(
        (output_id - output_rand).abs().max()
        / output_id.abs().max().clamp_min(1e-30)
    )
    norm_error = float(
        (wid.norm() - wrand.norm()).abs() / wid.norm().clamp_min(1e-30)
    )
    trace_error = float(
        (hid.trace() - hrand.trace()).abs() / hid.trace().abs().clamp_min(1e-30)
    )

    psd = {}
    for mode, hessian in (("identity", hid), ("random", hrand)):
        symmetry = float(
            (hessian - hessian.T).abs().max() / hessian.abs().max().clamp_min(1e-30)
        )
        generator = torch.Generator(device=weight.device).manual_seed(20260813)
        probes = torch.randn(8, hessian.shape[0], generator=generator, device=weight.device)
        quadratic = (probes * (probes @ hessian)).sum(1)
        psd[mode] = {
            "symmetry_relative_max": symmetry,
            "minimum_sampled_quadratic": float(quadratic.min()),
            "minimum_diagonal": float(hessian.diagonal().min()),
            "trace": float(hessian.trace()),
        }

    validation = {
        "residual_orthogonality_max_abs": orthogonality_error(r_low),
        "pca_low_orthogonality_max_abs": pca_orthogonality,
        "projection_presymmetry_identity_relative_max": identity_presymmetry,
        "projection_presymmetry_random_relative_max": random_presymmetry,
        "weight_frobenius_relative_error": norm_error,
        "hessian_trace_relative_error": trace_error,
        "unquantized_output_relative_max_error": output_error,
        "psd_sanity": psd,
        "low_dimension": wid.shape[1],
        "high_dimension_excluded": high_length if not per_head else pca_basis.shape[0] * high_length,
    }
    return {"identity": (wid, hid), "random": (wrand, hrand)}, validation


def weight_candidate(
    source: torch.Tensor, fmt: str, *, rounded: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = hardware_global_scale(source)
    rounded_scales, raw_scales = frozen_block_scales(source, fmt, alpha)
    scales = rounded_scales if rounded else torch.where(
        raw_scales == 0, torch.ones_like(raw_scales), raw_scales
    )
    quantized = quantize_with_frozen_scales(source, fmt, alpha, scales)
    indices = payload_indices(source, fmt, alpha, scales)[:, :source.shape[1]]
    maximum = 6.0 if fmt.endswith("e2") else 7.0
    group_index = torch.arange(source.shape[1], device=source.device) // BLOCK
    normalized = source.float() / (alpha * scales[:, group_index])
    saturation = normalized.abs() > maximum
    return quantized, scales, raw_scales, indices, saturation


def logical_tile_to_stored_slice(k_tile: int, n_tile: int):
    if k_tile < 0 or n_tile < 0:
        raise ValueError("tile indices must be non-negative")
    return (
        slice(n_tile * TILE_N, (n_tile + 1) * TILE_N),
        slice(k_tile * TILE_K, (k_tile + 1) * TILE_K),
    )


def tile_scores(
    source: torch.Tensor, e2: torch.Tensor, e0: torch.Tensor,
) -> dict:
    if source.shape != e2.shape or source.shape != e0.shape or source.ndim != 2:
        raise ValueError("tile tensors must have identical stored [N,K] shapes")
    n, k = source.shape
    npad, kpad = math.ceil(n / TILE_N) * TILE_N, math.ceil(k / TILE_K) * TILE_K
    def padded_error(candidate):
        error = torch.nn.functional.pad(source.float() - candidate.float(), (0, kpad-k, 0, npad-n))
        return error.reshape(npad // TILE_N, TILE_N, kpad // TILE_K, TILE_K).permute(0, 2, 1, 3).double().square().sum((-1, -2))
    se2, se0 = padded_error(e2), padded_error(e0)
    choose_e0 = se0 < se2
    selected = torch.where(choose_e0, se0, se2)
    return {
        "tile_count": choose_e0.numel(),
        "e0_tile_count": int(choose_e0.sum()),
        "e2_tile_count": choose_e0.numel() - int(choose_e0.sum()),
        "e2_sse": float(se2.sum()), "e0_sse": float(se0.sum()),
        "selected_sse": float(selected.sum()),
    }


@torch.no_grad()
def analyze_weight_basis(
    source: torch.Tensor, hessian: torch.Tensor,
) -> tuple[dict, dict]:
    """Analyze one stored low weight matrix without retaining raw tensors."""
    if source.device.type != "cuda" or hessian.device.type != "cuda":
        raise RuntimeError("production weight audit requires CUDA; no CPU fallback")
    if source.ndim != 2 or hessian.shape != (source.shape[1], source.shape[1]):
        raise ValueError("weight/Hessian shape mismatch")
    if not torch.isfinite(source).all() or not torch.isfinite(hessian).all():
        raise ValueError("non-finite weight audit input")
    n, k = source.shape
    alpha = hardware_global_scale(source)
    candidates = {}
    for rounded in (False, True):
        label = "rounded" if rounded else "exact"
        for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
            candidates[(label, fmt)] = weight_candidate(source, fmt, rounded=rounded)

    kpad = math.ceil(k / BLOCK) * BLOCK
    original = torch.nn.functional.pad(source.float(), (0, kpad-k))
    valid = torch.zeros_like(original, dtype=torch.bool)
    valid[:, :k] = True
    blocks = original.reshape(n, -1, BLOCK)
    valid_blocks = valid.reshape_as(blocks)
    valid_count = valid_blocks.sum(-1).clamp_min(1)
    absmax = blocks.abs().amax(-1)
    nonzero = absmax > 0
    rms = torch.sqrt((blocks.square() * valid_blocks).sum(-1) / valid_count)
    crest = torch.where(nonzero, absmax / rms.clamp_min(1e-30), torch.zeros_like(absmax))
    mean = (blocks * valid_blocks).sum(-1) / valid_count
    centered = (blocks - mean.unsqueeze(-1)) * valid_blocks
    variance = centered.square().sum(-1) / valid_count
    kurtosis = torch.where(
        variance > 0,
        centered.pow(4).sum(-1) / valid_count / variance.square(),
        torch.zeros_like(variance),
    )

    block_errors = {}
    for label in ("exact", "rounded"):
        for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
            q = candidates[(label, fmt)][0]
            qpad = torch.nn.functional.pad(q.float(), (0, kpad-k)).reshape_as(blocks)
            block_errors[(label, fmt)] = (
                (blocks - qpad).square() * valid_blocks
            ).sum(-1).double()

    loss = {}
    for label in ("exact", "rounded"):
        for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
            q = candidates[(label, fmt)][0]
            loss[(label, fmt, "raw")] = float((source.float() - q).double().square().sum())
            loss[(label, fmt, "hessian")] = float(hessian_trace_loss(source, q, hessian))

    rounded_e2 = candidates[("rounded", "hardware-fixed-e2")][0]
    rounded_e0 = candidates[("rounded", "hardware-fixed-e0")][0]
    tiles = tile_scores(source, rounded_e2, rounded_e0)
    best_fixed = min(tiles["e2_sse"], tiles["e0_sse"])
    tiles["e0_tile_ratio"] = tiles["e0_tile_count"] / tiles["tile_count"]
    tiles["gain_vs_best_fixed"] = (
        (best_fixed - tiles["selected_sse"]) / best_fixed if best_fixed > 0 else None
    )

    crest_bin = torch.bucketize(
        crest[nonzero], torch.tensor(CREST_EDGES[1:-1], device=source.device)
    ).clamp(0, len(CREST_EDGES)-2)
    crest_counts = torch.bincount(crest_bin, minlength=7)
    crest_wins = {}
    for label in ("exact", "rounded"):
        e0win = block_errors[(label, "hardware-fixed-e0")] < block_errors[(label, "hardware-fixed-e2")]
        crest_wins[label] = torch.bincount(
            crest_bin, weights=e0win[nonzero].double(), minlength=7
        )

    normalized = torch.where(
        absmax.unsqueeze(-1) > 0,
        blocks.abs() / absmax.unsqueeze(-1).clamp_min(1e-30),
        torch.zeros_like(blocks),
    )
    norm_index = (normalized[valid_blocks] * MAG_HIST_BINS).floor().long().clamp(0, MAG_HIST_BINS-1)

    detail = {
        "crest_counts": crest_counts.cpu().tolist(),
        "crest_exact_e0_wins": crest_wins["exact"].cpu().tolist(),
        "crest_rounded_e0_wins": crest_wins["rounded"].cpu().tolist(),
        "normalized_magnitude_hist": torch.bincount(norm_index, minlength=MAG_HIST_BINS).cpu().tolist(),
        "crest_fine_hist": torch.histc(
            crest[nonzero], bins=CREST_FINE_BINS, min=1.0, max=4.0
        ).long().cpu().tolist(),
        "kurtosis_fine_hist": torch.histc(
            kurtosis[nonzero].clamp(0, 16), bins=KURTOSIS_FINE_BINS,
            min=0.0, max=16.0,
        ).long().cpu().tolist(),
        "codebook_occupancy": {}, "block_scale_histogram": {},
    }
    for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
        q, scales, raw, indices, saturation = candidates[("rounded", fmt)]
        detail["codebook_occupancy"][fmt] = torch.bincount(
            indices.flatten().long(), minlength=15
        ).cpu().tolist()
        unique, counts = torch.unique(scales.float(), return_counts=True)
        detail["block_scale_histogram"][fmt] = {
            str(value): int(count) for value, count in zip(unique.cpu().tolist(), counts.cpu().tolist())
        }

    nonzero_count = int(nonzero.sum())
    block_count = blocks.shape[0] * blocks.shape[1]
    row = {
        "n": n, "k": k, "global_scale": float(alpha),
        "frobenius_norm": float(source.float().norm()),
        "block_count": block_count, "nonzero_block_count": nonzero_count,
        "element_count": n*k, "zero_element_count": int((source == 0).sum()),
        "zero_rate": float((source == 0).float().mean()),
        "zero_block_rate": (block_count - nonzero_count) / block_count,
        "crest_mean": float(crest[nonzero].mean()) if nonzero_count else 0.0,
        "crest_median": float(crest[nonzero].median()) if nonzero_count else 0.0,
        "kurtosis_mean": float(kurtosis[nonzero].mean()) if nonzero_count else 0.0,
        "kurtosis_median": float(kurtosis[nonzero].median()) if nonzero_count else 0.0,
        **tiles,
    }
    for label in ("exact", "rounded"):
        e2b = block_errors[(label, "hardware-fixed-e2")]
        e0b = block_errors[(label, "hardware-fixed-e0")]
        row[f"{label}_block_e0_win_rate"] = float((e0b < e2b).double().mean())
        row[f"{label}_block_e0_win_count"] = int((e0b < e2b).sum())
        for kind in ("raw", "hessian"):
            e2 = loss[(label, "hardware-fixed-e2", kind)]
            e0 = loss[(label, "hardware-fixed-e0", kind)]
            row[f"{label}_{kind}_e2_loss"] = e2
            row[f"{label}_{kind}_e0_loss"] = e0
            row[f"{label}_{kind}_e0_advantage"] = (e2-e0)/e2 if e2 else None
    exact_win = block_errors[("exact", "hardware-fixed-e0")] < block_errors[("exact", "hardware-fixed-e2")]
    rounded_win = block_errors[("rounded", "hardware-fixed-e0")] < block_errors[("rounded", "hardware-fixed-e2")]
    row["exact_rounded_winner_disagreement_rate"] = float((exact_win != rounded_win).double().mean())
    for fmt in ("hardware-fixed-e2", "hardware-fixed-e0"):
        _, scales, raw, _, saturation = candidates[("rounded", fmt)]
        nz = raw > 0
        rel = (scales[nz] - raw[nz]).abs() / raw[nz] if nz.any() else torch.zeros(1, device=source.device)
        prefix = "e2" if fmt.endswith("e2") else "e0"
        row[f"{prefix}_scale_min"] = float(scales.min())
        row[f"{prefix}_scale_max"] = float(scales.max())
        row[f"{prefix}_scale_rounding_relative_mean"] = float(rel.mean())
        row[f"{prefix}_scale_rounding_relative_max"] = float(rel.max())
        row[f"{prefix}_saturation_rate"] = float(saturation.double().mean())
    return row, detail


__all__ = [
    "CREST_EDGES", "CREST_FINE_BINS", "E0_VALUES", "E2_VALUES",
    "KURTOSIS_FINE_BINS", "MAG_HIST_BINS",
    "analyze_weight_basis", "assert_files_unchanged", "basis_for_layer",
    "file_snapshot", "layer_kind", "logical_tile_to_stored_slice",
    "orthogonality_error", "tile_scores", "transform_weight_hessian_pair",
    "weight_candidate",
]
