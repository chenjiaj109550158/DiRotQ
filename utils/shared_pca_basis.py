"""Build auditable shared PCA bases from the existing pre-rotation Hessians.

The production accuracy path stores one eigensystem per PixArt block/operator.
This module derives alternative dictionaries whose aliases deliberately share
the same tensor storage.  It never modifies the source PCA or Hessian caches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


SCHEMES = (
    "shared-width",
    "shared-operator",
    "shared-operator-stage4",
    "representative-operator",
)

FAMILIES = (
    "self_attn",
    "cross_attn_q",
    "ffn",
    "self_attn.value",
    "cross_attn_q.value",
    "ffn.down_proj",
)

HEAD_FAMILIES = {"self_attn.value", "cross_attn_q.value"}
DOWN_FAMILIES = {"ffn.down_proj"}


@dataclass(frozen=True)
class PixArtBasisConfig:
    num_layers: int = 28
    hidden: int = 1152
    num_heads: int = 16
    head_dim: int = 72
    intermediate: int = 4608
    damping: float = 0.01
    representative_block: int = 14


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def basis_key(block: int, family: str) -> str:
    return f"layer.{block}.{family}"


def hessian_key(block: int, family: str) -> str:
    suffix = {
        "self_attn": "attn1.to_q",
        "cross_attn_q": "attn2.to_q",
        "ffn": "ff.net.0.proj",
        "self_attn.value": "attn1.to_out.0",
        "cross_attn_q.value": "attn2.to_out.0",
        "ffn.down_proj": "ff.net.2",
    }[family]
    return f"transformer_blocks.{block}.{suffix}"


def stage_for_block(block: int) -> int:
    if not 0 <= block < 28:
        raise ValueError(f"PixArt block index out of range: {block}")
    return block // 7


def sharing_group(scheme: str, block: int, family: str) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"unsupported shared-basis scheme: {scheme}")
    if family not in FAMILIES:
        raise ValueError(f"unsupported PixArt basis family: {family}")
    kind = "head" if family in HEAD_FAMILIES else "down" if family in DOWN_FAMILIES else "hidden"
    if scheme == "shared-width":
        return kind
    if scheme in {"shared-operator", "representative-operator"}:
        return family
    return f"stage{stage_for_block(block)}:{family}"


def _head_diagonal(H: torch.Tensor, cfg: PixArtBasisConfig) -> torch.Tensor:
    if tuple(H.shape) != (cfg.hidden, cfg.hidden):
        raise ValueError(f"per-head Hessian has wrong shape: {tuple(H.shape)}")
    H4 = H.reshape(cfg.num_heads, cfg.head_dim, cfg.num_heads, cfg.head_dim)
    idx = torch.arange(cfg.num_heads)
    return H4[idx, :, idx, :]


def _source_covariance(
    source_basis: dict, hessians: dict[str, torch.Tensor], block: int, family: str,
    cfg: PixArtBasisConfig,
) -> torch.Tensor:
    if family in HEAD_FAMILIES:
        # PixArt's published quality path derives output-projection PCA from
        # the per-head output of to_v, not from the later post-attention input
        # seen by to_out/GPTQ.  Reconstruct that exact covariance source from
        # its eigensystem.  The stored isotropic damping does not change the
        # pooled eigenvectors because it only adds a scalar identity per head.
        key = basis_key(block, family)
        U = source_basis[key].detach().cpu().double()
        evals = source_basis[f"{key}.eigenvalues"].detach().cpu().double()
        H = U @ torch.diag_embed(evals) @ U.transpose(-1, -2)
        return H
    key = hessian_key(block, family)
    if key not in hessians:
        raise KeyError(f"missing pre-rotation Hessian {key}")
    H = hessians[key].detach().to(device="cpu", dtype=torch.float64)
    expected = (
        (cfg.num_heads, cfg.head_dim, cfg.head_dim)
        if family in HEAD_FAMILIES else
        (cfg.intermediate, cfg.intermediate)
        if family in DOWN_FAMILIES else
        (cfg.hidden, cfg.hidden)
    )
    if tuple(H.shape) != expected:
        raise ValueError(f"{key}: expected {expected}, got {tuple(H.shape)}")
    return H


def _eigh(H: torch.Tensor, damping: float) -> tuple[torch.Tensor, torch.Tensor]:
    H = (H + H.transpose(-1, -2)) * 0.5
    mean_diag = H.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    eye = torch.eye(H.shape[-1], dtype=H.dtype)
    if H.ndim == 2:
        damped = H + damping * mean_diag * eye
    elif H.ndim == 3:
        damped = H + damping * mean_diag[:, None, None] * eye[None]
    else:
        raise ValueError(f"unsupported covariance rank: {H.ndim}")
    evals, evecs = torch.linalg.eigh(damped)
    return evecs.float().contiguous(), evals.float().contiguous()


def _members(scheme: str, group: str, cfg: PixArtBasisConfig) -> list[tuple[int, str]]:
    return [
        (block, family)
        for block in range(cfg.num_layers)
        for family in FAMILIES
        if sharing_group(scheme, block, family) == group
    ]


def _unique_groups(scheme: str, cfg: PixArtBasisConfig) -> list[str]:
    return sorted({
        sharing_group(scheme, block, family)
        for block in range(cfg.num_layers)
        for family in FAMILIES
    })


def _validate_source_basis(source: dict, cfg: PixArtBasisConfig) -> None:
    for block in range(cfg.num_layers):
        for family in FAMILIES:
            key = basis_key(block, family)
            if key not in source or f"{key}.eigenvalues" not in source:
                raise KeyError(f"source PCA cache lacks {key} or its eigenvalues")


def build_shared_basis(
    source_basis: dict,
    hessians: dict[str, torch.Tensor],
    scheme: str,
    *,
    cfg: PixArtBasisConfig = PixArtBasisConfig(),
    source_basis_sha256: str | None = None,
    hessian_sha256: str | None = None,
) -> tuple[dict, dict]:
    """Return a shared alias dictionary and a JSON-serializable manifest."""
    if scheme not in SCHEMES:
        raise ValueError(f"unsupported shared-basis scheme: {scheme}")
    _validate_source_basis(source_basis, cfg)

    canonical: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    group_sources: dict[str, list[str]] = {}
    for group in _unique_groups(scheme, cfg):
        members = _members(scheme, group, cfg)
        group_sources[group] = [basis_key(block, family) for block, family in members]
        if scheme == "representative-operator":
            # Pre-registered representative: block 14 for every operator family.
            family = members[0][1]
            if not 0 <= cfg.representative_block < cfg.num_layers:
                raise ValueError(
                    f"representative block {cfg.representative_block} is outside "
                    f"the configured {cfg.num_layers} layers"
                )
            src_key = basis_key(cfg.representative_block, family)
            canonical[group] = (
                source_basis[src_key].detach().cpu().float().contiguous(),
                source_basis[f"{src_key}.eigenvalues"].detach().cpu().float().contiguous(),
            )
            group_sources[group] = [src_key]
            continue

        # Stream into one accumulator.  A list+stack of all 28 4608-square
        # float64 Hessians temporarily exceeds 9 GiB and is unnecessary.
        pooled = None
        for block, family in members:
            covariance = _source_covariance(source_basis, hessians, block, family, cfg)
            if pooled is None:
                pooled = torch.zeros_like(covariance)
            pooled.add_(covariance)
            del covariance
        assert pooled is not None
        pooled.div_(len(members))
        canonical[group] = _eigh(pooled, cfg.damping)
        del pooled

    result: dict = {}
    alias_map: dict[str, str] = {}
    for block in range(cfg.num_layers):
        for family in FAMILIES:
            key = basis_key(block, family)
            group = sharing_group(scheme, block, family)
            evecs, evals = canonical[group]
            result[key] = evecs
            result[f"{key}.eigenvalues"] = evals
            alias_map[key] = group

    result["__shared_basis_map__"] = alias_map
    result["__shared_basis_scheme__"] = scheme
    result["__shared_basis_version__"] = 1
    manifest = {
        "schema": "dirotq.shared_pca_basis",
        "version": 1,
        "model": "pixart-sigma",
        "scheme": scheme,
        "pooling": (
            "pre_registered_block_14_representative"
            if scheme == "representative-operator" else
            "equal_source_mean_of_pre_rotation_hessians"
        ),
        "damping": cfg.damping,
        "eigenvalue_order": "ascending",
        "source_basis_sha256": source_basis_sha256,
        "hessian_sha256": hessian_sha256,
        "unique_group_count": len(canonical),
        "groups": group_sources,
        "alias_count": len(alias_map),
        "config": cfg.__dict__,
    }
    return result, manifest


def unique_tensor_bytes(mapping: dict) -> int:
    """Count unique CPU/CUDA storages, ignoring metadata entries."""
    seen: set[tuple] = set()
    total = 0
    for value in mapping.values():
        if not isinstance(value, torch.Tensor):
            continue
        storage = value.untyped_storage()
        key = (value.device.type, value.device.index, storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def rotation_storage_report(transformer) -> dict:
    """Report logical versus deduplicated online-rotation storage."""
    from .quant_utils import ActQuantWrapper

    assignments = []
    for module in transformer.modules():
        if not isinstance(module, ActQuantWrapper):
            continue
        tensor = module.rotation if module.rotation is not None else module.rotation_per_head
        if tensor is not None:
            assignments.append(tensor)
    seen = set()
    unique_bytes = 0
    for tensor in assignments:
        storage = tensor.untyped_storage()
        key = (
            tensor.device.type, tensor.device.index, storage.data_ptr(),
            storage.nbytes(),
        )
        if key not in seen:
            seen.add(key)
            unique_bytes += storage.nbytes()
    return {
        "assignments": len(assignments),
        "unique_storages": len(seen),
        "logical_assignment_bytes": sum(t.numel() * t.element_size() for t in assignments),
        "unique_storage_bytes": unique_bytes,
    }


def validate_shared_basis(
    mapping: dict, *, cfg: PixArtBasisConfig = PixArtBasisConfig(), atol: float = 2e-4,
) -> dict:
    alias_map = mapping.get("__shared_basis_map__")
    if not isinstance(alias_map, dict):
        raise ValueError("derived basis lacks __shared_basis_map__")
    max_orth = 0.0
    group_ids: dict[str, set[int]] = {}
    for block in range(cfg.num_layers):
        for family in FAMILIES:
            key = basis_key(block, family)
            U = mapping[key]
            group = alias_map[key]
            group_ids.setdefault(group, set()).add(id(U))
            eye = torch.eye(U.shape[-1], dtype=torch.float32)
            if U.ndim == 2:
                err = (U.T @ U - eye).abs().max().item()
            else:
                err = (U.transpose(-1, -2) @ U - eye[None]).abs().max().item()
            max_orth = max(max_orth, err)
    bad_aliases = {group: len(ids) for group, ids in group_ids.items() if len(ids) != 1}
    if bad_aliases:
        raise ValueError(f"sharing aliases do not reuse one tensor object: {bad_aliases}")
    if max_orth > atol:
        raise ValueError(f"shared basis orthogonality error {max_orth:.3e} > {atol:.3e}")
    return {
        "unique_groups": len(group_ids),
        "alias_count": len(alias_map),
        "max_orthogonality_error": max_orth,
        "unique_tensor_bytes": unique_tensor_bytes(mapping),
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def speedup_rotation_parity(
    x: torch.Tensor, stored_weight: torch.Tensor, rotation: torch.Tensor,
    *, counter_rotate_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small explicit oracle for the speed-script orientation audit.

    ``stored_weight`` uses PyTorch Linear layout [N,K].
    """
    reference = x @ stored_weight.T
    weight = stored_weight @ rotation
    if not counter_rotate_weight:
        weight = stored_weight
    candidate = (x @ rotation) @ weight.T
    return reference, candidate
