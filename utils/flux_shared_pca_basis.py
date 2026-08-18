"""Build auditable shared-PCA artifacts for FLUX.1-dev/schnell.

The FLUX quality path stores an eigensystem for every calibration source.  A
source eigensystem is sufficient to reconstruct its (damped) covariance, so
the sharing audit can pool the exact PCA provenance without depending on a
GPTQ Hessian (which is collected at a different point in the graph).

All tensors remain CPU tensors.  Aliases in the returned dictionary point to
one tensor object per sharing group; ``models/flux-schnell/model_utils.py``
preserves that sharing when it materializes ``U @ R``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


SCHEMES = (
    "shared-width",
    "shared-operator",
    "shared-operator-stage4",
    "representative-operator",
)

DOUBLE_FAMILIES = (
    "img_attn",
    "txt_attn",
    "img_attn.value",
    "txt_attn.value",
    "img_ffn",
    "txt_ffn",
    "img_ffn.down",
    "txt_ffn.down",
)
SINGLE_FAMILIES = (
    "attn",
    "mlp",
    "attn_out.value",
    "mlp.down",
)
DOWN_FAMILIES = {"img_ffn.down", "txt_ffn.down", "mlp.down"}


@dataclass(frozen=True)
class FluxBasisConfig:
    num_double_layers: int = 19
    num_single_layers: int = 38
    hidden: int = 3072
    intermediate: int = 12288
    damping: float = 0.01
    representative_double: int = 9
    representative_single: int = 19
    num_stages: int = 4


@dataclass(frozen=True)
class Source:
    kind: str
    block: int
    family: str
    key: str
    width: str


def iter_sources(cfg: FluxBasisConfig = FluxBasisConfig()):
    for block in range(cfg.num_double_layers):
        for family in DOUBLE_FAMILIES:
            yield Source(
                "double", block, family, f"layer.{block}.{family}",
                "down" if family in DOWN_FAMILIES else "hidden",
            )
    for block in range(cfg.num_single_layers):
        for family in SINGLE_FAMILIES:
            yield Source(
                "single", block, family, f"single.{block}.{family}",
                "down" if family in DOWN_FAMILIES else "hidden",
            )


def _stage(block: int, layers: int, stages: int) -> int:
    if not 0 <= block < layers:
        raise ValueError(f"block {block} outside [0,{layers})")
    return min(stages - 1, block * stages // layers)


def sharing_group(
    scheme: str, source: Source, cfg: FluxBasisConfig = FluxBasisConfig(),
) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"unsupported FLUX sharing scheme: {scheme}")
    if scheme == "shared-width":
        return source.width
    operator = f"{source.kind}:{source.family}"
    if scheme in {"shared-operator", "representative-operator"}:
        return operator
    layers = cfg.num_double_layers if source.kind == "double" else cfg.num_single_layers
    return f"{source.kind}:stage{_stage(source.block, layers, cfg.num_stages)}:{source.family}"


def _validate_source(source_basis: dict, cfg: FluxBasisConfig) -> None:
    for source in iter_sources(cfg):
        if source.key not in source_basis:
            raise KeyError(f"source PCA cache lacks {source.key}")
        if f"{source.key}.eigenvalues" not in source_basis:
            raise KeyError(f"source PCA cache lacks {source.key}.eigenvalues")
        width = cfg.intermediate if source.width == "down" else cfg.hidden
        if tuple(source_basis[source.key].shape) != (width, width):
            raise ValueError(
                f"{source.key}: expected {(width, width)}, "
                f"got {tuple(source_basis[source.key].shape)}"
            )


def _covariance(source_basis: dict, key: str) -> torch.Tensor:
    U = source_basis[key].detach().cpu().double()
    eigenvalues = source_basis[f"{key}.eigenvalues"].detach().cpu().double()
    return (U * eigenvalues.unsqueeze(0)) @ U.T


def _eigh(covariance: torch.Tensor, damping: float):
    covariance = (covariance + covariance.T) * 0.5
    mean_diag = covariance.diagonal().mean()
    covariance.diagonal().add_(damping * mean_diag)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors.float().contiguous(), eigenvalues.float().contiguous()


def _representative(source: Source, cfg: FluxBasisConfig) -> str:
    block = (
        cfg.representative_double
        if source.kind == "double" else cfg.representative_single
    )
    prefix = "layer" if source.kind == "double" else "single"
    return f"{prefix}.{block}.{source.family}"


def build_flux_shared_basis(
    source_basis: dict,
    scheme: str,
    *,
    cfg: FluxBasisConfig = FluxBasisConfig(),
    source_basis_sha256: str | None = None,
) -> tuple[dict, dict]:
    """Pool FLUX calibration eigensystems according to ``scheme``."""
    _validate_source(source_basis, cfg)
    sources = list(iter_sources(cfg))
    groups = sorted({sharing_group(scheme, source, cfg) for source in sources})
    canonical: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    group_sources: dict[str, list[str]] = {}

    for group in groups:
        members = [source for source in sources if sharing_group(scheme, source, cfg) == group]
        if scheme == "representative-operator":
            key = _representative(members[0], cfg)
            canonical[group] = (
                source_basis[key].detach().cpu().float().contiguous(),
                source_basis[f"{key}.eigenvalues"].detach().cpu().float().contiguous(),
            )
            group_sources[group] = [key]
            continue
        pooled = None
        for source in members:
            covariance = _covariance(source_basis, source.key)
            if pooled is None:
                pooled = torch.zeros_like(covariance)
            pooled.add_(covariance)
            del covariance
        assert pooled is not None
        pooled.div_(len(members))
        canonical[group] = _eigh(pooled, cfg.damping)
        group_sources[group] = [source.key for source in members]
        del pooled

    result: dict = {}
    alias_map: dict[str, str] = {}
    for source in sources:
        group = sharing_group(scheme, source, cfg)
        eigenvectors, eigenvalues = canonical[group]
        result[source.key] = eigenvectors
        result[f"{source.key}.eigenvalues"] = eigenvalues
        alias_map[source.key] = group
    result["__shared_basis_map__"] = alias_map
    result["__shared_basis_scheme__"] = scheme
    result["__shared_basis_model__"] = "flux"
    result["__shared_basis_version__"] = 1
    manifest = {
        "schema": "dirotq.flux_shared_pca_basis",
        "version": 1,
        "model": "flux",
        "scheme": scheme,
        "pooling": (
            "fixed_middle_representative_per_operator"
            if scheme == "representative-operator"
            else "equal_source_mean_reconstructed_from_pca_eigensystems"
        ),
        "damping": cfg.damping,
        "eigenvalue_order": "ascending",
        "source_basis_sha256": source_basis_sha256,
        "unique_group_count": len(groups),
        "alias_count": len(alias_map),
        "groups": group_sources,
        "config": cfg.__dict__,
    }
    return result, manifest


def validate_flux_shared_basis(
    mapping: dict,
    *,
    cfg: FluxBasisConfig = FluxBasisConfig(),
    atol: float = 5e-4,
) -> dict:
    alias_map = mapping.get("__shared_basis_map__")
    if not isinstance(alias_map, dict):
        raise ValueError("derived FLUX basis lacks __shared_basis_map__")
    group_ids: dict[str, set[int]] = {}
    max_orthogonality_error = 0.0
    for source in iter_sources(cfg):
        tensor = mapping[source.key]
        group = alias_map[source.key]
        group_ids.setdefault(group, set()).add(id(tensor))
        eye = torch.eye(tensor.shape[-1], dtype=tensor.dtype)
        error = (tensor.T @ tensor - eye).abs().max().item()
        max_orthogonality_error = max(max_orthogonality_error, error)
    bad = {group: len(ids) for group, ids in group_ids.items() if len(ids) != 1}
    if bad:
        raise ValueError(f"FLUX sharing aliases copied tensor objects: {bad}")
    if max_orthogonality_error > atol:
        raise ValueError(
            f"FLUX shared basis orthogonality error "
            f"{max_orthogonality_error:.3e} > {atol:.3e}"
        )
    unique_bytes = 0
    seen = set()
    for source in iter_sources(cfg):
        for key in (source.key, f"{source.key}.eigenvalues"):
            tensor = mapping[key]
            storage = tensor.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if identity not in seen:
                seen.add(identity)
                unique_bytes += storage.nbytes()
    return {
        "unique_groups": len(group_ids),
        "alias_count": len(alias_map),
        "max_orthogonality_error": max_orthogonality_error,
        "unique_tensor_bytes": unique_bytes,
    }


def theoretical_basis_bytes(
    scheme: str | None,
    *,
    cfg: FluxBasisConfig = FluxBasisConfig(),
    dtype_bytes: int = 2,
) -> dict:
    """Exact dense online-rotation bytes before allocator overhead.

    ``scheme=None`` is the per-source quality baseline.  Q/K/V wrappers that
    consume one calibration source are counted once because their rotation can
    and should alias one materialized tensor.
    """
    sources = list(iter_sources(cfg))
    if scheme is None:
        groups = {source.key: [source] for source in sources}
    else:
        groups = {}
        for source in sources:
            groups.setdefault(sharing_group(scheme, source, cfg), []).append(source)
    bytes_by_width = {"hidden": 0, "down": 0}
    for members in groups.values():
        width = members[0].width
        dimension = cfg.intermediate if width == "down" else cfg.hidden
        bytes_by_width[width] += dimension * dimension * dtype_bytes
    total = sum(bytes_by_width.values())
    return {
        "scheme": scheme or "per-layer-pca",
        "dtype_bytes": dtype_bytes,
        "unique_groups": len(groups),
        "hidden_bytes": bytes_by_width["hidden"],
        "down_bytes": bytes_by_width["down"],
        "total_bytes": total,
    }
