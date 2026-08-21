"""Rank-64 residual-rotation helpers for the FLUX shared-width Scheme A.

The saved shared-width PCA artifact contains the complete ascending
eigensystem.  Scheme A therefore changes only the high/low split and the
dimension of the data-independent residual random rotation; it never refits
PCA.  The production generator intentionally matches ``gen_rotation.py``.
"""

from __future__ import annotations

import torch

from .hadamard_utils import random_orthogonal_matrix


def build_hidden_residual_rotation(
    hidden_dim: int,
    high_rank: int,
    *,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Return ``block_diag(R_low, I_high)`` using the production algorithm."""
    if not 0 < high_rank < hidden_dim:
        raise ValueError("high_rank must be strictly between zero and hidden_dim")
    torch.manual_seed(seed)
    low = random_orthogonal_matrix(hidden_dim - high_rank, device)
    high = torch.eye(high_rank, dtype=low.dtype, device=device)
    return torch.block_diag(low, high).cpu()


def validate_hidden_residual_rotation(
    rotation: torch.Tensor,
    high_rank: int,
    *,
    atol: float = 2e-10,
) -> dict[str, float | int]:
    """Fail closed on cross-subspace mixing, non-identity tail or non-orthogonality."""
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("rotation must be a square matrix")
    if not 0 < high_rank < rotation.shape[0]:
        raise ValueError("invalid high_rank")
    low_rank = rotation.shape[0] - high_rank
    work = rotation.double()
    cross = max(
        float(work[:low_rank, low_rank:].abs().max()),
        float(work[low_rank:, :low_rank].abs().max()),
    )
    tail = float(
        (work[low_rank:, low_rank:] - torch.eye(high_rank, dtype=torch.float64))
        .abs().max()
    )
    gram = work[:low_rank, :low_rank].T @ work[:low_rank, :low_rank]
    orthogonality = float(
        (gram - torch.eye(low_rank, dtype=torch.float64)).abs().max()
    )
    if cross > atol or tail > atol or orthogonality > atol:
        raise RuntimeError(
            "invalid residual rotation: "
            f"cross={cross:.3g}, tail={tail:.3g}, orthogonality={orthogonality:.3g}"
        )
    return {
        "hidden_dim": int(rotation.shape[0]),
        "high_rank": int(high_rank),
        "low_rank": int(low_rank),
        "cross_subspace_max_abs": cross,
        "high_identity_max_abs": tail,
        "low_orthogonality_max_abs": orthogonality,
    }
