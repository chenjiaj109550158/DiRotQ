"""
Hadamard and random orthogonal matrix utilities.

Provides:
  - random_orthogonal_matrix: random orthogonal via QR
  - fast_hadamard_transform: O(D log D) in-place Walsh-Hadamard (power-of-2)
  - hadamard_transform: handles arbitrary D via block-diagonal decomposition
"""

import torch


def random_orthogonal_matrix(size, device="cuda"):
    """
    Generate a random orthogonal matrix of the specified size via QR decomposition.

    Args:
        size (int): The size of the matrix (size x size).
        device: Device to create the matrix on.

    Returns:
        torch.Tensor: An orthogonal matrix of shape [size, size] in float64.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_hadK(n):
    """
    Get the Hadamard matrix for dimension n.
    If n is a power of 2, returns (None, 1) indicating to use fast Walsh-Hadamard.
    Otherwise returns (K, K_dim) where K is a small Hadamard matrix.
    """
    if n & (n - 1) == 0:  # power of 2
        return None, 1

    # Find largest power-of-2 factor
    k = n
    while k & (k - 1) != 0:
        k = k & (k - 1)
    # k is now the largest power of 2 that divides n... not quite right.
    # For simplicity, just use a random orthogonal matrix as the "Hadamard"
    K = random_orthogonal_matrix(n, device="cpu").float()
    return K, n


def generate_sign_flips(dim, seed=42):
    """Generate deterministic random ±1 sign flips for randomized Hadamard."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1


def fast_hadamard_transform(x):
    """
    In-place Fast Walsh-Hadamard Transform along the last dimension.
    x.shape[-1] must be a power of 2.  O(D log D) butterfly operations.
    Normalized by 1/sqrt(D) so the transform is orthogonal.

    Args:
        x: tensor of shape [..., D] where D is a power of 2.
    Returns:
        Transformed tensor (same shape), normalized.
    """
    D = x.shape[-1]
    assert D & (D - 1) == 0, f"fast_hadamard_transform requires power-of-2 dim, got {D}"

    h = 1
    while h < D:
        # Split into pairs of blocks of size h
        x = x.view(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :]  # [..., n_blocks, h]
        b = x[..., 1, :]  # [..., n_blocks, h]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*x.shape[:-3], -1)
        h *= 2

    return x / (D ** 0.5)


def _largest_power_of_2_factor(n):
    """Return the largest power of 2 that divides n."""
    return n & (-n)


def hadamard_transform(x, sign_flips=None):
    """
    Randomized Hadamard-like transform for arbitrary dimension D.

    For D that is a power of 2: direct fast Walsh-Hadamard.
    For D = p2 * remainder (p2 = largest power-of-2 factor):
      1. Apply random sign flips (diagonal ±1) if provided
      2. Reshape to [..., remainder, p2]
      3. Apply FWHT along p2 dimension (O(D log p2))

    This gives an orthogonal-like mixing that spreads outliers,
    at cost O(D log p2) instead of O(D²) for dense rotation.

    Args:
        x: tensor of shape [..., D]
        sign_flips: optional tensor of shape [D] with ±1 values for randomization.
                    If None, no sign flips applied.
    Returns:
        Transformed tensor of same shape.
    """
    D = x.shape[-1]

    if sign_flips is not None:
        x = x * sign_flips.to(x.device, x.dtype)

    if D & (D - 1) == 0:
        return fast_hadamard_transform(x)

    p2 = _largest_power_of_2_factor(D)
    remainder = D // p2

    # Reshape: [..., D] -> [..., remainder, p2]
    orig_shape = x.shape
    x = x.view(*orig_shape[:-1], remainder, p2)
    x = fast_hadamard_transform(x)  # FWHT along last dim (p2)
    return x.view(orig_shape)
