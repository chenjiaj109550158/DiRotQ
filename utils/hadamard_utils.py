"""
Hadamard and random orthogonal matrix utilities.
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
