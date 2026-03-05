"""
Rotation utilities for DiRotQ.
Handles absorbing PCA basis rotations into model weights.
"""

import torch


def rotate_linear_input_side(linear, U):
    """
    Absorb rotation into linear layer's input side: W_new = W_old @ U
    This means the linear now expects inputs in the rotated basis.

    Args:
        linear: nn.Linear module
        U: [in_features, in_features] orthogonal matrix (float64)
    """
    dtype = linear.weight.data.dtype
    W = linear.weight.data.to(device=U.device, dtype=torch.float64)
    linear.weight.data = (W @ U).to(dtype=dtype)
    # Bias is not affected by input rotation


def rotate_linear_output_side(linear, U):
    """
    Absorb rotation into linear layer's output side: W_new = U^T @ W_old
    This maps the output back to the original (un-rotated) basis.

    Args:
        linear: nn.Linear module
        U: [out_features, out_features] orthogonal matrix (float64)
    """
    dtype = linear.weight.data.dtype
    W = linear.weight.data.to(device=U.device, dtype=torch.float64)
    linear.weight.data = (U.T @ W).to(dtype=dtype)
    if linear.bias is not None:
        b = linear.bias.data.to(device=U.device, dtype=torch.float64)
        linear.bias.data = (U.T @ b).to(dtype=dtype)


def rotate_linear_output_per_head(linear, U_per_head, num_heads, head_dim):
    """
    Apply per-head rotation to the output dimension of a projection (e.g., to_v).
    Weight shape: [num_heads * head_dim, in_features]

    After rotation, output[h] = U_per_head[h]^T @ output_original[h] for each head h.

    Args:
        linear: nn.Linear module with weight [H*d, in]
        U_per_head: [num_heads, head_dim, head_dim] per-head rotation (float64)
        num_heads: number of attention heads
        head_dim: dimension per head
    """
    dtype = linear.weight.data.dtype
    W = linear.weight.data.to(device=U_per_head.device, dtype=torch.float64)
    # W: [H*d, in] -> [H, d, in]
    W = W.view(num_heads, head_dim, -1)
    # Apply U^T on the output (row) dimension per head: U^T @ W
    W = torch.bmm(U_per_head.transpose(1, 2), W)  # [H, d, in]
    linear.weight.data = W.reshape(num_heads * head_dim, -1).to(dtype=dtype)

    if linear.bias is not None:
        b = linear.bias.data.to(device=U_per_head.device, dtype=torch.float64)
        b = b.view(num_heads, head_dim)  # [H, d]
        # U^T @ b per head
        b = torch.bmm(U_per_head.transpose(1, 2), b.unsqueeze(-1)).squeeze(-1)  # [H, d]
        linear.bias.data = b.reshape(num_heads * head_dim).to(dtype=dtype)


def rotate_linear_input_per_head(linear, U_per_head, num_heads, head_dim):
    """
    Apply per-head rotation to the input dimension of a projection (e.g., to_out[0]).
    Weight shape: [out_features, num_heads * head_dim]

    The input is organized as [head_0_d0, ..., head_0_d71, head_1_d0, ...].
    After rotation, the linear expects input in the per-head rotated basis.

    Args:
        linear: nn.Linear module with weight [out, H*d]
        U_per_head: [num_heads, head_dim, head_dim] per-head rotation (float64)
        num_heads: number of attention heads
        head_dim: dimension per head
    """
    dtype = linear.weight.data.dtype
    W = linear.weight.data.to(device=U_per_head.device, dtype=torch.float64)
    out_features = W.shape[0]
    # W: [out, H*d] -> [out, H, d]
    W = W.view(out_features, num_heads, head_dim)
    # Apply W @ U per head: einsum('ohi,hij->ohj', W, U)
    W = torch.einsum('ohd,hde->ohe', W, U_per_head)
    linear.weight.data = W.reshape(out_features, num_heads * head_dim).to(dtype=dtype)
    # Bias is on the output side, not affected by input rotation
