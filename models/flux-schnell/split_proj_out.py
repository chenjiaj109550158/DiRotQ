"""
Split the fused FluxSingleTransformerBlock.proj_out into two Linear branches
(mirrors SVDQuant's ConcatLinear).

The fused proj_out is a single nn.Linear with in_features = dim + mlp_hidden
(3072 + 12288 = 15360) that is applied to concat([attn_out, mlp_out]). Having
one layer with one per-token scale across two heterogeneous input distributions
is a known quality hit; SVDQuant splits them so the attention-out branch and
the MLP-down branch each get their own rotation, their own per-token scale,
and their own mixed-precision slice.

After splitting, each block has:
  proj_out.linears[0] : nn.Linear(dim,        dim, bias=False)
  proj_out.linears[1] : nn.Linear(mlp_hidden, dim, bias=True)
  out = linears[0](x[..., :dim]) + linears[1](x[..., dim:])
"""

import torch
import torch.nn as nn


class FluxProjOutSplit(nn.Module):
    """Drop-in replacement for the fused FluxSingleTransformerBlock.proj_out.

    Behaviour matches the original Linear on concat([attn_out, mlp_out]):
        y = W[:, :dim] @ attn + W[:, dim:] @ mlp + bias
    but the two halves live in separate nn.Linear children so downstream
    code (activation quantizers, GPTQ, rotations) treats them as two layers.
    """

    def __init__(self, attn_in: int, mlp_in: int, out: int, bias: bool,
                 device=None, dtype=None):
        super().__init__()
        self.attn_in = attn_in
        self.mlp_in = mlp_in
        self.in_features = attn_in + mlp_in
        self.out_features = out
        self.linears = nn.ModuleList([
            nn.Linear(attn_in, out, bias=False,
                      device=device, dtype=dtype),
            nn.Linear(mlp_in, out, bias=bias,
                      device=device, dtype=dtype),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_part = x[..., :self.attn_in].contiguous()
        mlp_part  = x[..., self.attn_in:].contiguous()
        return self.linears[0](attn_part) + self.linears[1](mlp_part)

    @staticmethod
    def from_linear(linear: nn.Linear, attn_in: int) -> "FluxProjOutSplit":
        """Construct from a fused nn.Linear whose weight columns are
        [attn (:attn_in) || mlp (attn_in:)]."""
        assert isinstance(linear, nn.Linear)
        total_in = linear.in_features
        mlp_in = total_in - attn_in
        assert mlp_in > 0, f"Expected mlp_in > 0, got in_features={total_in}, attn_in={attn_in}"
        split = FluxProjOutSplit(
            attn_in=attn_in,
            mlp_in=mlp_in,
            out=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            split.linears[0].weight.copy_(linear.weight[:, :attn_in])
            split.linears[1].weight.copy_(linear.weight[:, attn_in:])
            if linear.bias is not None:
                split.linears[1].bias.copy_(linear.bias)
        return split


def split_flux_single_proj_out(transformer) -> int:
    """Replace each FluxSingleTransformerBlock.proj_out with FluxProjOutSplit.

    Idempotent: blocks whose proj_out is already a FluxProjOutSplit are skipped.
    Returns the number of blocks that were converted this call.
    """
    blocks = getattr(transformer, "single_transformer_blocks", None)
    if blocks is None:
        return 0

    n = 0
    for block in blocks:
        proj = block.proj_out
        if isinstance(proj, FluxProjOutSplit):
            continue
        if not isinstance(proj, nn.Linear):
            raise TypeError(
                f"Unexpected proj_out type {type(proj).__name__}; "
                "expected nn.Linear or FluxProjOutSplit"
            )
        # The attention-out side has width `dim` (hidden). Recover it from
        # the block's own norm (dim == norm.linear.weight.shape[1] / 3 for
        # AdaLayerNormZeroSingle) or fall back to proj_out.out_features
        # (dim == out_features for FluxSingleTransformerBlock).
        attn_in = proj.out_features
        block.proj_out = FluxProjOutSplit.from_linear(proj, attn_in=attn_in)
        n += 1
    return n


def is_flux_split_applied(transformer) -> bool:
    blocks = getattr(transformer, "single_transformer_blocks", None)
    if blocks is None or len(blocks) == 0:
        return False
    return isinstance(blocks[0].proj_out, FluxProjOutSplit)
