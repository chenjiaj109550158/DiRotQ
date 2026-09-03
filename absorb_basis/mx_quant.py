"""OCP MXFP4 (e2m1 elements, block-32, E8M0 power-of-two shared scale)
reference quantizers — the single source of truth for the MX guest round.

Spec (OCP Microscaling v1.0):
  shared scale X = 2^e,  e = floor(log2(max|v| in block)) - emax_elem,
  emax(e2m1) = 2;  elements = round-to-nearest-even e2m1 of v / X,
  saturating at +-6. All-zero blocks -> e = 0 (any scale works).

Weight side: static per (out_row, in_group32).  Act side: dynamic per
(token, in_group32).  `mx_weight_scales` feeds GPTQ via scales_override
(the GPTQ grid then exactly matches the deployed decode).
"""

import torch

E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
BLOCK = 32
EMAX_E2M1 = 2


def _e8m0_scale(absmax: torch.Tensor) -> torch.Tensor:
    """E8M0 shared scale per block: 2^(floor(log2(absmax)) - 2).
    floor(log2) is taken bit-exactly from the fp32 exponent field so the
    torch reference and the Triton kernel agree bitwise."""
    a = absmax.float()
    e = ((a.view(torch.int32) >> 23) & 0xFF) - 127 - EMAX_E2M1
    e = e.clamp(min=-127, max=127)
    s = torch.pow(2.0, e.float())
    return torch.where(a > 0, s, torch.ones_like(s))


def _round_e2m1(y: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(y.abs(), E2M1_BOUNDS.to(y.device))
    return E2M1_POS.to(y.device)[idx] * y.sign()


def mx_act_sim(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Dynamic MXFP4 activation quant-dequant (torch reference)."""
    shp = x.shape
    xg = x.float().reshape(-1, shp[-1] // block, block)
    s = _e8m0_scale(xg.abs().amax(dim=-1, keepdim=True))
    q = _round_e2m1(xg / s) * s
    return q.reshape(shp).to(x.dtype)


def mx_weight_scales(W: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Static per-group E8M0 scales for a weight [oc, ic] -> [oc, ic/block]."""
    oc, ic = W.shape
    g = W.float().reshape(oc, ic // block, block)
    return _e8m0_scale(g.abs().amax(dim=-1))


def mx_weight_sim(W: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Static MXFP4 weight quant-dequant (torch reference / RTN)."""
    oc, ic = W.shape
    g = W.float().reshape(oc, ic // block, block)
    s = mx_weight_scales(W, block).unsqueeze(-1)
    return (_round_e2m1(g / s) * s).reshape(oc, ic).to(W.dtype)


def mx_pack_weight(W_q: torch.Tensor, block: int = BLOCK):
    """Pack an ALREADY-on-grid dequantized weight into (codes uint8
    [oc, ic/2], exps int8 [oc, ic/block]). codes: two 4-bit fields per
    byte, low nibble = even column; nibble = sign(1) | pos_index(3)."""
    oc, ic = W_q.shape
    s = mx_weight_scales(W_q, block)  # scales are reconstructable from grid values
    g = W_q.float().reshape(oc, ic // block, block) / s.unsqueeze(-1)
    pos = E2M1_POS.to(W_q.device)
    idx = (g.abs().unsqueeze(-1) - pos.view(1, 1, 1, -1)).abs().argmin(dim=-1)
    nib = (idx | ((g < 0).long() << 3)).reshape(oc, ic).to(torch.uint8)
    codes = nib[:, 0::2] | (nib[:, 1::2] << 4)
    exps = torch.log2(s).round().to(torch.int8)
    return codes.contiguous(), exps.contiguous()


def mx_unpack_weight(codes: torch.Tensor, exps: torch.Tensor,
                     block: int = BLOCK) -> torch.Tensor:
    """Inverse of mx_pack_weight (torch reference for validation)."""
    oc = codes.shape[0]
    ic = codes.shape[1] * 2
    nib = torch.empty(oc, ic, dtype=torch.uint8, device=codes.device)
    nib[:, 0::2] = codes & 0xF
    nib[:, 1::2] = codes >> 4
    pos = E2M1_POS.to(codes.device)
    val = pos[(nib & 0x7).long()] * torch.where(nib & 0x8 > 0, -1.0, 1.0)
    s = torch.pow(2.0, exps.float()).repeat_interleave(block, dim=1)
    return val * s


@torch.no_grad()
def quantize_residual_mx(W_res: torch.Tensor, H: torch.Tensor, device: str,
                         gptq: bool = True, damp_pct: float = 0.01,
                         block_size: int = 128, prepared=None) -> torch.Tensor:
    """GPTQ (or RTN) on the exact MXFP4e2 grid: per-group-32 E8M0 scales
    passed as scales_override so the GPTQ grid equals the deployed decode.
    Returns the dequantized weight (on-grid values)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.gptq_utils import _gptq_quantize_layer
    eff = mx_weight_scales(W_res)  # [oc, ic/32]
    if gptq:
        W_q = _gptq_quantize_layer(
            W_res, H.to(device=device, dtype=torch.float32),
            bits=4, groupsize=BLOCK, sym=True, damp_pct=damp_pct,
            block_size=block_size, num_inv_tries=8, device=device,
            nvfp4=True, scales_override=eff, prepared=prepared)
        if W_q is not None:
            return W_q
        print("[warn] GPTQ failed; falling back to MX RTN")
    return mx_weight_sim(W_res)


if __name__ == "__main__":
    torch.manual_seed(0)
    # OCP spec checks
    x = torch.randn(64, 128) * torch.logspace(-3, 3, 128)
    q = mx_act_sim(x)
    g = x.reshape(-1, 4, 32)
    qg = q.reshape(-1, 4, 32)
    s = _e8m0_scale(g.abs().amax(-1, keepdim=True))
    assert torch.all(qg.abs() <= 6 * s + 1e-6), "saturation violated"
    ratio = (qg / s)
    grid = torch.cat([-E2M1_POS.flip(0), E2M1_POS])
    d = (ratio.unsqueeze(-1) - grid).abs().min(-1).values
    assert d.max() < 1e-5, "off-grid element"
    # pack/unpack roundtrip on a weight
    W = torch.randn(96, 256)
    Wq = mx_weight_sim(W)
    codes, exps = mx_pack_weight(Wq)
    assert torch.allclose(mx_unpack_weight(codes, exps), Wq.float(),
                          atol=0, rtol=0), "pack roundtrip failed"
    # scales are exact powers of two
    assert torch.all(torch.pow(2.0, exps.float()) ==
                     mx_weight_scales(Wq)), "scale reconstruct failed"
    print("MX_QUANT_SELFTEST_OK")
