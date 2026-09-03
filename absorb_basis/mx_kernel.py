"""Real-quant MXFP4e2 W4A4 linear for the MX guest round (PLAN_MX).

Storage: packed e2m1 codes (uint8, two per byte) + per-group-32 E8M0
exponents (int8). Forward: ONE Triton kernel fusing [dynamic per-token
group-32 act quantization] + [weight decode] + [GEMM]; fp16 tensor-core
dot with fp32 accumulation (e2m1 x e2m1 products are exact in fp16, so
the math is bit-faithful to true 4-bit MMA up to summation order). The
16-bit lora branch and optional smoothing run as plain torch GEMMs.

Validation (run this file):
  1. identity-weight trick: kernel output with W=I equals the torch
     reference `mx_act_sim(x)` BITWISE (isolates the fused act quant);
  2. full layer vs torch reference within fp16 summation tolerance.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from absorb_basis.mx_quant import mx_act_sim, mx_pack_weight, mx_unpack_weight


@triton.jit
def _decode_e2m1(idx):
    v = tl.where(idx == 1, 0.5, (1.0 + 0.5 * (idx % 2).to(tl.float32))
                 * tl.exp2(((idx // 2) - 1).to(tl.float32)))
    return tl.where(idx == 0, 0.0, v)


@triton.jit
def mx_gemm_kernel(x_ptr, codes_ptr, exps_ptr, y_ptr,
                   M, N, K,
                   sxm, sxk, scn, sck, sen, seg, sym, syn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        # ---- activations: load fp16 -> fused MX quant ----
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                    mask=(offs_m[:, None] < M), other=0.0).to(tl.float32)
        xg = tl.reshape(x, (BM, BK // 32, 32))
        amax = tl.max(tl.abs(xg), axis=2)
        e = ((amax.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 129
        e = tl.minimum(tl.maximum(e, -127), 127)
        s = tl.where(amax > 0, tl.exp2(e.to(tl.float32)), 1.0)
        y = xg / s[:, :, None]
        ay = tl.abs(y)
        # strict > matches torch.bucketize(right=False) tie-breaking
        idx = ((ay > 0.25).to(tl.int32) + (ay > 0.75) + (ay > 1.25)
               + (ay > 1.75) + (ay > 2.5) + (ay > 3.5) + (ay > 5.0))
        val = _decode_e2m1(idx) * tl.where(y < 0, -1.0, 1.0)
        xq = tl.reshape(val * s[:, :, None], (BM, BK)).to(tl.float16)
        # ---- weights: unpack codes + e8m0 exponents ----
        offs_kb = k0 // 2 + tl.arange(0, BK // 2)
        cb = tl.load(codes_ptr + offs_n[:, None] * scn + offs_kb[None, :] * sck,
                     mask=(offs_n[:, None] < N), other=0)
        nib_lo = (cb & 0xF).to(tl.int32)
        nib_hi = ((cb >> 4) & 0xF).to(tl.int32)
        nib = tl.interleave(nib_lo, nib_hi)  # [BN, BK]
        wval = _decode_e2m1(nib & 0x7) * tl.where((nib & 0x8) > 0, -1.0, 1.0)
        offs_kg = k0 // 32 + tl.arange(0, BK // 32)
        we = tl.load(exps_ptr + offs_n[:, None] * sen + offs_kg[None, :] * seg,
                     mask=(offs_n[:, None] < N), other=0).to(tl.float32)
        wg = tl.reshape(wval, (BN, BK // 32, 32)) * tl.exp2(we)[:, :, None]
        w = tl.reshape(wg, (BN, BK)).to(tl.float16)
        acc += tl.dot(xq, tl.trans(w))
    y_ptrs = y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn
    tl.store(y_ptrs, acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def mx_gemm(x: torch.Tensor, codes: torch.Tensor, exps: torch.Tensor,
            N: int) -> torch.Tensor:
    M, K = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float16)
    if M >= 512:
        BM, BN, BK, warps, stages = 128, 128, 64, 8, 4
    else:
        BM, BN, BK, warps, stages = 32, 64, 64, 4, 2
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    mx_gemm_kernel[grid](
        x, codes, exps, y, M, N, K,
        x.stride(0), x.stride(1), codes.stride(0), codes.stride(1),
        exps.stride(0), exps.stride(1), y.stride(0), y.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return y


class MXW4A4Linear(nn.Module):
    """Drop-in quantized linear: y = MXGEMM(x/s) + (x @ Ld) @ Lu^T + b."""

    def __init__(self, codes, exps, lora_down, lora_up, bias=None, smooth=None):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("exps", exps)
        self.register_buffer("lora_down", lora_down)   # [ic, r] fp16
        self.register_buffer("lora_up", lora_up)       # [oc, r] fp16
        self.register_buffer("bias_", bias)
        self.register_buffer("smooth", smooth)         # [ic] fp16 or None
        self.out_features = codes.shape[0]

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.float16)
        xm = x2 / self.smooth if self.smooth is not None else x2
        y = mx_gemm(xm.contiguous(), self.codes, self.exps, self.out_features)
        y = y + (x2 @ self.lora_down) @ self.lora_up.t()
        if self.bias_ is not None:
            y = y + self.bias_
        return y.reshape(*shp[:-1], self.out_features).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    # 1) identity-weight trick: isolates the fused act quantization
    K = 256
    I = torch.eye(K, device=dev)
    codes, exps = mx_pack_weight(I)
    x = (torch.randn(128, K, device=dev) * torch.logspace(-2, 2, K, device=dev)
         ).to(torch.float16)
    y = mx_gemm(x, codes, exps, K)
    ref = mx_act_sim(x)
    exact = torch.equal(y, ref.to(torch.float16))
    print(f"act-quant bitwise (identity W): {exact}")
    assert exact
    # 2) full layer vs torch reference
    oc, ic, M = 320, 512, 200
    W = torch.randn(oc, ic, device=dev)
    from absorb_basis.mx_quant import mx_weight_sim
    Wq = mx_weight_sim(W)
    codes, exps = mx_pack_weight(Wq)
    assert torch.allclose(mx_unpack_weight(codes, exps), Wq.float())
    x = (torch.randn(M, ic, device=dev)).to(torch.float16)
    y = mx_gemm(x, codes, exps, oc).float()
    ref = (mx_act_sim(x).float() @ Wq.float().t())
    rel = (y - ref).norm() / ref.norm()
    print(f"full layer rel err vs reference: {rel:.2e}")
    assert rel < 2e-3
    # 3) quick timing vs fp16 matmul
    import time
    x = torch.randn(4096, 1152, device=dev, dtype=torch.float16)
    W = torch.randn(4608, 1152, device=dev)
    codes, exps = mx_pack_weight(mx_weight_sim(W))
    Wh = W.half()
    for _ in range(3):
        mx_gemm(x, codes, exps, 4608); x @ Wh.t()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50):
        mx_gemm(x, codes, exps, 4608)
    torch.cuda.synchronize(); t1 = time.time()
    for _ in range(50):
        x @ Wh.t()
    torch.cuda.synchronize(); t2 = time.time()
    print(f"mx_gemm {1000*(t1-t0)/50:.3f} ms vs fp16 matmul {1000*(t2-t1)/50:.3f} ms")
    print("MX_KERNEL_SELFTEST_OK")
