"""bench_nunchaku_pack.py

Benchmark using the deepcompressor-port packer (`speedup/nunchaku_pack.py`) on
representative DiRotQ low-region shapes from flux-dev. Compares:
  - cuBLAS bf16 GEMM (baseline)
  - nunchaku gemm_w4a4 with weights packed via our copy of deepcompressor

This is the "use it to convert dirotq low prec weights to gemm_w4a4" step.

DiRotQ rotates the activation, then the W4A4 path acts on the low region
W[:, :n_low]. Here we sweep representative (M, K=n_low, N) triples for flux
double + single blocks and report (latency, speedup, rel_err vs same-recipe
fake-quant reference).
"""

import sys
import types
import torch

# Bypass nunchaku.__init__ (it pulls heavy diffusers/transformers deps).
fake_pkg = types.ModuleType('nunchaku')
fake_pkg.__path__ = ['/home/riftuser/miniconda3/envs/mldiffusion2/lib/python3.12/site-packages/nunchaku']
sys.modules['nunchaku'] = fake_pkg
fake_ops_pkg = types.ModuleType('nunchaku.ops')
fake_ops_pkg.__path__ = ['/home/riftuser/miniconda3/envs/mldiffusion2/lib/python3.12/site-packages/nunchaku/ops']
sys.modules['nunchaku.ops'] = fake_ops_pkg

from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speedup.nunchaku_pack import convert_dirotq_low_weight_to_nunchaku
from speedup.utils import run_timed


def fake_quant_reference(x, W, bias, gs=64, denom=7.0):
    """Same recipe as DiRotQ's INT4: per-group max/denom RTN."""
    M, K = x.shape
    N = W.shape[0]
    Wg = W.float().reshape(N, K // gs, gs)
    ws = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / denom
    W_qd = ((Wg / ws).round().clamp(-8, 7) * ws).reshape(N, K).to(W.dtype)
    Xg = x.float().reshape(M, K // gs, gs)
    as_ = Xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / denom
    X_qd = ((Xg / as_).round().clamp(-8, 7) * as_).reshape(M, K).to(x.dtype)
    return X_qd @ W_qd.t() + bias


def main():
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"{'shape':<25}{'M':>6}{'K':>6}{'N':>6}{'fp16':>10}{'nunchaku':>10}{'speedup':>10}{'rel_err':>12}")
    print('-' * 92)

    # Representative flux-dev shapes (low region after 10% high-bits split).
    shapes = [
        ('flux attn QKV',  4608, 3072, 3072),     # 90% of 3072 ≈ 2752, but pad to gs
        ('flux ff_up',     4608, 3072, 12288),
        ('flux ff_down',   4608, 12288, 3072),
        ('flux out_proj',  4608, 3072, 3072),
        ('flux attn KQV-2',4608, 3072, 9216),     # joint attn QKV*3
    ]

    for label, M, K, N in shapes:
        # Make K a multiple of 256 (mem_k * num_k_unrolls = 64*2 = 128, + gs=64).
        # For nunchaku int4: K % 128 == 0, N % 128 == 0.
        assert K % 128 == 0 and N % 128 == 0, f"{label}: K={K} N={N}"

        x = torch.randn(M, K, dtype=dtype, device=device) * 0.5
        W = torch.randn(N, K, dtype=dtype, device=device) * 0.05
        bias = torch.randn(N, dtype=dtype, device=device) * 0.01

        # Pack via deepcompressor port
        pack = convert_dirotq_low_weight_to_nunchaku(
            W_low=W, bias=bias, group_size=64, rank=32)

        # Quantize activations via nunchaku's op (fused with neutral LoRA)
        qact, ascales, lora_act_in = svdq_quantize_w4a4_act_fuse_lora_cuda(
            input=x.contiguous(), lora_down=pack['lora_down'], smooth=pack['smooth'],
            fp4=False, pad_size=256)

        out = torch.empty((qact.shape[0], N), dtype=dtype, device=device)

        def call_nunchaku():
            qa, asc, la = svdq_quantize_w4a4_act_fuse_lora_cuda(
                input=x.contiguous(), lora_down=pack['lora_down'],
                smooth=pack['smooth'], fp4=False, pad_size=256)
            svdq_gemm_w4a4_cuda(
                act=qa, wgt=pack['qweight'], out=out,
                ascales=asc, wscales=pack['wscales'],
                lora_act_in=la, lora_up=pack['lora_up'],
                bias=pack['bias'],
                lora_scales=[1.0] * (pack['rank'] // 16))

        # One-time correctness check
        call_nunchaku()
        out_M = out[:M]
        y_ref = fake_quant_reference(x, W, bias, gs=64)
        rel_err = ((out_M.float() - y_ref.float()).abs().mean()
                   / y_ref.float().abs().mean()).item()

        # Latencies
        sb = run_timed(lambda: torch.nn.functional.linear(x, W, bias),
                       warmup=10, repeats=30)['mean'] * 1000
        sn = run_timed(call_nunchaku, warmup=10, repeats=30)['mean'] * 1000

        print(f"{label:<25}{M:>6}{K:>6}{N:>6}{sb:>10.4f}{sn:>10.4f}"
              f"{sb/sn:>9.2f}x{rel_err:>12.2e}")


if __name__ == '__main__':
    main()
