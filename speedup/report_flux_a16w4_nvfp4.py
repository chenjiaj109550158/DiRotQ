"""Flux-dev A16W4 (weight-only NVFP4) speedup + memory.

Weight-only point of comparison to the full W4A4 NVFP4 run
(report_flux_nvfp4_no_rot_ffdown.py): weights are NVFP4 (4-bit), activations
stay bf16, and there is NO rotation (weight-only quant doesn't need the
activation-outlier rotation that A4 does).

Measured: ~2x weight-memory reduction, speedup ~1.0x (0.96-0.98x) — on a
compute-bound workload (flux, M=4608) the matmul stays bf16, so weight-only
4-bit buys memory/bandwidth, not compute. Contrast with the W4A4 run, which
quantizes activations too and gets the ~2x compute speedup.

There is no Blackwell mixed bf16xFP4 GEMM, so each quantized Linear
dequantizes its 4-bit weight to bf16 (a fused Triton kernel, see
speedup/weight_only_nvfp4.py) and runs cuBLAS. The dequant adds only ~15 ms
total, so the forward stays at ~fp16 speed.

Two subprocess phases (fp16 native + A16W4), combined into one JSON.

Output: speedup/results/flux_a16w4_nvfp4.json
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_DIROTQ_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TMP_FP16_JSON = RESULTS_DIR / "_tmp_fp16_phase_a16w4.json"
TMP_W4_JSON = RESULTS_DIR / "_tmp_a16w4_phase.json"
FINAL_JSON = RESULTS_DIR / "flux_a16w4_nvfp4.json"

WEIGHTS_FP16_GB = 23.80
DTYPE = torch.bfloat16
DEVICE = 'cuda'

# Norm/modulation linears stay fp16 (DiRotQ does not quantize modulators) —
# matches the layers the W4A4 run leaves unquantized.
_SKIP_SUFFIXES = ('norm1.linear', 'norm1_context.linear', 'norm.linear')


def _stub_torchao():
    try:
        import torchao.quantization as _t
    except ImportError:
        return
    class _Stub:
        pass
    for n in ('Float8WeightOnlyConfig', 'Float8DynamicActivationFloat8WeightConfig'):
        if not hasattr(_t, n):
            setattr(_t, n, _Stub)


def make_inputs(batch=1):
    return {
        'hidden_states': torch.randn(batch, 4096, 64, dtype=DTYPE, device=DEVICE),
        'encoder_hidden_states': torch.randn(batch, 512, 4096, dtype=DTYPE, device=DEVICE),
        'pooled_projections': torch.randn(batch, 768, dtype=DTYPE, device=DEVICE),
        'timestep': torch.tensor([500] * batch, dtype=torch.long, device=DEVICE),
        'guidance': torch.tensor([3.5] * batch, dtype=DTYPE, device=DEVICE),
        'img_ids': torch.randint(0, 64, (4096, 3), dtype=DTYPE, device=DEVICE),
        'txt_ids': torch.zeros(512, 3, dtype=DTYPE, device=DEVICE),
        'return_dict': False,
    }


def time_fwd(model, inputs, *, warmup=2, repeats=4):
    for _ in range(warmup):
        model(**inputs)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(**inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    drop = max(1, len(times) // 5)
    trimmed = times[drop:-drop] if len(times) > 2 * drop else times
    mean = sum(trimmed) / len(trimmed)
    var = sum((t - mean) ** 2 for t in trimmed) / max(1, len(trimmed) - 1)
    return mean, var ** 0.5


def measure(model, B, **kw):
    inp = make_inputs(B)
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        ms, std = time_fwd(model, inp, **kw)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del inp; gc.collect(); torch.cuda.empty_cache()
    return ms * 1000, std * 1000, peak


# ===========================================================================
# Phase 1: fp16 baseline (native at every batch — 96 GB)
# ===========================================================================

def run_phase_fp16():
    _stub_torchao()
    print("=== Phase 1: fp16 (native, no offload) ===", flush=True)
    from diffusers import FluxTransformer2DModel
    m = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer',
        torch_dtype=DTYPE).to(DEVICE).eval()
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    results = {'native': {}}
    for B in [1, 2, 4]:
        ms, std, peak = measure(m, B, warmup=2, repeats=4)
        results['native'][B] = {'ms': ms, 'std_ms': std, 'peak_gb': peak}
        print(f"  fp16 B={B} (native): {ms:>7.1f} ± {std:.1f} ms, peak {peak:.2f} GB", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()
    with open(TMP_FP16_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {TMP_FP16_JSON}", flush=True)


# ===========================================================================
# Phase 2: A16W4 (weight-only NVFP4) — no rotation, no nunchaku
# ===========================================================================

def _set_submodule(root, qualified_name, new_mod):
    parent = root
    *path, last = qualified_name.split('.')
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, last, new_mod)


def run_phase_a16w4():
    _stub_torchao()
    print("=== Phase 2: A16W4 (weight-only NVFP4, no rotation) ===", flush=True)
    if str(_DIROTQ_ROOT) not in sys.path:
        sys.path.insert(0, str(_DIROTQ_ROOT))
    from diffusers import FluxTransformer2DModel
    from speedup.weight_only_nvfp4 import WeightOnlyNVFP4Linear

    m = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer',
        torch_dtype=DTYPE).to(DEVICE).eval()

    # Replace quantizable Linears inside the transformer blocks (skip modulators).
    targets = []
    for name, mod in m.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not (name.startswith('transformer_blocks')
                or name.startswith('single_transformer_blocks')):
            continue
        if any(name.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        targets.append(name)

    print(f"  quantizing {len(targets)} Linears to weight-only NVFP4...", flush=True)
    w_bytes = 0
    for i, name in enumerate(targets):
        lin = m.get_submodule(name)
        q = WeightOnlyNVFP4Linear.from_linear(lin)
        w_bytes += q.weight_bytes()
        _set_submodule(m, name, q)
        del lin
        if i % 40 == 0:
            gc.collect(); torch.cuda.empty_cache()
    gc.collect(); torch.cuda.empty_cache()
    w4_weight_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  quantized-weight bytes (codes+scales): {w_bytes/1e9:.2f} GB", flush=True)
    print(f"  total model footprint after quant: {w4_weight_gb:.2f} GB", flush=True)

    results = {'measurements': {}, 'w4_weight_gb': w4_weight_gb,
               'quant_weight_gb': w_bytes / 1e9, 'n_quantized': len(targets),
               'source': 'weight-only NVFP4 (group-16, per-channel global scale), fused-Triton-dequant + cuBLAS, no rotation'}
    for B in [1, 2, 4]:
        try:
            ms, std, peak = measure(m, B, warmup=2, repeats=4)
            results['measurements'][B] = {'ms': ms, 'std_ms': std, 'peak_gb': peak}
            print(f"  A16W4 B={B}: {ms:>7.1f} ± {std:.1f} ms, peak {peak:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            results['measurements'][B] = {'oom': True, 'error': str(e)[:200]}
            print(f"  A16W4 B={B}: OOM", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    with open(TMP_W4_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {TMP_W4_JSON}", flush=True)


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_orchestrator():
    print("Running fp16 phase in subprocess...")
    if subprocess.run([sys.executable, '-u', __file__, '--phase=fp16']).returncode != 0:
        sys.exit("fp16 phase failed")
    print("\nRunning A16W4 phase in subprocess...")
    if subprocess.run([sys.executable, '-u', __file__, '--phase=a16w4']).returncode != 0:
        sys.exit("a16w4 phase failed")

    with open(TMP_FP16_JSON) as f:
        fp16 = json.load(f)
    with open(TMP_W4_JSON) as f:
        w4 = json.load(f)

    print("\n" + "=" * 78)
    print("Flux-dev A16W4 (weight-only NVFP4) — results")
    print("=" * 78)
    print("Hardware: NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120)")
    print("A16W4: bf16 activations, NVFP4 weights (4-bit), no rotation.")
    print("fp16 baseline is NATIVE at every batch.\n")
    print(f"{'B':>3}  {'fp16 (ms)':>12}  {'A16W4 (ms)':>12}  "
          f"{'fp16 peak GB':>13}  {'A16W4 peak GB':>14}  {'speedup':>9}")
    print('-' * 78)
    summary = []
    for B in [1, 2, 4]:
        f16 = fp16['native'].get(str(B)) or fp16['native'].get(B)
        ww = w4['measurements'].get(str(B)) or w4['measurements'].get(B)
        if not f16 or not ww or 'oom' in ww:
            continue
        sp = f16['ms'] / ww['ms']
        print(f"{B:>3}  {f16['ms']:>12.0f}  {ww['ms']:>12.0f}  "
              f"{f16['peak_gb']:>13.2f}  {ww['peak_gb']:>14.2f}  {sp:>8.2f}x")
        summary.append({'B': B, 'fp16_ms': f16['ms'], 'a16w4_ms': ww['ms'],
                        'fp16_peak_gb': f16['peak_gb'], 'a16w4_peak_gb': ww['peak_gb'],
                        'speedup': sp})

    # ---- Memory breakdown: peak VRAM = resident weights + activations/working ----
    # fp16   weights = 23.80 GB; A16W4 weights = measured resident model footprint
    # after quant (4-bit linears + the fp16 parts left unquantized: norms, embeds).
    a16w4_weights = w4.get('w4_weight_gb')
    print()
    print("Memory breakdown (peak VRAM = weights + activations + working):")
    print(f"{'B':>3}  {'fp16 W':>8}{'fp16 A':>8}{'fp16 tot':>10}"
          f"  {'A16W4 W':>9}{'A16W4 A':>9}{'A16W4 tot':>11}"
          f"  {'tot ratio':>10}")
    print('-' * 72)
    for r in summary:
        f_total = r['fp16_peak_gb']
        f_acts = f_total - WEIGHTS_FP16_GB
        n_total = r['a16w4_peak_gb']
        n_acts = n_total - a16w4_weights
        r['fp16_weights_gb'] = WEIGHTS_FP16_GB
        r['fp16_acts_gb'] = f_acts
        r['a16w4_weights_gb'] = a16w4_weights
        r['a16w4_acts_gb'] = n_acts
        r['mem_ratio'] = f_total / n_total
        print(f"{r['B']:>3}  {WEIGHTS_FP16_GB:>8.2f}{f_acts:>8.2f}{f_total:>10.2f}"
              f"  {a16w4_weights:>9.2f}{n_acts:>9.2f}{n_total:>11.2f}"
              f"  {r['mem_ratio']:>9.2f}×")

    print()
    print(f"Weights (resident): fp16 {WEIGHTS_FP16_GB:.2f} GB  vs  A16W4 {a16w4_weights:.2f} GB"
          f"  →  {WEIGHTS_FP16_GB/a16w4_weights:.2f}× smaller")
    print(f"  (quantized linears only: {w4.get('quant_weight_gb'):.2f} GB of codes+scales "
          f"across {w4.get('n_quantized')} layers; norms/embeds stay fp16)")
    if summary:
        acts = [r['a16w4_acts_gb'] for r in summary if r['a16w4_acts_gb'] > 0]
        f_acts = [r['fp16_acts_gb'] for r in summary if r['a16w4_acts_gb'] > 0]
        if acts:
            print(f"Activations: bf16 in both paths (A16 keeps activations 16-bit) — "
                  f"~equal ({sum(f_acts)/sum(acts):.2f}× avg).")

    out = {
        "report": "Flux-dev A16W4 (weight-only NVFP4)",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        "hardware_vram_gb": 96,
        "source": w4.get('source'),
        "weights_gb": {"fp16": WEIGHTS_FP16_GB,
                       "a16w4_quant_only": w4.get('quant_weight_gb'),
                       "a16w4_model_footprint": w4.get('w4_weight_gb')},
        "fp16_phase": fp16,
        "a16w4_phase": w4,
        "summary": summary,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(FINAL_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {FINAL_JSON}")
    for p in (TMP_FP16_JSON, TMP_W4_JSON):
        try:
            p.unlink()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('fp16', 'a16w4'), default=None)
    args = ap.parse_args()
    if args.phase == 'fp16':
        run_phase_fp16()
    elif args.phase == 'a16w4':
        run_phase_a16w4()
    else:
        run_orchestrator()


if __name__ == "__main__":
    main()
