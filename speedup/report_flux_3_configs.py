"""3-config Flux-dev W4A4 comparison.

Runs all three configurations end-to-end and produces a combined report:

  1. fp16 baseline (same as the 2-config script — sequential CPU offload
     for B≥2 since fp16 OOMs at higher batches).

  2. A16W4 baseline — int4 weights, bf16 activations, NO rotation. Only
     the layers DiRotQ would quantize get the int4 treatment (same K∈{3072,
     12288, 15360} eligibility); modulators / norms / embedders stay
     bf16 (W16A16). Uses `torch._weight_int4pack_mm`. This shows the
     speedup from int4 weights ALONE — useful as a comparison point for
     "what does rotation buy you" against config 3.

  3. DiRotQ W4A4 with no rotation on K=12288 ff_down — the existing path
     from `report_flux_w4a4_no_rot_ffdown.py` (nunchaku v2 fused kernels
     + DiRotQ rotation injected before every K=3072 op). This is the
     full DiRotQ-on-SVDQuant-kernel measurement.

Phases 1 and 3 are delegated via subprocess to the existing
`report_flux_w4a4_no_rot_ffdown.py` script (which writes `_tmp_fp16_phase.json`
and `_tmp_nunchaku_phase.json`). Phase 2 is implemented here and writes
`_tmp_a16w4_phase.json`. The orchestrator combines all three.

Usage:
  python -u speedup/report_flux_3_configs.py
  # or single-phase debug:
  python -u speedup/report_flux_3_configs.py --phase=a16w4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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

# Reuse the same temp paths as the existing script for phases 1 and 3
TMP_FP16_JSON = RESULTS_DIR / "_tmp_fp16_phase.json"
TMP_NU_JSON = RESULTS_DIR / "_tmp_nunchaku_phase.json"
TMP_A16W4_JSON = RESULTS_DIR / "_tmp_a16w4_phase.json"
FINAL_JSON = RESULTS_DIR / "flux_3_configs.json"

EXISTING_SCRIPT = _HERE / "report_flux_w4a4_no_rot_ffdown.py"

PCIE_BW_GB_S = 25.9
WEIGHTS_FP16_GB = 23.80

DTYPE = torch.bfloat16
DEVICE = 'cuda'


def _stub_torchao():
    import torchao.quantization as _t
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
# A16W4 baseline (no rotation; only DiRotQ-eligible Linears get int4 weights)
# ===========================================================================

# Same K eligibility as DiRotQ-W4A4 phase
ELIGIBLE_K = {3072, 12288, 15360}


def _is_quantizable(name: str, mod: nn.Linear) -> bool:
    K = mod.in_features
    N = mod.out_features
    if name == 'proj_out':
        return False
    if name.endswith(('.norm.linear', '.norm1.linear', '.norm1_context.linear')):
        return False
    if name.startswith(('x_embedder', 'context_embedder', 'time_text_embed')):
        return False
    if K not in ELIGIBLE_K:
        return False
    if N % 8 != 0:
        return False
    if K % 64 != 0:
        return False
    return True


class W4A16Linear(nn.Module):
    """W4A16 baseline using bitsandbytes' NF4 4-bit Linear.

    Why bnb (not `torch._weight_int4pack_mm`): the PyTorch built-in is
    optimized for LLM batch-1 token decoding (M=1) and is ~9× slower than
    cuBLAS bf16 at M=4096. bnb's NF4 kernel uses tensor-core MMA with
    fused dequant and matches fp16 to within ~5% at M≥1024 — closer to
    what SVDQuant's paper baseline reports. No rotation (this is the
    A16W4 baseline, not DiRotQ).
    """

    def __init__(self, fp_linear: nn.Linear, quant_type: str = 'nf4'):
        super().__init__()
        import bitsandbytes as bnb
        self.K = fp_linear.in_features
        self.N = fp_linear.out_features
        self._lin = bnb.nn.Linear4bit(
            self.K, self.N,
            bias=fp_linear.bias is not None,
            quant_type=quant_type,
            compute_dtype=DTYPE,
        )
        # Install raw fp16 weight via Params4bit; .to(cuda) triggers quant.
        self._lin.weight = bnb.nn.Params4bit(
            fp_linear.weight.data.detach().clone().to('cpu'),
            quant_type=quant_type, requires_grad=False)
        if fp_linear.bias is not None:
            self._lin.bias = nn.Parameter(
                fp_linear.bias.data.clone(), requires_grad=False)
        self._lin.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.K)
        y = self._lin(x_flat)
        return y.reshape(*orig_shape[:-1], self.N)


def run_phase_a16w4():
    _stub_torchao()
    print("=== Phase 2: A16W4 baseline (no rotation) ===", flush=True)

    from diffusers import FluxTransformer2DModel

    print("Loading fp16 FluxTransformer2DModel (CPU)...", flush=True)
    m = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer',
        torch_dtype=DTYPE).eval()

    print("Walking model, replacing eligible Linears with W4A16Linear...", flush=True)
    n_replaced = 0
    n_kept = 0
    by_K = {}

    def walk(parent, parent_name=''):
        nonlocal n_replaced, n_kept
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear):
                if _is_quantizable(full, child):
                    K = child.in_features
                    setattr(parent, child_name, W4A16Linear(child))
                    n_replaced += 1
                    by_K[K] = by_K.get(K, 0) + 1
                else:
                    n_kept += 1
            else:
                walk(child, full)
    walk(m)
    gc.collect(); torch.cuda.empty_cache()
    print(f"  replaced {n_replaced} Linears (by K: {by_K}); kept fp16: {n_kept}", flush=True)

    m.to(DEVICE)
    a16w4_weight_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  A16W4 weight footprint: {a16w4_weight_gb:.2f} GB", flush=True)

    results = {'measurements': {}, 'a16w4_weight_gb': a16w4_weight_gb,
               'by_K': by_K,
               'source': 'A16W4 baseline (bitsandbytes NF4), no rotation'}

    for B in [1, 2, 4]:
        try:
            ms, std, peak = measure(m, B, warmup=2, repeats=4)
            results['measurements'][B] = {
                'ms': ms, 'std_ms': std, 'peak_gb': peak, 'offload': False}
            print(f"  A16W4 B={B}: {ms:>7.1f} ± {std:.1f} ms, peak {peak:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  A16W4 B={B}: OOM — installing accelerate.cpu_offload and retrying...", flush=True)
            gc.collect(); torch.cuda.empty_cache()
            try:
                from accelerate import cpu_offload
                blocks = list(m.transformer_blocks) + list(m.single_transformer_blocks)
                for blk in blocks:
                    cpu_offload(blk, execution_device=DEVICE, offload_buffers=True)
                torch.cuda.empty_cache()
                ms, std, peak = measure(m, B, warmup=1, repeats=2)
                results['measurements'][B] = {
                    'ms': ms, 'std_ms': std, 'peak_gb': peak, 'offload': True}
                print(f"  A16W4 B={B} (offload): {ms:>7.1f} ± {std:.1f} ms, peak {peak:.2f} GB", flush=True)
            except torch.cuda.OutOfMemoryError as e2:
                results['measurements'][B] = {'oom': True, 'error': str(e2)[:200]}
                print(f"  A16W4 B={B}: OOM even with offload", flush=True)
                gc.collect(); torch.cuda.empty_cache()

    with open(TMP_A16W4_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {TMP_A16W4_JSON}", flush=True)


# ===========================================================================
# Orchestrator: run all 3 phases as separate subprocesses + combine
# ===========================================================================

def run_orchestrator():
    print("Running fp16 phase (delegated to existing script)...")
    r = subprocess.run([sys.executable, '-u', str(EXISTING_SCRIPT), '--phase=fp16'])
    if r.returncode != 0:
        sys.exit(f"fp16 phase failed (exit {r.returncode})")

    print("\nRunning A16W4 baseline phase (this script)...")
    r = subprocess.run([sys.executable, '-u', str(_HERE / 'report_flux_3_configs.py'),
                        '--phase=a16w4'])
    if r.returncode != 0:
        sys.exit(f"a16w4 phase failed (exit {r.returncode})")

    print("\nRunning DiRotQ W4A4 phase (delegated to existing script)...")
    r = subprocess.run([sys.executable, '-u', str(EXISTING_SCRIPT), '--phase=nunchaku'])
    if r.returncode != 0:
        sys.exit(f"nunchaku phase failed (exit {r.returncode})")

    with open(TMP_FP16_JSON) as f:
        fp16 = json.load(f)
    with open(TMP_A16W4_JSON) as f:
        a16w4 = json.load(f)
    with open(TMP_NU_JSON) as f:
        nu = json.load(f)

    print()
    print("=" * 92)
    print("Flux-dev — fp16  vs  A16W4 baseline (no rot)  vs  DiRotQ-W4A4 (no-rot ff_down)")
    print("=" * 92)
    print("Hardware: NVIDIA GeForce RTX 4090 (24 GB)")
    print()

    # ---- baseline / weight footprints ----
    fp16_b1_native = fp16['native'].get('1') or fp16['native'].get(1)
    fp16_act_b1 = fp16_b1_native['peak_gb'] - WEIGHTS_FP16_GB
    a16w4_w = a16w4['a16w4_weight_gb']
    nu_w = nu['w4a4_weight_gb']

    # ---- speed table ----
    print(f"{'B':>3}  {'fp16 (ms)':>16}  {'A16W4 (ms)':>15}  {'DiRotQ-W4A4 (ms)':>18}  "
          f"{'A16W4 sp':>10}  {'W4A4 sp':>9}")
    print('-' * 92)
    summary_rows = []
    for B in [1, 2, 4]:
        fp16_native = fp16['native'].get(str(B)) or fp16['native'].get(B)
        fp16_offload = fp16['offload'].get(str(B)) or fp16['offload'].get(B)
        a16_meas = a16w4['measurements'].get(str(B)) or a16w4['measurements'].get(B)
        nu_meas = nu['measurements'].get(str(B)) or nu['measurements'].get(B)

        if B == 1 and fp16_native:
            fp16_ms = fp16_native['ms']
            fp16_label = "native"
        elif fp16_offload:
            fp16_ms = fp16_offload['ms']
            fp16_label = "offload"
        else:
            continue

        if not a16_meas or 'oom' in a16_meas:
            a16_ms = None; a16_peak = None; a16_label = "OOM"
        else:
            a16_ms = a16_meas['ms']
            a16_peak = a16_meas['peak_gb']
            a16_label = "offload" if a16_meas.get('offload') else "native"
        if not nu_meas or 'oom' in nu_meas:
            nu_ms = None; nu_peak = None
        else:
            nu_ms = nu_meas['ms']
            nu_peak = nu_meas['peak_gb']

        a16_sp = (fp16_ms / a16_ms) if a16_ms else None
        nu_sp = (fp16_ms / nu_ms) if nu_ms else None

        ns = f"{fp16_ms:.0f} ({fp16_label})"
        a16_str = f"{a16_ms:.0f} ({a16_label})" if a16_ms else "OOM"
        nu_str = f"{nu_ms:.0f}" if nu_ms else "—"
        a16_sp_s = f"{a16_sp:.2f}x" if a16_sp else "—"
        nu_sp_s = f"{nu_sp:.2f}x" if nu_sp else "—"
        print(f"{B:>3}  {ns:>16}  {a16_str:>15}  {nu_str:>18}  "
              f"{a16_sp_s:>10}  {nu_sp_s:>9}")
        summary_rows.append({
            'B': B,
            'fp16_ms': fp16_ms, 'fp16_label': fp16_label,
            'a16w4_ms': a16_ms, 'a16w4_peak_gb': a16_peak, 'a16w4_label': a16_label,
            'a16w4_speedup': a16_sp,
            'w4a4_ms': nu_ms, 'w4a4_peak_gb': nu_peak,
            'w4a4_speedup': nu_sp,
        })

    # ---- memory breakdown ----
    print()
    print("Memory breakdown (peak VRAM = weights + activations + working):")
    print(f"{'B':>3}  {'fp16 total':>11}  "
          f"{'A16W4 W':>9}{'A16W4 A':>9}{'A16W4 total':>13}  "
          f"{'W4A4 W':>9}{'W4A4 A':>9}{'W4A4 total':>12}")
    print('-' * 88)
    for r in summary_rows:
        B = r['B']
        f_acts = fp16_act_b1 * B if B > 1 else fp16_act_b1
        f_total = WEIGHTS_FP16_GB + f_acts
        if r['a16w4_peak_gb'] is not None:
            a_total = r['a16w4_peak_gb']; a_acts = a_total - a16w4_w
        else:
            a_total = a_acts = None
        if r['w4a4_peak_gb'] is not None:
            n_total = r['w4a4_peak_gb']; n_acts = n_total - nu_w
        else:
            n_total = n_acts = None
        a_total_s = f"{a_total:>13.2f}" if a_total else f"{'OOM':>13}"
        a_acts_s = f"{a_acts:>9.2f}" if a_acts is not None else f"{'—':>9}"
        n_total_s = f"{n_total:>12.2f}" if n_total else f"{'—':>12}"
        n_acts_s = f"{n_acts:>9.2f}" if n_acts is not None else f"{'—':>9}"
        print(f"{B:>3}  {f_total:>11.2f}  "
              f"{a16w4_w:>9.2f}{a_acts_s}{a_total_s}  "
              f"{nu_w:>9.2f}{n_acts_s}{n_total_s}")

    # ---- Subtract nunchaku kernel-API overhead from W4A4 weight footprint ----
    # SVDQW4A4Linear stores LoRA (proj_down [K,R], proj_up [N,R]) + smooth
    # (smooth_factor [K], smooth_factor_orig [K]) — DiRotQ uses none of these.
    # We pass zeros / ones to satisfy the kernel signature. Compute the
    # allocated overhead analytically from flux's known layer shapes.
    RANK = 32
    BF16 = 2  # bytes
    def per_layer_dummy(K, N):
        # smooth_factor + smooth_factor_orig: 2*K*bf16
        # proj_down: K*R*bf16   proj_up: N*R*bf16
        return 2 * K * BF16 + K * RANK * BF16 + N * RANK * BF16
    DOUBLE = [(3072, 9216), (3072, 3072), (3072, 9216), (3072, 3072),
              (3072, 12288), (12288, 3072), (3072, 12288), (12288, 3072)]
    SINGLE = [(3072, 9216), (3072, 12288), (12288, 3072), (3072, 3072)]
    dummy_bytes = (19 * sum(per_layer_dummy(K, N) for K, N in DOUBLE) +
                    38 * sum(per_layer_dummy(K, N) for K, N in SINGLE))
    dummy_gb = dummy_bytes / 1e9
    nu_w_effective = nu_w - dummy_gb

    print()
    print(f"Weights (raw allocated):")
    print(f"  fp16   {WEIGHTS_FP16_GB:.2f} GB")
    print(f"  A16W4  {a16w4_w:.2f} GB  ({WEIGHTS_FP16_GB/a16w4_w:.2f}× smaller)")
    print(f"  W4A4   {nu_w:.2f} GB  ({WEIGHTS_FP16_GB/nu_w:.2f}× smaller)")
    print()
    print(f"DiRotQ-effective weights (W4A4 minus the {dummy_gb*1000:.0f} MB of LoRA + smooth")
    print(f"tensors that nunchaku's kernel API requires but DiRotQ doesn't use):")
    print(f"  W4A4   {nu_w_effective:.2f} GB  ({WEIGHTS_FP16_GB/nu_w_effective:.2f}× smaller)")

    out = {
        "report": "Flux-dev: fp16 vs A16W4 baseline (no rotation) vs DiRotQ-W4A4 (no-rot ff_down)",
        "gpu": "NVIDIA GeForce RTX 4090",
        "hardware_vram_gb": 24,
        "weights_gb": {
            "fp16": WEIGHTS_FP16_GB,
            "a16w4": a16w4_w,
            "w4a4_raw": nu_w,
            "w4a4_dummy_overhead": dummy_gb,
            "w4a4_dirotq_effective": nu_w_effective,
        },
        "weight_savings_x": {
            "a16w4": WEIGHTS_FP16_GB / a16w4_w,
            "w4a4_raw": WEIGHTS_FP16_GB / nu_w,
            "w4a4_dirotq_effective": WEIGHTS_FP16_GB / nu_w_effective,
        },
        "fp16_phase": fp16,
        "a16w4_phase": a16w4,
        "w4a4_phase": nu,
        "summary": summary_rows,
        "pcie_bw_gbs": PCIE_BW_GB_S,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(FINAL_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {FINAL_JSON}")

    for p in (TMP_FP16_JSON, TMP_A16W4_JSON, TMP_NU_JSON):
        try: p.unlink()
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('a16w4',), default=None)
    args = ap.parse_args()
    if args.phase == 'a16w4':
        run_phase_a16w4()
    else:
        run_orchestrator()


if __name__ == "__main__":
    main()
