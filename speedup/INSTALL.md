# Reproducing the speedup numbers

Cold-start checklist for `report_flux_w4a4_no_rot_ffdown.py` (2 configs:
fp16 + DiRotQ-W4A4) and `report_flux_3_configs.py` (3 configs: fp16 +
A16W4 baseline + DiRotQ-W4A4) on a fresh machine.

## 1. Hardware

- **GPU**: NVIDIA RTX 4090 (24 GB) or any **sm_89 / Ada-class** card
  (RTX 4080/3090Ti also work; 16 GB cards work for the W4A4 phase but
  cannot run fp16 native B=1 — that path needs ≥ 24 GB). The nunchaku
  W4A4 kernel uses `mma.m16n8k64.s4.s4.s32` which exists on **sm_75–sm_89
  (Turing through Ada)**. Hopper (sm_90) and Blackwell (sm_100) removed
  int4 tensor cores; nunchaku falls back to int8 there and you'll see
  different absolute numbers (relative speedups still hold).
- **PCIe Gen 4 ×16** — pinned host→device bandwidth is the dominant cost
  for the fp16 sequential-offload path at B≥2. Measured ≈ 26 GB/s on
  this rig; PCIe Gen 3 will roughly halve fp16's offload throughput.

## 2. System software

| Component        | Tested version                  | Notes |
|------------------|---------------------------------|-------|
| OS               | Ubuntu 22.04 (glibc 2.35)        | bitsandbytes wheels need glibc ≥ 2.31. |
| NVIDIA driver    | 575.57.08                        | Driver ≥ 525.85 required for CUDA 12.4 runtime. |
| CUDA toolkit     | not needed system-wide           | The torch wheel ships its own CUDA 12.4 runtime libs; only need a system-level toolkit if you compile something from source. |
| Python           | 3.12.13                          | 3.10 / 3.11 also work; pin in conda env. |
| Disk             | ≥ 50 GB free in `~/.cache/`     | FLUX.1-dev fp16 ≈ 32 GB; SVDQuant int4 ckpt ≈ 6.4 GB; plus pip wheels. |
| Network          | required at first run            | HF Hub downloads. Subsequent runs are cache hits. |

## 3. Python environment

```bash
conda create -n mldiffusion2 python=3.12 -y
conda activate mldiffusion2

# PyTorch 2.6 + CUDA 12.4 (the version nunchaku 1.0.1 was built against)
pip install torch==2.6.0+cu124 \
            --index-url https://download.pytorch.org/whl/cu124

# Diffusers / transformers stack — versions are load-bearing
pip install diffusers==0.36.0 \
            transformers==4.57.6 \
            accelerate==1.12.0 \
            safetensors==0.7.0 \
            huggingface_hub \
            numpy

# Compiler / kernel tooling
pip install triton==3.2.0
pip install torchao==0.7.0      # transformers 4.57 imports it eagerly

# nunchaku — must match torch 2.6
pip install nunchaku==1.0.1+torch2.6
# (or fetch the matching wheel from
#  https://github.com/nunchaku-tech/nunchaku/releases)

# A16W4 baseline kernel (only used by report_flux_3_configs.py)
pip install bitsandbytes==0.49.2
```

Exact versions captured at the time these numbers were generated:

```
torch                 2.6.0+cu124
torchao               0.7.0
transformers          4.57.6
diffusers             0.36.0
accelerate            1.12.0
safetensors           0.7.0
triton                3.2.0
nunchaku              1.0.1+torch2.6
bitsandbytes          0.49.2
numpy                 2.4.4
```

## 4. HuggingFace assets (downloaded at runtime)

The scripts pull two checkpoints to `~/.cache/huggingface/hub/`:

| Repo                                  | File                                              | Size  | Why |
|---------------------------------------|---------------------------------------------------|-------|-----|
| `black-forest-labs/FLUX.1-dev`        | full transformer (3 safetensors shards)           | ~32 GB | bf16 weights for the fp16 baseline AND the source from which DiRotQ-W4A4 packs its int4 weights. |
| `nunchaku-tech/nunchaku-flux.1-dev`   | `svdq-int4_r32-flux.1-dev.safetensors`            | 6.4 GB | Provides nunchaku's pre-quantized v2 layout used as the structural template (the W4A4 phase re-packs from fp16 weights, but uses this checkpoint to instantiate the v2 module shape). |

Both are gated. Steps:

1. Make a HuggingFace account.
2. Visit https://huggingface.co/black-forest-labs/FLUX.1-dev and click
   "Agree and access" (license acceptance is per-user, can't be done
   programmatically).
3. Create an access token at https://huggingface.co/settings/tokens
   with read access.
4. Export it before running:

   ```bash
   export HUGGING_FACE_HUB_TOKEN=hf_…your_token…
   ```

   The scripts read this from the environment — never hard-code the
   token in source.

## 5. Quick smoke test

After installation, confirm every import the scripts need resolves:

```bash
python -c "
import torchao.quantization as _t
class _S: pass
for n in ('Float8WeightOnlyConfig', 'Float8DynamicActivationFloat8WeightConfig'):
    if not hasattr(_t, n): setattr(_t, n, _S)
import sys; sys.path.insert(0, '/path/to/DiRotQ')
from speedup.nunchaku_pack import convert_dirotq_low_weight_to_nunchaku
from speedup.kernels.int8_rotation import int8_rotation_forward, quantize_U_int8
from nunchaku.models.transformers.transformer_flux_v2 import (
    NunchakuFluxTransformer2DModelV2, NunchakuFluxTransformerBlock,
    NunchakuFluxSingleTransformerBlock)
from nunchaku.ops.fused import fused_qkv_norm_rottary, fused_gelu_mlp
from accelerate import cpu_offload
import bitsandbytes as bnb
from diffusers import FluxTransformer2DModel
print('OK')
"
```

If you get `AttributeError: module 'transformers' has no attribute
'CLIPTextModel'`, the torchao stub didn't run early enough — see
landmine #1 below.

## 6. Run

```bash
# 2-config (fp16 + DiRotQ-W4A4)
python -u speedup/report_flux_w4a4_no_rot_ffdown.py

# 3-config (fp16 + A16W4 baseline + DiRotQ-W4A4)
python -u speedup/report_flux_3_configs.py
```

Each writes a JSON to `speedup/results/`. First run takes ~10–15 min
(model download dominates); subsequent runs are ~3–5 min.

Expected numbers (RTX 4090, B=1/2/4, M=4608):

| Config           | B=1 (ms)         | B=2 (ms, fp16 offload) | B=4 (ms, fp16 offload) |
|------------------|------------------|------------------------|------------------------|
| fp16             | ~620 native      | ~5 000 offload         | ~7 200 offload         |
| A16W4 (bnb NF4)  | ~640 native      | ~1 330 native          | ~2 630 native          |
| DiRotQ-W4A4      | ~273 native      | ~545 native            | ~1 085 native          |

Variance is < 1 % between runs (within-process timing noise).

## 7. Landmines / non-obvious dependencies

1. **transformers 4.57 vs torchao 0.7** — transformers expects
   `torchao.quantization.Float8WeightOnlyConfig` which torchao 0.7 doesn't
   define. Without a stub, every `from diffusers import …` lazy-loads
   `transformers.models.clip.modeling_clip` which silently returns `None`
   for `CLIPTextModel`, `PreTrainedModel`, etc. The two scripts in this
   folder stub it at the top of the file:

   ```python
   import torchao.quantization as _t
   class _Stub: pass
   for n in ('Float8WeightOnlyConfig',
             'Float8DynamicActivationFloat8WeightConfig'):
       if not hasattr(_t, n):
           setattr(_t, n, _Stub)
   ```

   This must run **before** any `transformers` / `diffusers` import.
   Upgrading torchao would also fix it but risks breaking other deps.

2. **nunchaku v2 forward is hard-coded for B=1.** It pads `rotary_emb`
   to shape `(1, M, D)` and the int4 GEMM kernel asserts
   `rotary_emb.shape[0] * rotary_emb.shape[1] == M_padded_activation`.
   At B>1 the activation has shape `(B*M, K)` so the assertion fails.
   The W4A4 phase patches the forward to expand `rotary_emb_txt`,
   `rotary_emb_img`, and `rotary_emb_single` to the actual batch size
   before passing them to the blocks (see `patched_forward` in
   `report_flux_w4a4_no_rot_ffdown.py`).

3. **NunchakuFluxAttention shares the `to_out` ModuleList with the
   original FluxAttention.** When you wrap a block with
   `NunchakuFluxTransformerBlock(orig)`, the wrapper mutates
   `orig.attn.to_out[0]` in place (replaces it with `SVDQW4A4Linear`).
   So you must `clone()` all original fp16 weights into a dict BEFORE
   constructing the wrapper, then populate the wrapped block from that
   dict — see `gather_double_weights` / `gather_single_weights` in the
   W4A4 phase.

4. **AWQW4A16Linear (modulator)** materialised on `meta` device when the
   v2 wrapper is built. The script swaps it back to the original fp16
   `nn.Linear` (DiRotQ doesn't quantize modulators):

   ```python
   new_blk.norm1.linear = orig_blk.norm1.linear.to(DEVICE)
   ```

5. **bitsandbytes Linear4bit weight assignment.** Don't do
   `lin.weight.data = W.clone()` — that overwrites bnb's internal
   `Params4bit` object with a raw fp16 tensor and the gemm kernel breaks
   silently for M ≥ 16 (M = 1 still works because it routes through
   gemv). Use the proper API:

   ```python
   lin = bnb.nn.Linear4bit(K, N, bias=…, quant_type='nf4',
                            compute_dtype=torch.bfloat16)
   lin.weight = bnb.nn.Params4bit(
       fp_W.detach().clone().to('cpu'),
       quant_type='nf4', requires_grad=False)
   lin.to('cuda')   # this triggers the actual quantization
   ```

6. **`torch._weight_int4pack_mm` is the wrong kernel for diffusion.**
   It's optimized for LLM batch-1 token decoding (M=1) and runs ~9× slower
   than cuBLAS bf16 at flux's M=4096. We tried it as the A16W4 baseline
   in an earlier iteration and got 0.18× speedup at B=1 — wildly off
   from SVDQuant's "A16W4 ≈ fp16" claim. bitsandbytes NF4 (or Marlin /
   AWQ-CUTLASS, neither of which is in this env) is the right choice.

7. **`accelerate.cpu_offload` for the fp16 phase.** This is what
   diffusers' `enable_sequential_cpu_offload` uses under the hood
   (pinned host memory + CUDA stream pre-fetch). A naive
   `.to(cpu)`/`.to(cuda)` per layer is roughly 2× slower than
   accelerate's hook, which would inflate fp16's offload time and make
   the headline speedups look unrealistically good.

8. **Subprocess-isolated phases.** The orchestrator runs each phase
   in a fresh Python subprocess. Reasons:
   - fp16 native B=1 peaks at 24.30 GB; doing nunchaku packing in the
     same process leaves cached driver allocations that fragment memory
     and cause spurious OOMs.
   - nunchaku's C++ kernel state is a process global; running fp16 first
     in the same process can leave kernel handles in a state that makes
     the next nunchaku call fail with cryptic CUDA assertions.

9. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is set inside
   the scripts but it's worth knowing about — without it, allocator
   fragmentation can cause OOMs even when total free VRAM is plenty.

## 8. Files in this folder (none are optional)

```
speedup/
├── README.md
├── INSTALL.md                                  ← this file
├── __init__.py                                 ← package marker (empty)
├── nunchaku_pack.py                            ← deepcompressor packer port
├── report_flux_w4a4_no_rot_ffdown.py           ← 2-config orchestrator
├── report_flux_3_configs.py                    ← 3-config orchestrator
├── kernels/
│   ├── __init__.py                             ← package marker (empty)
│   └── int8_rotation.py                        ← Triton int8 rotation kernel
└── results/
    └── (auto-generated JSON output)
```

`nunchaku_pack.py` and `kernels/int8_rotation.py` are the only internal
modules the orchestrators import. Both are self-contained (`torch` /
`triton` only — no further internal deps).
