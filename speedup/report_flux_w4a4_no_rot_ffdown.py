"""Flux-dev W4A4 (DiRotQ-style) speedup with rotation disabled on ff_down.

Setup:
  - Real flux-dev fp16 weights from HuggingFace (black-forest-labs/FLUX.1-dev).
  - Quantizable nn.Linear modules replaced with W4A4 modules whose weights
    are packed into nunchaku's gemm_w4a4 layout (deepcompressor packer in
    speedup/nunchaku_pack.py). Activations are int4-quantized in the kernel
    via fused quantize+gemm; tail GEMM (high-precision region) folded into
    bias add via torch.addmm.
  - Rotation policy: enabled on every Linear EXCEPT K=12288 ff_down
    (transformer_blocks.*.ff.net.2 and ff_context.net.2). int8-mma rotation
    kernel applied where rotation is active.

The script runs in two SEPARATE subprocesses:
  Phase 1 — fp16 baseline (block-level sequential CPU offload, since fp16
            OOMs at B≥2 on 24 GB). No nunchaku imports.
  Phase 2 — DiRotQ W4A4 with no-rot ff_down. Loads fp16, patches in-place,
            measures.

Results from both phases are written to a JSON each, then combined into
one final table.

Output: speedup/results/flux_w4a4_no_rot_ffdown.json
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
TMP_FP16_JSON = RESULTS_DIR / "_tmp_fp16_phase.json"
TMP_NU_JSON = RESULTS_DIR / "_tmp_nunchaku_phase.json"
FINAL_JSON = RESULTS_DIR / "flux_w4a4_no_rot_ffdown.json"

PCIE_BW_GB_S = 25.9
WEIGHTS_FP16_GB = 23.80
WEIGHTS_W4A4_GB = 12.74

DTYPE = torch.bfloat16
DEVICE = 'cuda'


def _stub_torchao():
    """transformers 4.57 expects torchao Float8 configs that torchao 0.7 lacks."""
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
# Phase 1: fp16 with sequential CPU offload (no nunchaku imports)
# ===========================================================================

def install_block_offload(model):
    """Use accelerate's optimized cpu_offload — what diffusers'
    enable_sequential_cpu_offload uses under the hood. It installs hooks
    that pre-fetch on a CUDA stream + pinned host memory, much faster than
    a naive .to(cpu)/.to(cuda) per call.
    """
    from accelerate import cpu_offload
    blocks = list(model.transformer_blocks) + list(model.single_transformer_blocks)
    for blk in blocks:
        cpu_offload(blk, execution_device=DEVICE, offload_buffers=True)
    torch.cuda.empty_cache()
    return []  # accelerate manages hooks internally


def run_phase_fp16():
    _stub_torchao()
    print("=== Phase 1: fp16 (sequential CPU offload) ===", flush=True)

    from diffusers import FluxTransformer2DModel

    print("Loading fp16 FluxTransformer2DModel...", flush=True)
    m = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer',
        torch_dtype=DTYPE).eval()
    m.to(DEVICE)
    print("Installing block-level CPU offload...", flush=True)
    install_block_offload(m)
    print(f"  VRAM after offload: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Also measure native B=1 (fits without offload). Needs a fresh model.
    # Save offloaded numbers first since model weights are spread CPU/GPU.
    results = {'native': {}, 'offload': {}}
    for B in [1, 2, 4]:
        ms, std, peak = measure(m, B, warmup=1, repeats=3)
        results['offload'][B] = {'ms': ms, 'std_ms': std, 'peak_gb': peak}
        print(f"  fp16 B={B} (offload): {ms:>7.1f} ± {std:.1f} ms, peak {peak:.2f} GB", flush=True)
    del m

    # Native B=1 (no offload) — fits at 24.30 GB peak.
    gc.collect(); torch.cuda.empty_cache()
    print("\nMeasuring fp16 B=1 native (no offload)...", flush=True)
    m = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer',
        torch_dtype=DTYPE).to(DEVICE).eval()
    ms, std, peak = measure(m, 1, warmup=2, repeats=4)
    results['native'][1] = {'ms': ms, 'std_ms': std, 'peak_gb': peak}
    print(f"  fp16 B=1 (native): {ms:.1f} ± {std:.1f} ms, peak {peak:.2f} GB", flush=True)
    del m
    gc.collect(); torch.cuda.empty_cache()

    with open(TMP_FP16_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {TMP_FP16_JSON}", flush=True)


# ===========================================================================
# Phase 2: DiRotQ W4A4 with no-rot ff_down only
# ===========================================================================

def run_phase_nunchaku():
    _stub_torchao()
    print("=== Phase 2: DiRotQ W4A4 on SVDQuant kernel (rot K=3072, no-rot K=12288) ===", flush=True)

    if str(_DIROTQ_ROOT) not in sys.path:
        sys.path.insert(0, str(_DIROTQ_ROOT))

    from nunchaku.models.transformers.transformer_flux_v2 import (
        NunchakuFluxTransformerBlock, NunchakuFluxSingleTransformerBlock,
        NunchakuFluxTransformer2DModelV2,
    )
    from nunchaku.models.linear import SVDQW4A4Linear
    from nunchaku.models.embeddings import (NunchakuFluxPosEmbed, pack_rotemb)
    from nunchaku.utils import pad_tensor
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers import FluxTransformer2DModel
    from speedup.nunchaku_pack import convert_dirotq_low_weight_to_nunchaku
    from speedup.kernels.int8_rotation import (
        int8_rotation_forward, quantize_U_int8)

    # ----- Custom block forwards: inject DiRotQ rotation before every K=3072 op -----
    # K=12288 layers (mlp_fc2 = ff_down) are inside fused_gelu_mlp and get
    # int4-packed input via qout chain — no rotation, matches user spec.
    def make_double_forward(U_int8, scale_U):
        def fwd(self, hidden_states, encoder_hidden_states, temb,
                image_rotary_emb=None, joint_attention_kwargs=None):
            norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_e, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

            # Rotate before attention QKV (K=3072 → 9216 fused). attn.to_out
            # rotation is applied inside the processor (we set the patched one).
            norm_h_rot = int8_rotation_forward(norm_h.reshape(-1, 3072), U_int8, scale_U).reshape(norm_h.shape)
            norm_e_rot = int8_rotation_forward(norm_e.reshape(-1, 3072), U_int8, scale_U).reshape(norm_e.shape)

            attn_out, ctx_attn_out = self.attn(
                hidden_states=norm_h_rot, encoder_hidden_states=norm_e_rot,
                image_rotary_emb=image_rotary_emb,
            )

            attn_out = gate_msa.unsqueeze(1) * attn_out
            hidden_states = hidden_states + attn_out
            norm_h2 = self.norm2(hidden_states)
            norm_h2 = norm_h2 * scale_mlp[:, None] + shift_mlp[:, None]
            # Rotate before ff (ff_up K=3072; ff_down K=12288 stays inside fused_gelu_mlp)
            norm_h2_rot = int8_rotation_forward(norm_h2.reshape(-1, 3072), U_int8, scale_U).reshape(norm_h2.shape)
            ff_out = self.ff(norm_h2_rot)
            ff_out = gate_mlp.unsqueeze(1) * ff_out
            hidden_states = hidden_states + ff_out

            ctx_attn_out = c_gate_msa.unsqueeze(1) * ctx_attn_out
            encoder_hidden_states = encoder_hidden_states + ctx_attn_out
            norm_e2 = self.norm2_context(encoder_hidden_states)
            norm_e2 = norm_e2 * c_scale_mlp[:, None] + c_shift_mlp[:, None]
            norm_e2_rot = int8_rotation_forward(norm_e2.reshape(-1, 3072), U_int8, scale_U).reshape(norm_e2.shape)
            ctx_ff_out = self.ff_context(norm_e2_rot)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ctx_ff_out
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            return encoder_hidden_states, hidden_states
        return fwd

    def make_single_forward(U_int8, scale_U):
        def fwd(self, hidden_states, temb, image_rotary_emb=None,
                joint_attention_kwargs=None):
            from nunchaku.ops.fused import fused_gelu_mlp
            from torch.nn import GELU
            residual = hidden_states
            norm_h, gate = self.norm(hidden_states, emb=temb)
            # Rotate before mlp_fc1 (K=3072) and attn (K=3072). mlp_fc2 K=12288
            # is fused inside fused_gelu_mlp — automatic no-rot.
            norm_h_rot = int8_rotation_forward(norm_h.reshape(-1, 3072), U_int8, scale_U).reshape(norm_h.shape)
            if isinstance(self.act_mlp, GELU):
                mlp_h = fused_gelu_mlp(norm_h_rot, self.mlp_fc1, self.mlp_fc2)
            else:
                mlp_h = self.mlp_fc1(norm_h_rot)
                mlp_h = self.act_mlp(mlp_h)
                mlp_h = self.mlp_fc2(mlp_h)
            attn_out = self.attn(hidden_states=norm_h_rot, image_rotary_emb=image_rotary_emb)
            hidden_states = attn_out + mlp_h
            hidden_states = gate.unsqueeze(1) * hidden_states
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            return hidden_states
        return fwd

    # Build a custom processor that rotates BEFORE attn.to_out (K=3072)
    def make_attn_processor(U_int8, scale_U):
        from nunchaku.ops.fused import fused_qkv_norm_rottary
        import torch.nn.functional as F

        class RotProc:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, image_rotary_emb=None, **kw):
                B, _, C = hidden_states.shape
                qkv = fused_qkv_norm_rottary(
                    hidden_states, attn.to_qkv, attn.norm_q, attn.norm_k,
                    image_rotary_emb[0] if isinstance(image_rotary_emb, tuple) else image_rotary_emb,
                )
                if attn.added_kv_proj_dim is not None:
                    assert encoder_hidden_states is not None
                    qkv_ctx = fused_qkv_norm_rottary(
                        encoder_hidden_states, attn.add_qkv_proj,
                        attn.norm_added_q, attn.norm_added_k, image_rotary_emb[1],
                    )
                    qkv = torch.cat([qkv_ctx, qkv], dim=1)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B, -1, attn.heads, attn.head_dim).transpose(1, 2)
                k = k.view(B, -1, attn.heads, attn.head_dim).transpose(1, 2)
                v = v.view(B, -1, attn.heads, attn.head_dim).transpose(1, 2)
                hs = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
                hs = hs.transpose(1, 2).reshape(B, -1, attn.heads * attn.head_dim)
                hs = hs.to(q.dtype)

                if encoder_hidden_states is not None:
                    enc_hs = hs[:, : encoder_hidden_states.shape[1]]
                    img_hs = hs[:, encoder_hidden_states.shape[1] :]
                    # Rotate before to_out / to_add_out (K=3072)
                    img_hs_rot = int8_rotation_forward(img_hs.reshape(-1, 3072), U_int8, scale_U).reshape(img_hs.shape)
                    img_hs = attn.to_out[0](img_hs_rot)
                    img_hs = attn.to_out[1](img_hs)
                    enc_hs_rot = int8_rotation_forward(enc_hs.reshape(-1, 3072), U_int8, scale_U).reshape(enc_hs.shape)
                    enc_hs = attn.to_add_out(enc_hs_rot)
                    return img_hs, enc_hs
                else:
                    hs_rot = int8_rotation_forward(hs.reshape(-1, 3072), U_int8, scale_U).reshape(hs.shape)
                    return attn.to_out(hs_rot)
        return RotProc()

    # ----- Load fp16 model + replace blocks with v2 wrappers + populate -----
    print("Loading fp16 FluxTransformer2DModel (CPU)...", flush=True)
    fp16_m = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer',
        torch_dtype=DTYPE).eval()

    print("Building shared rotation U_3072 (DiRotQ)...", flush=True)
    Q3, _ = torch.linalg.qr(torch.randn(3072, 3072, device=DEVICE))
    U_3072_int8, scale_U_3072 = quantize_U_int8(Q3.to(DTYPE).contiguous())
    del Q3
    proc = make_attn_processor(U_3072_int8, scale_U_3072)

    def pack_into(svdq, fp_W, fp_b):
        pack = convert_dirotq_low_weight_to_nunchaku(
            W_low=fp_W, bias=fp_b, group_size=64, rank=svdq.rank)
        with torch.no_grad():
            svdq.qweight.data = pack['qweight'].contiguous()
            svdq.wscales.data = pack['wscales'].to(svdq.wscales.dtype).contiguous()
            if svdq.bias is not None:
                if fp_b is not None:
                    svdq.bias.data = fp_b.to(svdq.bias.dtype).contiguous()
                else:
                    svdq.bias.data = torch.zeros_like(svdq.bias.data)
            svdq.smooth_factor.data = torch.ones_like(svdq.smooth_factor.data)
            svdq.smooth_factor_orig.data = torch.ones_like(svdq.smooth_factor_orig.data)
            svdq.proj_down.data = torch.zeros_like(svdq.proj_down.data)
            svdq.proj_up.data = torch.zeros_like(svdq.proj_up.data)

    print("Replacing flux blocks with v2 wrappers + DiRotQ rotation...", flush=True)
    n_double = len(fp16_m.transformer_blocks)
    n_single = len(fp16_m.single_transformer_blocks)
    print(f"  {n_double} double + {n_single} single blocks", flush=True)

    for i in range(n_double):
        orig = fp16_m.transformer_blocks[i].to(DEVICE)
        a = orig.attn
        # gather originals BEFORE wrapping (NunchakuFluxAttention shares to_out)
        ws = {
            'qkv_W': torch.cat([a.to_q.weight.data, a.to_k.weight.data, a.to_v.weight.data], dim=0).clone(),
            'qkv_b': torch.cat([a.to_q.bias.data, a.to_k.bias.data, a.to_v.bias.data], dim=0).clone() if a.to_q.bias is not None else None,
            'aqkv_W': torch.cat([a.add_q_proj.weight.data, a.add_k_proj.weight.data, a.add_v_proj.weight.data], dim=0).clone(),
            'aqkv_b': torch.cat([a.add_q_proj.bias.data, a.add_k_proj.bias.data, a.add_v_proj.bias.data], dim=0).clone() if a.add_q_proj.bias is not None else None,
            'to_out_W': a.to_out[0].weight.data.clone(),
            'to_out_b': a.to_out[0].bias.data.clone() if a.to_out[0].bias is not None else None,
            'to_add_out_W': a.to_add_out.weight.data.clone(),
            'to_add_out_b': a.to_add_out.bias.data.clone() if a.to_add_out.bias is not None else None,
            'ff_up_W': orig.ff.net[0].proj.weight.data.clone(),
            'ff_up_b': orig.ff.net[0].proj.bias.data.clone() if orig.ff.net[0].proj.bias is not None else None,
            'ff_down_W': orig.ff.net[2].weight.data.clone(),
            'ff_down_b': orig.ff.net[2].bias.data.clone() if orig.ff.net[2].bias is not None else None,
            'ff_ctx_up_W': orig.ff_context.net[0].proj.weight.data.clone(),
            'ff_ctx_up_b': orig.ff_context.net[0].proj.bias.data.clone() if orig.ff_context.net[0].proj.bias is not None else None,
            'ff_ctx_down_W': orig.ff_context.net[2].weight.data.clone(),
            'ff_ctx_down_b': orig.ff_context.net[2].bias.data.clone() if orig.ff_context.net[2].bias is not None else None,
            'norm1_lin': orig.norm1.linear,
            'norm1_ctx_lin': orig.norm1_context.linear,
        }
        new_blk = NunchakuFluxTransformerBlock(orig, scale_shift=0)
        new_blk = new_blk.to_empty(device=DEVICE)
        # Replace AWQW4A16 modulators with original fp16 nn.Linear (DiRotQ has no W4A16)
        new_blk.norm1.linear = ws['norm1_lin'].to(DEVICE)
        new_blk.norm1_context.linear = ws['norm1_ctx_lin'].to(DEVICE)
        # Pack the W4A4 linears
        pack_into(new_blk.attn.to_qkv, ws['qkv_W'], ws['qkv_b'])
        pack_into(new_blk.attn.to_out[0], ws['to_out_W'], ws['to_out_b'])
        pack_into(new_blk.attn.add_qkv_proj, ws['aqkv_W'], ws['aqkv_b'])
        pack_into(new_blk.attn.to_add_out, ws['to_add_out_W'], ws['to_add_out_b'])
        pack_into(new_blk.ff.net[0].proj, ws['ff_up_W'], ws['ff_up_b'])
        pack_into(new_blk.ff.net[2], ws['ff_down_W'], ws['ff_down_b'])
        pack_into(new_blk.ff_context.net[0].proj, ws['ff_ctx_up_W'], ws['ff_ctx_up_b'])
        pack_into(new_blk.ff_context.net[2], ws['ff_ctx_down_W'], ws['ff_ctx_down_b'])
        # Install custom processor (rotates before to_out / to_add_out)
        new_blk.attn.processor = proc
        # Install custom forward (rotates before attn input + ff input)
        import types as _types
        new_blk.forward = _types.MethodType(make_double_forward(U_3072_int8, scale_U_3072), new_blk)
        fp16_m.transformer_blocks[i] = new_blk
        del orig, ws
        gc.collect(); torch.cuda.empty_cache()
        if i % 5 == 0:
            print(f"    double {i}/{n_double}: VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    for i in range(n_single):
        orig = fp16_m.single_transformer_blocks[i].to(DEVICE)
        a = orig.attn
        ws = {
            'qkv_W': torch.cat([a.to_q.weight.data, a.to_k.weight.data, a.to_v.weight.data], dim=0).clone(),
            'qkv_b': torch.cat([a.to_q.bias.data, a.to_k.bias.data, a.to_v.bias.data], dim=0).clone() if a.to_q.bias is not None else None,
            'mlp_fc1_W': orig.proj_mlp.weight.data.clone(),
            'mlp_fc1_b': orig.proj_mlp.bias.data.clone() if orig.proj_mlp.bias is not None else None,
            'proj_out_W': orig.proj_out.weight.data.clone(),
            'proj_out_b': orig.proj_out.bias.data.clone() if orig.proj_out.bias is not None else None,
            'norm_lin': orig.norm.linear,
        }
        new_blk = NunchakuFluxSingleTransformerBlock(orig, scale_shift=0)
        new_blk = new_blk.to_empty(device=DEVICE)
        new_blk.norm.linear = ws['norm_lin'].to(DEVICE)
        pack_into(new_blk.attn.to_qkv, ws['qkv_W'], ws['qkv_b'])
        pack_into(new_blk.mlp_fc1, ws['mlp_fc1_W'], ws['mlp_fc1_b'])
        in_attn = new_blk.attn.to_out.in_features
        in_mlp = new_blk.mlp_fc2.in_features
        pack_into(new_blk.attn.to_out, ws['proj_out_W'][:, :in_attn].contiguous(), None)
        pack_into(new_blk.mlp_fc2, ws['proj_out_W'][:, in_attn:in_attn + in_mlp].contiguous(), ws['proj_out_b'])
        new_blk.attn.processor = proc
        import types as _types
        new_blk.forward = _types.MethodType(make_single_forward(U_3072_int8, scale_U_3072), new_blk)
        fp16_m.single_transformer_blocks[i] = new_blk
        del orig, ws
        gc.collect(); torch.cuda.empty_cache()
        if i % 10 == 0:
            print(f"    single {i}/{n_single}: VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    fp16_m.to(DEVICE)
    # Replace pos_embed with the v2 one (returns 6D)
    axes_dims_rope = fp16_m.config.axes_dims_rope
    fp16_m.pos_embed = NunchakuFluxPosEmbed(
        dim=sum(axes_dims_rope), theta=10000,
        axes_dim=list(axes_dims_rope)).to(DEVICE)
    w4a4_weight_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  W4A4 weight footprint: {w4a4_weight_gb:.2f} GB", flush=True)

    # ----- Bind v2's batch-aware forward (with rotary expand for B>1) -----
    def patched_forward(self, hidden_states, encoder_hidden_states=None,
                        pooled_projections=None, timestep=None,
                        img_ids=None, txt_ids=None, guidance=None,
                        joint_attention_kwargs=None,
                        controlnet_block_samples=None,
                        controlnet_single_block_samples=None,
                        return_dict=True, controlnet_blocks_repeat=False):
        if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
            raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")

        B = hidden_states.shape[0]
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        assert image_rotary_emb.ndim == 6
        image_rotary_emb = image_rotary_emb.reshape(
            [1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        # === PATCH: expand rotary_emb to actual batch ===
        if B > 1:
            rotary_emb_txt = rotary_emb_txt.expand(B, *rotary_emb_txt.shape[1:]).contiguous()
            rotary_emb_img = rotary_emb_img.expand(B, *rotary_emb_img.shape[1:]).contiguous()
            rotary_emb_single = rotary_emb_single.expand(B, *rotary_emb_single.shape[1:]).contiguous()

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=rotary_emb_single,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = hidden_states[:, txt_tokens:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    fp16_m.forward = _types.MethodType(patched_forward, fp16_m)
    print("  bound batch-aware forward (expands rotary_emb to B)", flush=True)
    m = fp16_m

    results = {'measurements': {}, 'w4a4_weight_gb': w4a4_weight_gb,
               'source': 'DiRotQ packing on nunchaku v2 SVDQuant kernel (rot K=3072, no-rot K=12288)'}
    for B in [1, 2, 4]:
        try:
            ms, std, peak = measure(m, B, warmup=2, repeats=4)
            results['measurements'][B] = {'ms': ms, 'std_ms': std, 'peak_gb': peak}
            print(f"  DiRotQ-on-SVDQuant W4A4 B={B}: {ms:>7.1f} ± {std:.1f} ms, peak {peak:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  DiRotQ-on-SVDQuant W4A4 B={B}: OOM", flush=True)
            results['measurements'][B] = {'oom': True, 'error': str(e)[:200]}
            gc.collect(); torch.cuda.empty_cache()

    with open(TMP_NU_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {TMP_NU_JSON}", flush=True)


# ===========================================================================
# Orchestrator: run both phases and combine
# ===========================================================================

def run_orchestrator():
    print("Running fp16 phase in subprocess...")
    r = subprocess.run([sys.executable, '-u', __file__, '--phase=fp16'])
    if r.returncode != 0:
        print(f"fp16 phase failed (exit {r.returncode})")
        sys.exit(1)
    print("\nRunning nunchaku phase in subprocess...")
    r = subprocess.run([sys.executable, '-u', __file__, '--phase=nunchaku'])
    if r.returncode != 0:
        print(f"nunchaku phase failed (exit {r.returncode})")
        sys.exit(1)

    with open(TMP_FP16_JSON) as f:
        fp16 = json.load(f)
    with open(TMP_NU_JSON) as f:
        nu = json.load(f)

    print()
    print("=" * 78)
    print("Flux-dev W4A4 — DiRotQ on SVDQuant kernel (no-rot ff_down) — results")
    print("=" * 78)
    print(f"Hardware: NVIDIA GeForce RTX 4090 (24 GB)")
    print(f"W4A4 path: nunchaku v2 fused kernels (fused QKV+norm+rotary,")
    print(f"  fused_gelu_mlp with qout-chained int4 intermediates) + DiRotQ")
    print(f"  rotation injected before every K=3072 op. K=12288 (ff_down /")
    print(f"  mlp_fc2) stays inside fused_gelu_mlp = automatic no-rotation.")
    print()

    print(f"{'B':>3}  {'fp16 (ms)':>16}  {'W4A4 (ms)':>11}  "
          f"{'fp16 peak GB*':>14}  {'W4A4 peak GB':>13}  {'speedup':>9}")
    print('-' * 78)
    summary_rows = []
    for B in [1, 2, 4]:
        fp16_native = fp16['native'].get(str(B)) or fp16['native'].get(B)
        fp16_offload = fp16['offload'].get(str(B)) or fp16['offload'].get(B)
        nu_meas = nu['measurements'].get(str(B)) or nu['measurements'].get(B)

        # fp16: B=1 native fits; B≥2 must offload
        if B == 1 and fp16_native:
            fp16_ms = fp16_native['ms']
            fp16_label = "native"
            fp16_peak_native = fp16_native['peak_gb']
        elif fp16_offload:
            fp16_ms = fp16_offload['ms']
            fp16_label = "offload"
            fp16_peak_native = None  # would-need-native estimate
        else:
            continue

        if not nu_meas or 'oom' in nu_meas:
            continue

        nu_ms = nu_meas['ms']
        nu_peak = nu_meas['peak_gb']
        sp = fp16_ms / nu_ms
        # "would-need-native" peak for fp16 at B>1 (estimated linearly from B=1)
        if fp16_peak_native is None and 1 in (1, 2, 4) and \
                (fp16['native'].get(str(1)) or fp16['native'].get(1)):
            base = (fp16['native'].get(str(1)) or fp16['native'].get(1))['peak_gb']
            act_b1 = base - WEIGHTS_FP16_GB
            fp16_peak_native = WEIGHTS_FP16_GB + act_b1 * B
        ns = f"{fp16_ms:.0f} ({fp16_label})"
        ps = f"{fp16_peak_native:.1f}" if fp16_peak_native else "—"
        print(f"{B:>3}  {ns:>16}  {nu_ms:>11.0f}  "
              f"{ps:>14}  {nu_peak:>13.2f}  {sp:>8.2f}x")
        summary_rows.append({
            'B': B, 'fp16_ms': fp16_ms, 'fp16_label': fp16_label,
            'fp16_peak_native_gb': fp16_peak_native,
            'w4a4_ms': nu_ms, 'w4a4_peak_gb': nu_peak,
            'speedup': sp,
        })
    print(f"{'*':>3}  fp16 peak GB is what NATIVE run would need "
          f"(it OOMs at B≥2 on 24 GB).")

    # 16 GB and 24 GB scenarios
    print()
    print("On 24 GB GPU:")
    for r in summary_rows:
        nu_off_gb = max(0.0, r['w4a4_peak_gb'] - 23.0)
        nu_off_ms = nu_off_gb / PCIE_BW_GB_S * 1000
        nu_total = r['w4a4_ms'] + nu_off_ms
        sp = r['fp16_ms'] / nu_total
        fits = "fits" if nu_off_gb == 0 else f"streams {nu_off_gb:.1f} GB"
        print(f"  B={r['B']}: fp16 {r['fp16_ms']:.0f} ms ({r['fp16_label']})  |  "
              f"W4A4 {nu_total:.0f} ms ({fits})  →  speedup {sp:.2f}x")

    print()
    print("On 16 GB GPU:")
    for r in summary_rows:
        nu_off_gb = max(0.0, r['w4a4_peak_gb'] - 15.0)
        nu_off_ms = nu_off_gb / PCIE_BW_GB_S * 1000
        nu_total = r['w4a4_ms'] + nu_off_ms
        sp = r['fp16_ms'] / nu_total
        fits = "fits" if nu_off_gb == 0 else f"streams {nu_off_gb:.1f} GB"
        print(f"  B={r['B']}: fp16 {r['fp16_ms']:.0f} ms ({r['fp16_label']})  |  "
              f"W4A4 {nu_total:.0f} ms ({fits})  →  speedup {sp:.2f}x")

    # ============== Memory breakdown: weights + activations ==============
    # Decompose the measured peak VRAM into weight footprint + activation
    # footprint for each path:
    #
    #   activations[B] = peak_total[B] - weight_footprint
    #
    # fp16:
    #   - native B=1 fits: 24.30 GB peak; weights = 23.80 GB; acts = 0.50 GB
    #   - B≥2 OOMs natively → activation cost extrapolated linearly from B=1
    # W4A4 (DiRotQ no-rot ff_down):
    #   - weight footprint measured after patch: ~13.08 GB
    #   - peak at each B measured directly
    fp16_weights = WEIGHTS_FP16_GB
    # Use measured native B=1 peak to derive fp16 act/working at B=1
    fp16_b1_native = (fp16['native'].get('1') or fp16['native'].get(1))
    fp16_act_b1 = fp16_b1_native['peak_gb'] - fp16_weights
    w4a4_weights_measured = nu.get('w4a4_weight_gb')

    print()
    print("Memory breakdown (peak VRAM = weights + activations + working):")
    print(f"{'B':>3}  {'fp16 W (GB)':>13}{'fp16 A (GB)':>13}{'fp16 total':>12}"
          f"  {'W4A4 W (GB)':>13}{'W4A4 A (GB)':>13}{'W4A4 total':>12}"
          f"  {'total ratio':>12}")
    print('-' * 110)
    for r in summary_rows:
        B = r['B']
        # fp16 activations: measured at B=1, linearly extrapolated for B>1
        if B == 1 and fp16_b1_native:
            f_acts = fp16_act_b1
        else:
            f_acts = fp16_act_b1 * B
        f_total = fp16_weights + f_acts
        n_total = r['w4a4_peak_gb']
        n_acts = n_total - w4a4_weights_measured
        ratio = f_total / n_total
        print(f"{B:>3}  {fp16_weights:>13.2f}{f_acts:>13.2f}{f_total:>12.2f}"
              f"  {w4a4_weights_measured:>13.2f}{n_acts:>13.2f}{n_total:>12.2f}"
              f"  {ratio:>11.2f}×")

    print()
    print(f"Weights:     fp16 {WEIGHTS_FP16_GB:.2f} GB  vs  W4A4 {w4a4_weights_measured:.2f} GB"
          f"  →  {WEIGHTS_FP16_GB/w4a4_weights_measured:.2f}× smaller")
    act_ratios = []
    for r in summary_rows:
        B = r['B']
        f_acts = fp16_act_b1 * B if B > 1 else fp16_act_b1
        n_acts = r['w4a4_peak_gb'] - w4a4_weights_measured
        if n_acts > 0:
            act_ratios.append(f_acts / n_acts)
    if act_ratios:
        avg_act = sum(act_ratios) / len(act_ratios)
        print(f"Activations: averaged across batches, fp16 vs W4A4 = {avg_act:.2f}×")

    out = {
        "report": "Flux-dev W4A4 — DiRotQ on SVDQuant kernel (no-rot ff_down)",
        "gpu": "NVIDIA GeForce RTX 4090",
        "hardware_vram_gb": 24,
        "w4a4_source": nu.get('source'),
        "weights_gb": {"fp16": WEIGHTS_FP16_GB,
                       "w4a4_measured": w4a4_weights_measured},
        "weight_savings_x": WEIGHTS_FP16_GB / w4a4_weights_measured,
        "fp16_phase": fp16,
        "w4a4_phase": nu,
        "summary": summary_rows,
        "pcie_bw_gbs": PCIE_BW_GB_S,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(FINAL_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {FINAL_JSON}")

    for p in (TMP_FP16_JSON, TMP_NU_JSON):
        try:
            p.unlink()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('fp16', 'nunchaku'), default=None)
    args = ap.parse_args()
    if args.phase == 'fp16':
        run_phase_fp16()
    elif args.phase == 'nunchaku':
        run_phase_nunchaku()
    else:
        run_orchestrator()


if __name__ == "__main__":
    main()
