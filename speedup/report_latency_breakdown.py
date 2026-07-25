"""Kernel-level latency breakdown for DiRotQ-W4A4 on the SVDQuant nunchaku kernel.

Builds the exact same model as `report_flux_w4a4_no_rot_ffdown.py` (DiRotQ
rotation on every K=3072 op, no rotation on K=12288 ff_down, int4 GEMM via
nunchaku v2 fused kernels) and profiles one forward pass per batch size with
`torch.profiler`, bucketing every CUDA kernel launch into one of:

  - rotation          Triton int8 `x @ U` kernel (kernels/int8_rotation.py),
                       injected before every K=3072 op. Includes the int8
                       quantize of the rotation's own output (fused in the
                       same Triton kernel).
  - quant_dequant     nunchaku's `quantize_w4a4_fuse_lora_kernel` — the
                       activation int4-quantize pass launched immediately
                       before each gemm_w4a4 kernel (a separate kernel, not
                       fused into the GEMM itself).
  - int4_gemm         nunchaku's fused W4A4 CUDA kernels: the qkv/out/ff
                       gemm_w4a4 tensor-core kernel itself (RMSNorm+RoPE and
                       GELU epilogues are compiled into the same kernel
                       launch as the surrounding int4 GEMM, so they're not
                       separable from it), plus fused_qkv_norm_rottary /
                       fused_gelu_mlp wrapper-level launches.
  - attention_bf16    scaled_dot_product_attention (flash-attention kernels).
                       Not int4 — Q/K/V/O projections are quantized, the
                       softmax-attention itself runs bf16.
  - high_precision_bf16  Everything NOT DiRotQ-eligible: modulator Linears
                       (norm1.linear / norm1_context.linear / norm.linear),
                       x_embedder, context_embedder, time_text_embed,
                       pos_embed, norm_out, proj_out (the "high-precision
                       branch" in the paper's terminology).
  - norm_elementwise  LayerNorm/RMSNorm/elementwise ops (residual adds, gate
                       multiplies, AdaLN affine) that don't cleanly belong to
                       either GEMM path.
  - other             Anything left over (memcpy, synchronization, etc).

Usage:
  python -u speedup/report_latency_breakdown.py            # B=1,2,4
  python -u speedup/report_latency_breakdown.py --dump-raw # also print the
                                                            # raw top-40
                                                            # kernels (by
                                                            # self CUDA time)
                                                            # before bucketing
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

_HERE = Path(__file__).resolve().parent
_DIROTQ_ROOT = _HERE.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "latency_breakdown.json"

DTYPE = torch.bfloat16
DEVICE = "cuda"


def _stub_torchao():
    import torchao.quantization as _t

    class _Stub:
        pass

    for n in ("Float8WeightOnlyConfig", "Float8DynamicActivationFloat8WeightConfig"):
        if not hasattr(_t, n):
            setattr(_t, n, _Stub)


def make_inputs(batch=1):
    return {
        "hidden_states": torch.randn(batch, 4096, 64, dtype=DTYPE, device=DEVICE),
        "encoder_hidden_states": torch.randn(batch, 512, 4096, dtype=DTYPE, device=DEVICE),
        "pooled_projections": torch.randn(batch, 768, dtype=DTYPE, device=DEVICE),
        "timestep": torch.tensor([500] * batch, dtype=torch.long, device=DEVICE),
        "guidance": torch.tensor([3.5] * batch, dtype=DTYPE, device=DEVICE),
        "img_ids": torch.randint(0, 64, (4096, 3), dtype=DTYPE, device=DEVICE),
        "txt_ids": torch.zeros(512, 3, dtype=DTYPE, device=DEVICE),
        "return_dict": False,
    }


# ===========================================================================
# Model construction — identical to report_flux_w4a4_no_rot_ffdown.py's
# run_phase_nunchaku(), factored out so it returns the model instead of
# measuring + writing JSON inline.
# ===========================================================================

def build_dirotq_w4a4_model():
    from nunchaku.models.transformers.transformer_flux_v2 import (
        NunchakuFluxTransformerBlock, NunchakuFluxSingleTransformerBlock,
    )
    from nunchaku.models.embeddings import NunchakuFluxPosEmbed, pack_rotemb
    from nunchaku.utils import pad_tensor
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers import FluxTransformer2DModel
    from speedup.nunchaku_pack import convert_dirotq_low_weight_to_nunchaku
    from speedup.kernels.int8_rotation import int8_rotation_forward, quantize_U_int8

    def make_double_forward(U_int8, scale_U):
        def fwd(self, hidden_states, encoder_hidden_states, temb,
                image_rotary_emb=None, joint_attention_kwargs=None):
            norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_e, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

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
                    img_hs = hs[:, encoder_hidden_states.shape[1]:]
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

    print("Loading fp16 FluxTransformer2DModel (CPU)...", flush=True)
    fp16_m = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="transformer",
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
            svdq.qweight.data = pack["qweight"].contiguous()
            svdq.wscales.data = pack["wscales"].to(svdq.wscales.dtype).contiguous()
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

    for i in range(n_double):
        orig = fp16_m.transformer_blocks[i].to(DEVICE)
        a = orig.attn
        ws = {
            "qkv_W": torch.cat([a.to_q.weight.data, a.to_k.weight.data, a.to_v.weight.data], dim=0).clone(),
            "qkv_b": torch.cat([a.to_q.bias.data, a.to_k.bias.data, a.to_v.bias.data], dim=0).clone() if a.to_q.bias is not None else None,
            "aqkv_W": torch.cat([a.add_q_proj.weight.data, a.add_k_proj.weight.data, a.add_v_proj.weight.data], dim=0).clone(),
            "aqkv_b": torch.cat([a.add_q_proj.bias.data, a.add_k_proj.bias.data, a.add_v_proj.bias.data], dim=0).clone() if a.add_q_proj.bias is not None else None,
            "to_out_W": a.to_out[0].weight.data.clone(),
            "to_out_b": a.to_out[0].bias.data.clone() if a.to_out[0].bias is not None else None,
            "to_add_out_W": a.to_add_out.weight.data.clone(),
            "to_add_out_b": a.to_add_out.bias.data.clone() if a.to_add_out.bias is not None else None,
            "ff_up_W": orig.ff.net[0].proj.weight.data.clone(),
            "ff_up_b": orig.ff.net[0].proj.bias.data.clone() if orig.ff.net[0].proj.bias is not None else None,
            "ff_down_W": orig.ff.net[2].weight.data.clone(),
            "ff_down_b": orig.ff.net[2].bias.data.clone() if orig.ff.net[2].bias is not None else None,
            "ff_ctx_up_W": orig.ff_context.net[0].proj.weight.data.clone(),
            "ff_ctx_up_b": orig.ff_context.net[0].proj.bias.data.clone() if orig.ff_context.net[0].proj.bias is not None else None,
            "ff_ctx_down_W": orig.ff_context.net[2].weight.data.clone(),
            "ff_ctx_down_b": orig.ff_context.net[2].bias.data.clone() if orig.ff_context.net[2].bias is not None else None,
            "norm1_lin": orig.norm1.linear,
            "norm1_ctx_lin": orig.norm1_context.linear,
        }
        new_blk = NunchakuFluxTransformerBlock(orig, scale_shift=0)
        new_blk = new_blk.to_empty(device=DEVICE)
        new_blk.norm1.linear = ws["norm1_lin"].to(DEVICE)
        new_blk.norm1_context.linear = ws["norm1_ctx_lin"].to(DEVICE)
        pack_into(new_blk.attn.to_qkv, ws["qkv_W"], ws["qkv_b"])
        pack_into(new_blk.attn.to_out[0], ws["to_out_W"], ws["to_out_b"])
        pack_into(new_blk.attn.add_qkv_proj, ws["aqkv_W"], ws["aqkv_b"])
        pack_into(new_blk.attn.to_add_out, ws["to_add_out_W"], ws["to_add_out_b"])
        pack_into(new_blk.ff.net[0].proj, ws["ff_up_W"], ws["ff_up_b"])
        pack_into(new_blk.ff.net[2], ws["ff_down_W"], ws["ff_down_b"])
        pack_into(new_blk.ff_context.net[0].proj, ws["ff_ctx_up_W"], ws["ff_ctx_up_b"])
        pack_into(new_blk.ff_context.net[2], ws["ff_ctx_down_W"], ws["ff_ctx_down_b"])
        new_blk.attn.processor = proc
        new_blk.forward = types.MethodType(make_double_forward(U_3072_int8, scale_U_3072), new_blk)
        fp16_m.transformer_blocks[i] = new_blk
        del orig, ws
        gc.collect(); torch.cuda.empty_cache()

    for i in range(n_single):
        orig = fp16_m.single_transformer_blocks[i].to(DEVICE)
        a = orig.attn
        ws = {
            "qkv_W": torch.cat([a.to_q.weight.data, a.to_k.weight.data, a.to_v.weight.data], dim=0).clone(),
            "qkv_b": torch.cat([a.to_q.bias.data, a.to_k.bias.data, a.to_v.bias.data], dim=0).clone() if a.to_q.bias is not None else None,
            "mlp_fc1_W": orig.proj_mlp.weight.data.clone(),
            "mlp_fc1_b": orig.proj_mlp.bias.data.clone() if orig.proj_mlp.bias is not None else None,
            "proj_out_W": orig.proj_out.weight.data.clone(),
            "proj_out_b": orig.proj_out.bias.data.clone() if orig.proj_out.bias is not None else None,
            "norm_lin": orig.norm.linear,
        }
        new_blk = NunchakuFluxSingleTransformerBlock(orig, scale_shift=0)
        new_blk = new_blk.to_empty(device=DEVICE)
        new_blk.norm.linear = ws["norm_lin"].to(DEVICE)
        pack_into(new_blk.attn.to_qkv, ws["qkv_W"], ws["qkv_b"])
        pack_into(new_blk.mlp_fc1, ws["mlp_fc1_W"], ws["mlp_fc1_b"])
        in_attn = new_blk.attn.to_out.in_features
        in_mlp = new_blk.mlp_fc2.in_features
        pack_into(new_blk.attn.to_out, ws["proj_out_W"][:, :in_attn].contiguous(), None)
        pack_into(new_blk.mlp_fc2, ws["proj_out_W"][:, in_attn:in_attn + in_mlp].contiguous(), ws["proj_out_b"])
        new_blk.attn.processor = proc
        new_blk.forward = types.MethodType(make_single_forward(U_3072_int8, scale_U_3072), new_blk)
        fp16_m.single_transformer_blocks[i] = new_blk
        del orig, ws
        gc.collect(); torch.cuda.empty_cache()

    fp16_m.to(DEVICE)
    axes_dims_rope = fp16_m.config.axes_dims_rope
    fp16_m.pos_embed = NunchakuFluxPosEmbed(
        dim=sum(axes_dims_rope), theta=10000,
        axes_dim=list(axes_dims_rope)).to(DEVICE)
    print(f"  W4A4 weight footprint: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    def patched_forward(self, hidden_states, encoder_hidden_states=None,
                        pooled_projections=None, timestep=None,
                        img_ids=None, txt_ids=None, guidance=None,
                        joint_attention_kwargs=None,
                        controlnet_block_samples=None,
                        controlnet_single_block_samples=None,
                        return_dict=True, controlnet_blocks_repeat=False):
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

        image_rotary_emb = image_rotary_emb.reshape(
            [1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

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
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)

    fp16_m.forward = types.MethodType(patched_forward, fp16_m)
    return fp16_m


# ===========================================================================
# Kernel-name -> category bucketing
# ===========================================================================

# Substring match, checked in order (first match wins). Names come from
# `torch.profiler`'s CUDA kernel events (nvtx/kineto names for aten ops,
# Triton-compiled kernels, and nunchaku's raw CUDA kernel symbols).
BUCKET_RULES = [
    ("rotation", ["int8_rotation", "rotation_kernel", "quantize_u_int8", "triton__0", "gemm_a8w8_int8"]),
    ("quant_dequant", ["quantize_w4a4", "act_quant", "dequant"]),
    ("int4_gemm", ["gemm_w4a4", "w4a4", "fused_gelu_mlp", "fused_qkv_norm_rottary",
                   "svdq", "awq", "nunchaku"]),
    ("attention_bf16", ["scaled_dot_product_attention", "flash", "attention", "fmha", "sdpa"]),
    ("high_precision_bf16", ["addmm", "gemm", "cublas", "cutlass", "linear"]),
    ("norm_elementwise", ["norm", "layer_norm", "rsqrt", "mul", "add", "clip", "cat",
                          "elementwise", "index", "copy", "fill", "reshape", "view",
                          "vectorized", "clamp"]),
]
# NOTE: order matters — more specific rotation/int4/attention patterns are
# checked before the generic "gemm"/"norm"/elementwise catch-alls below them,
# so e.g. a Triton rotation kernel matches "rotation" and not "gemm" even
# though internally it's also a matmul.


def bucket_name(name: str) -> str:
    n = name.lower()
    for bucket, patterns in BUCKET_RULES:
        for p in patterns:
            if p in n:
                return bucket
    return "other"


def profile_batch(model, B, warmup=3, iters=8, dump_raw=False):
    """Profile via the Chrome-trace export, not key_averages().

    key_averages()'s self_device_time_total double-counts: a CPU op that
    launches exactly one child kernel (e.g. aten::_flash_attention_forward
    -> flash_fwd_kernel) gets the SAME device time attributed to both the
    CPU-side event and the kernel event, so summing over all events counts
    it twice. The Chrome trace's raw slices are unambiguous: cat=="kernel"
    (or gpu_memcpy/gpu_memset) is real, non-overlapping GPU-executed time.
    """
    import tempfile
    inp = make_inputs(B)
    with torch.no_grad():
        for _ in range(warmup):
            model(**inp)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(**inp)
            torch.cuda.synchronize()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        trace_path = tf.name
    prof.export_chrome_trace(trace_path)
    with open(trace_path) as f:
        trace = json.load(f)
    Path(trace_path).unlink(missing_ok=True)

    device_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    kernel_events = [e for e in trace["traceEvents"]
                     if e.get("cat") in device_cats and "dur" in e]

    total_us = sum(e["dur"] for e in kernel_events)
    if dump_raw:
        agg = {}
        for e in kernel_events:
            agg[e["name"]] = agg.get(e["name"], 0.0) + e["dur"]
        print(f"\n--- B={B} raw top-40 kernels by total device time (avg over {iters} iters) ---")
        for name, us in sorted(agg.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {us/iters/1000:8.4f} ms/iter  ({100*us/total_us:5.1f}%)  {name}")

    buckets = {}
    for e in kernel_events:
        b = bucket_name(e["name"])
        buckets[b] = buckets.get(b, 0.0) + e["dur"]

    per_iter_ms = {b: t / iters / 1000 for b, t in buckets.items()}
    total_ms = sum(per_iter_ms.values())
    del inp
    gc.collect(); torch.cuda.empty_cache()
    return per_iter_ms, total_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-raw", action="store_true",
                     help="print raw top-40 kernels per batch before bucketing")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    _stub_torchao()
    model = build_dirotq_w4a4_model()

    results = {}
    print()
    print("=" * 100)
    print("DiRotQ-W4A4 (no-rot ff_down) — kernel-level latency breakdown (torch.profiler, self CUDA time)")
    print("=" * 100)
    for B in args.batches:
        per_iter_ms, total_ms = profile_batch(model, B, dump_raw=args.dump_raw)
        results[B] = {"buckets_ms": per_iter_ms, "profiled_total_ms": total_ms}
        print(f"\nB={B}  (profiled GPU-busy total: {total_ms:.1f} ms)")
        for b, ms in sorted(per_iter_ms.items(), key=lambda kv: -kv[1]):
            pct = 100 * ms / total_ms if total_ms else 0
            print(f"  {b:<22} {ms:8.2f} ms  ({pct:5.1f}%)")

    out = {
        "report": "DiRotQ-W4A4 (no-rot ff_down) kernel-level latency breakdown",
        "gpu": "NVIDIA GeForce RTX 4090",
        "method": "torch.profiler self CUDA device time, averaged over 8 iters after 3 warmup, "
                  "bucketed by kernel-name substring match. Peak-memory breakdown is not "
                  "re-measured here — see results/flux_w4a4_no_rot_ffdown.json (same model, "
                  "already has the weights/activations/peak-VRAM numbers) and the README's "
                  "combined table.",
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
