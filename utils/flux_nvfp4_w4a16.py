"""Hardware-faithful NVFP4 W4A16 weights for FLUX adaptive norms.

The 76 adaptive-normalization Linear layers are intentionally outside the
normal ActQuantWrapper routing: their input remains BF16 while their weight is
stored as E2M1 payload plus a layer-global FP32 scale and K16 E4M3 scales.
Ada execution materializes the decoded BF16 weight, so this module is an
accuracy reference rather than a native packed-NVFP4 performance kernel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux_real_quant import validate_relocated_model_id
from .hardware_weight_fp4 import (
    WEIGHT_GROUP_SIZE,
    decode_packing_record,
    frozen_block_scales,
    hardware_global_scale,
    pack_nibbles,
    tensor_sha256,
)
from .quant_utils import round_to_nf4_codebook


SCHEMA = "dirotq.flux_nvfp4_w4a16"
VERSION = 1
FORMAT = "hardware-fixed-e2"
TARGET_SUFFIXES = (".norm1.linear", ".norm1_context.linear", ".norm.linear")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(model_id: str) -> dict[str, Any]:
    return {
        "model": "flux-dev",
        "model_id": str(model_id),
        "weight_bits": 4,
        "activation_bits": 16,
        "weight_format": "E2M1",
        "group_size": WEIGHT_GROUP_SIZE,
        "global_scale": "layer-fp32-amax-over-2688",
        "block_scale": "e4m3fn-k16-amax-over-6",
        "quantization": "hardware-faithful-nvfp4-rtn",
        "target_families": ["norm1.linear", "norm1_context.linear", "norm.linear"],
        "runtime": "decoded-bf16-accuracy-reference",
    }


def is_target(name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES)


@torch.inference_mode()
def quantize_weight(weight: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("NVFP4 W4A16 weight must be a 2-D FP16/BF16 tensor")
    source = weight.detach().float()
    if not torch.isfinite(source).all():
        raise ValueError("non-finite NVFP4 W4A16 weight")
    alpha = hardware_global_scale(source)
    scales, raw_scales = frozen_block_scales(source, FORMAT, alpha)
    n, k = source.shape
    pad = (-k) % WEIGHT_GROUP_SIZE
    padded = F.pad(source, (0, pad)) if pad else source
    blocks = padded.reshape(n, -1, WEIGHT_GROUP_SIZE)
    normalized = blocks / (alpha * scales.unsqueeze(-1))
    codes = round_to_nf4_codebook(normalized)
    reconstructed = (codes * scales.unsqueeze(-1) * alpha).reshape(n, -1)[:, :k]
    # Avoid the general [N,K,15] distance tensor: these values are already
    # exact E2M1 codes, so a bounded series of equality masks is equivalent
    # and keeps peak host memory proportional to the weight size.
    codebook = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    indices = torch.empty_like(codes, dtype=torch.uint8)
    matched = torch.zeros_like(codes, dtype=torch.bool)
    for index, value in enumerate(codebook):
        mask = codes == value
        indices[mask] = index
        matched |= mask
    if not matched.all():
        raise AssertionError("E2M1 payload contains a non-codebook value")
    record = {
        "format": FORMAT,
        "global_scale": alpha.float().cpu(),
        "block_scales": scales.to(torch.float8_e4m3fn).cpu(),
        "packed_payload": pack_nibbles(indices.reshape(n, -1)).cpu(),
        "logical_shape": [k, n],
        "stored_shape": [n, k],
        "group_size": WEIGHT_GROUP_SIZE,
        "high_branch_hash": "none-w4a16",
        "reconstructed_low_hash_fp32": tensor_sha256(reconstructed),
    }
    decoded = decode_packing_record(record, dtype=torch.float32)
    torch.testing.assert_close(decoded, reconstructed.cpu(), rtol=0, atol=0)
    maximum = 6.0
    nonzero = raw_scales > 0
    relative = torch.zeros_like(raw_scales)
    relative[nonzero] = (
        (scales[nonzero] - raw_scales[nonzero]).abs() / raw_scales[nonzero]
    )
    report = {
        "logical_shape": [n, k],
        "padded_k": k + pad,
        "payload_bytes": record["packed_payload"].numel(),
        "block_scale_bytes": record["block_scales"].numel(),
        "global_scale_bytes": 4,
        "global_scale": float(alpha),
        "block_scale_min": float(scales.min()),
        "block_scale_max": float(scales.max()),
        "scale_rounding_relative_mean": (
            float(relative[nonzero].mean()) if nonzero.any() else 0.0
        ),
        "saturation_count": int((normalized.abs() > maximum).sum()),
        "payload_count": normalized.numel(),
        "reconstructed_fp32_sha256": tensor_sha256(decoded),
    }
    return record, report


class NVFP4W4A16Linear(nn.Module):
    """BF16 activation with an exactly decoded hardware-NVFP4 weight."""

    def __init__(
        self, record: dict[str, Any], *, bias: torch.Tensor | None,
        runtime_dtype: torch.dtype = torch.bfloat16,
        require_cuda: bool = True,
    ) -> None:
        super().__init__()
        if record.get("format") != FORMAT or int(record["group_size"]) != 16:
            raise RuntimeError("invalid NVFP4 W4A16 packing record")
        decoded = decode_packing_record(record, dtype=runtime_dtype)
        expected_hash = record.get("reconstructed_low_hash_fp32")
        actual_hash = tensor_sha256(decode_packing_record(record, dtype=torch.float32))
        if expected_hash != actual_hash:
            raise RuntimeError("NVFP4 W4A16 reconstructed weight hash mismatch")
        self.register_buffer("weight", decoded.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().to(runtime_dtype).contiguous())
        self.in_features = int(record["stored_shape"][1])
        self.out_features = int(record["stored_shape"][0])
        self.require_cuda = bool(require_cuda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("NVFP4 W4A16 activation must remain FP16/BF16")
        if x.device != self.weight.device:
            raise RuntimeError("NVFP4 W4A16 activation/weight device mismatch")
        if self.require_cuda and x.device.type != "cuda":
            raise RuntimeError("NVFP4 W4A16 runtime forbids silent CPU fallback")
        return F.linear(x, self.weight, self.bias)


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


@torch.inference_mode()
def build_cache_from_safetensors(transformer_dir: Path) -> tuple[dict, dict]:
    from safetensors import safe_open

    transformer_dir = Path(transformer_dir)
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    weight_keys = sorted(
        key for key in weight_map
        if key.endswith(tuple(suffix + ".weight" for suffix in TARGET_SUFFIXES))
    )
    if len(weight_keys) != 76:
        raise RuntimeError(f"expected 76 adaptive-norm weights, found {len(weight_keys)}")
    handles = {
        shard: safe_open(str(transformer_dir / shard), framework="pt", device="cpu")
        for shard in sorted({weight_map[key] for key in weight_keys})
    }
    states, reports = {}, {}
    for weight_key in weight_keys:
        name = weight_key.removesuffix(".weight")
        handle = handles[weight_map[weight_key]]
        weight = handle.get_tensor(weight_key)
        bias_key = name + ".bias"
        bias = handles[weight_map[bias_key]].get_tensor(bias_key)
        record, report = quantize_weight(weight)
        states[name] = {"packing": record, "bias": bias.detach().cpu()}
        reports[name] = report
    aggregate = {
        "layers": len(states),
        "payload_bytes": sum(r["payload_bytes"] for r in reports.values()),
        "block_scale_bytes": sum(r["block_scale_bytes"] for r in reports.values()),
        "global_scale_bytes": sum(r["global_scale_bytes"] for r in reports.values()),
        "bias_bytes": sum(s["bias"].numel() * s["bias"].element_size() for s in states.values()),
        "saturation_count": sum(r["saturation_count"] for r in reports.values()),
        "payload_count": sum(r["payload_count"] for r in reports.values()),
        "group_size": 16,
        "weight_format": "E2M1",
        "scale_format": "FP32-global-plus-E4M3-K16",
        "source": "exact-HF-safetensors-adaptive-norm-weights",
    }
    aggregate["persistent_packed_bytes"] = (
        aggregate["payload_bytes"] + aggregate["block_scale_bytes"]
        + aggregate["global_scale_bytes"] + aggregate["bias_bytes"]
    )
    return {"schema": SCHEMA, "version": VERSION, "layers": states}, {
        "aggregate": aggregate, "layers": reports,
    }


def install_states(
    transformer: nn.Module, states: dict[str, dict[str, Any]], *, require_cuda: bool,
) -> dict[str, Any]:
    expected = {name for name, mod in transformer.named_modules() if is_target(name, mod)}
    if expected != set(states) or len(expected) != 76:
        raise RuntimeError("NVFP4 W4A16 cache layer coverage mismatch")
    hashes = {}
    for name in sorted(states):
        state = states[name]
        packed = NVFP4W4A16Linear(
            state["packing"], bias=state.get("bias"), require_cuda=require_cuda,
        )
        hashes[name] = state["packing"]["reconstructed_low_hash_fp32"]
        _set_submodule(transformer, name, packed)
    return {"layers": len(states), "reconstructed_hashes": hashes}


def save_cache(cache: dict, report: dict, path: Path, *, cache_provenance: dict) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite NVFP4 W4A16 cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({**cache, "provenance": cache_provenance}, temporary)
    temporary.replace(path)
    manifest = {
        **report["aggregate"], "cache": str(path), "cache_size": path.stat().st_size,
        "cache_sha256": sha256_file(path), "provenance": cache_provenance,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_cache(
    transformer: nn.Module, path: Path, *, expected_provenance: dict,
    runtime_model_id: str, require_cuda: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise RuntimeError("unsupported NVFP4 W4A16 cache schema")
    actual = payload.get("provenance", {})
    for key, value in expected_provenance.items():
        if key != "model_id" and actual.get(key) != value:
            raise RuntimeError(f"NVFP4 W4A16 provenance mismatch for {key}")
    validate_relocated_model_id(actual.get("model_id", ""), runtime_model_id)
    installed = install_states(transformer, payload["layers"], require_cuda=require_cuda)
    return {
        **installed, "cache": str(path), "cache_size": path.stat().st_size,
        "cache_sha256": sha256_file(path), "provenance": actual,
    }
