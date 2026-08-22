"""Real packed W4A16 replacements for FLUX adaptive-norm Linear layers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packed_int4_runtime import decode_weight_int4, pack_signed_int4
from .flux_real_quant import validate_relocated_model_id


SCHEMA = "dirotq.flux_modulator_w4a16"
VERSION = 1
GROUP_SIZE = 64
TARGET_SUFFIXES = (".norm1.linear", ".norm1_context.linear", ".norm.linear")


def w4a16_provenance(model_id: str) -> dict[str, Any]:
    normalized = str(model_id).lower()
    model = "flux-schnell" if "flux.1-schnell" in normalized else "flux-dev"
    return {
        "model": model,
        "model_id": str(model_id),
        "group_size": GROUP_SIZE,
        "weight_bits": 4,
        "activation_bits": 16,
        "scale_dtype": "torch.bfloat16",
        "quantization": "symmetric-signed-int4-rtn-amax-div7",
        "target_families": ["norm1.linear", "norm1_context.linear", "norm.linear"],
    }


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_w4a16_modulator(name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES)


@torch.inference_mode()
def quantize_w4a16_weight(
    weight: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Symmetric K-group RTN used by the SVDQuant-compatible W4A16 arm."""
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("W4A16 weight must be a 2-D FP16/BF16 tensor")
    n, logical_k = weight.shape
    padded_k = ((logical_k + group_size - 1) // group_size) * group_size
    work = weight.detach()
    if padded_k != logical_k:
        work = F.pad(work, (0, padded_k - logical_k))
    grouped = work.reshape(n, -1, group_size).float()
    amax = grouped.abs().amax(dim=-1)
    scales_fp32 = amax / 7.0
    safe = torch.where(amax == 0, torch.ones_like(scales_fp32), scales_fp32)
    scales = safe.to(torch.bfloat16)
    codes = torch.round(grouped / scales.float().unsqueeze(-1)).clamp(-8, 7)
    codes = torch.where(amax.unsqueeze(-1) == 0, torch.zeros_like(codes), codes)
    payload = pack_signed_int4(codes.to(torch.int8).reshape(n, padded_k)).contiguous()
    report = {
        "logical_shape": [n, logical_k],
        "padded_shape": [n, padded_k],
        "payload_bytes": payload.numel(),
        "scale_bytes": scales.numel() * scales.element_size(),
        "saturation_count": int(((codes == -8) | (codes == 7)).sum().item()),
        "zero_groups": int((amax == 0).sum().item()),
    }
    return payload, scales.contiguous(), report


class PackedW4A16Linear(nn.Module):
    """BF16 activation times persistent packed signed-INT4 weight."""

    def __init__(
        self,
        qweight: torch.Tensor,
        weight_scales: torch.Tensor,
        *,
        logical_k: int,
        bias: torch.Tensor | None,
        require_cuda: bool = True,
    ) -> None:
        super().__init__()
        if qweight.dtype != torch.uint8 or weight_scales.dtype != torch.bfloat16:
            raise TypeError("W4A16 buffers require uint8 payload and BF16 scales")
        self.register_buffer("qweight", qweight.detach().contiguous())
        self.register_buffer("weight_scales", weight_scales.detach().contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())
        self.in_features = int(logical_k)
        self.out_features = int(qweight.shape[0])
        self.group_size = GROUP_SIZE
        self.require_cuda = bool(require_cuda)

    @property
    def persistent_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.qweight, self.weight_scales, self.bias)
            if tensor is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.qweight.device:
            raise RuntimeError("W4A16 activation and packed weight are on different devices")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("W4A16 activation must remain FP16/BF16")
        original = x.shape[:-1]
        flat = x.reshape(-1, self.in_features).contiguous()
        if flat.device.type == "cuda":
            from .packed_w4a16_triton import packed_w4a16_gemm

            output = packed_w4a16_gemm(
                flat, self.qweight, self.weight_scales, self.in_features
            )
        else:
            if self.require_cuda:
                raise RuntimeError("real W4A16 runtime forbids silent CPU fallback")
            decoded = decode_weight_int4(
                self.qweight,
                self.weight_scales,
                self.in_features,
                self.group_size,
                flat.dtype,
            )
            output = F.linear(flat, decoded, None).float()
        if self.bias is not None:
            output.add_(self.bias.float())
        return output.to(x.dtype).reshape(*original, self.out_features)


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


def _state_from_module(module: PackedW4A16Linear) -> dict[str, Any]:
    return {
        "qweight": module.qweight.detach().cpu(),
        "weight_scales": module.weight_scales.detach().cpu(),
        "bias": None if module.bias is None else module.bias.detach().cpu(),
        "logical_k": module.in_features,
        "out_features": module.out_features,
        "group_size": module.group_size,
    }


@torch.inference_mode()
def build_and_install_w4a16_modulators(
    transformer: nn.Module,
    *,
    require_cuda: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = [name for name, mod in transformer.named_modules() if is_w4a16_modulator(name, mod)]
    if len(names) != 76:
        raise RuntimeError(f"expected 76 FLUX adaptive-norm Linears, found {len(names)}")
    states: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for name in names:
        dense = transformer.get_submodule(name)
        if not isinstance(dense, nn.Linear):
            raise TypeError(f"{name}: expected nn.Linear")
        payload, scales, report = quantize_w4a16_weight(dense.weight)
        packed = PackedW4A16Linear(
            payload,
            scales,
            logical_k=dense.in_features,
            bias=dense.bias,
            require_cuda=require_cuda,
        )
        _set_submodule(transformer, name, packed)
        states[name] = _state_from_module(packed)
        reports[name] = {**report, "persistent_bytes": packed.persistent_bytes}
    aggregate = {
        "layers": len(states),
        "payload_bytes": sum(v["payload_bytes"] for v in reports.values()),
        "scale_bytes": sum(v["scale_bytes"] for v in reports.values()),
        "bias_bytes": sum(
            0 if state["bias"] is None else state["bias"].numel() * state["bias"].element_size()
            for state in states.values()
        ),
        "persistent_bytes": sum(v["persistent_bytes"] for v in reports.values()),
        "group_size": GROUP_SIZE,
        "scale_dtype": "torch.bfloat16",
        "quantization": "symmetric-signed-int4-rtn-amax-div7",
    }
    return {"schema": SCHEMA, "version": VERSION, "layers": states}, {
        "aggregate": aggregate,
        "layers": reports,
    }


@torch.inference_mode()
def build_w4a16_cache_from_safetensors(
    transformer_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the same sidecar directly from the immutable HF weight shards.

    Loading the complete FLUX pipeline merely to access these 76 tensors adds
    tens of GiB of unrelated host memory.  Safetensors slicing keeps only one
    source Linear live while preserving the exact official checkpoint bytes.
    """
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
        raise RuntimeError(f"expected 76 checkpoint modulator weights, found {len(weight_keys)}")
    handles = {
        shard: safe_open(str(transformer_dir / shard), framework="pt", device="cpu")
        for shard in sorted({weight_map[key] for key in weight_keys})
    }
    states: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for weight_key in weight_keys:
        module_name = weight_key.removesuffix(".weight")
        handle = handles[weight_map[weight_key]]
        weight = handle.get_tensor(weight_key)
        bias_key = module_name + ".bias"
        bias = handles[weight_map[bias_key]].get_tensor(bias_key)
        payload, scales, report = quantize_w4a16_weight(weight)
        state = {
            "qweight": payload.cpu(),
            "weight_scales": scales.cpu(),
            "bias": bias.detach().cpu(),
            "logical_k": int(weight.shape[1]),
            "out_features": int(weight.shape[0]),
            "group_size": GROUP_SIZE,
        }
        persistent = (
            state["qweight"].numel()
            + state["weight_scales"].numel() * 2
            + state["bias"].numel() * state["bias"].element_size()
        )
        states[module_name] = state
        reports[module_name] = {**report, "persistent_bytes": persistent}
        del weight, bias, payload, scales
    aggregate = {
        "layers": len(states),
        "payload_bytes": sum(v["payload_bytes"] for v in reports.values()),
        "scale_bytes": sum(v["scale_bytes"] for v in reports.values()),
        "bias_bytes": sum(
            state["bias"].numel() * state["bias"].element_size()
            for state in states.values()
        ),
        "persistent_bytes": sum(v["persistent_bytes"] for v in reports.values()),
        "group_size": GROUP_SIZE,
        "scale_dtype": "torch.bfloat16",
        "quantization": "symmetric-signed-int4-rtn-amax-div7",
        "source": "exact-HF-safetensors-modulator-weights",
    }
    return {"schema": SCHEMA, "version": VERSION, "layers": states}, {
        "aggregate": aggregate,
        "layers": reports,
    }


def install_w4a16_states(
    transformer: nn.Module,
    states: dict[str, dict[str, Any]],
    *,
    require_cuda: bool = True,
) -> dict[str, Any]:
    expected = {name for name, mod in transformer.named_modules() if is_w4a16_modulator(name, mod)}
    if expected != set(states) or len(expected) != 76:
        raise RuntimeError("W4A16 cache layer coverage does not match FLUX modulators")
    total = 0
    for name in sorted(states):
        state = states[name]
        if int(state["group_size"]) != GROUP_SIZE:
            raise RuntimeError(f"{name}: W4A16 group-size mismatch")
        packed = PackedW4A16Linear(
            state["qweight"],
            state["weight_scales"],
            logical_k=int(state["logical_k"]),
            bias=state.get("bias"),
            require_cuda=require_cuda,
        )
        _set_submodule(transformer, name, packed)
        total += packed.persistent_bytes
    return {"layers": len(states), "persistent_bytes": total}


def save_w4a16_cache(
    cache: dict[str, Any],
    report: dict[str, Any],
    path: Path,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**cache, "provenance": provenance}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    manifest = {
        **report["aggregate"],
        "cache": str(path),
        "cache_sha256": sha256_file(path),
        "cache_size": path.stat().st_size,
        "provenance": provenance,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_w4a16_cache(
    transformer: nn.Module,
    path: Path,
    *,
    expected_provenance: dict[str, Any],
    runtime_model_id: str | None = None,
    require_cuda: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise RuntimeError("unsupported FLUX W4A16 cache schema")
    provenance = payload.get("provenance", {})
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise RuntimeError(
                f"W4A16 provenance mismatch for {key}: {provenance.get(key)!r} != {value!r}"
            )
    if runtime_model_id is not None:
        validate_relocated_model_id(provenance.get("model_id", ""), runtime_model_id)
    result = install_w4a16_states(
        transformer, payload["layers"], require_cuda=require_cuda
    )
    return {
        **result,
        "cache": str(path),
        "cache_sha256": sha256_file(path),
        "cache_size": path.stat().st_size,
        "provenance": provenance,
    }
