"""Install the five FLUX shared-PCA arms as persistent packed INT4 models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .packed_int4_runtime import (
    PackedSplitInt4Linear,
    decode_weight_int4,
    fit_reconstructed_int4_weight,
    packed_linear_state,
)
from .quant_utils import ActQuantWrapper, find_qlayers


SCHEMA = "dirotq.flux_split_real_int4"
VERSION = 1


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _drop_dense_aliases(layer: ActQuantWrapper) -> None:
    # ActQuantWrapper registers module.weight/bias a second time for legacy
    # callers.  They must be removed or the supposedly packed model silently
    # retains the dense BF16 matrix.
    for name in ("weight", "bias"):
        if hasattr(layer, name):
            delattr(layer, name)
        layer.register_parameter(name, None)


def _split_cached_weight(layer: ActQuantWrapper) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = layer.module.weight.detach()
    high = int(layer.quantizer.high_bits_length)
    if layer.rotation_per_head is not None:
        heads = int(layer.num_heads)
        head_dim = int(layer.head_dim)
        if weight.shape[1] != heads * head_dim:
            raise ValueError("per-head packed weight shape mismatch")
        per_head = weight.reshape(weight.shape[0], heads, head_dim)
        low_dim = head_dim - high
        low = per_head[:, :, :low_dim].reshape(weight.shape[0], heads * low_dim)
        tail = (
            per_head[:, :, low_dim:].reshape(weight.shape[0], heads * high)
            if high else None
        )
        return low.contiguous(), None if tail is None else tail.contiguous()
    if high:
        return weight[:, :-high].contiguous(), weight[:, -high:].contiguous()
    return weight.contiguous(), None


def _install_state(
    layer: ActQuantWrapper,
    state: dict[str, Any],
    *,
    require_cuda: bool,
) -> PackedSplitInt4Linear:
    packed = PackedSplitInt4Linear(
        state["qweight"],
        state["weight_scales"],
        logical_low_k=int(state["logical_low_k"]),
        group_size=int(state["group_size"]),
        high_weight=state.get("high_weight"),
        bias=state.get("bias"),
        require_cuda=require_cuda,
    )
    if packed.in_features != layer.module.in_features:
        raise ValueError(
            f"packed/cache input mismatch: {packed.in_features} != {layer.module.in_features}"
        )
    if packed.out_features != layer.module.out_features:
        raise ValueError(
            f"packed/cache output mismatch: {packed.out_features} != {layer.module.out_features}"
        )
    _drop_dense_aliases(layer)
    layer.module = packed
    layer._real_int4 = True
    if (
        layer.rotation is not None
        or layer.rotation_per_head is not None
        or getattr(layer, "use_hadamard", False)
        or getattr(layer, "perm_idx", None) is not None
    ):
        layer._unrot_fused = True
    return packed


def exact_gptq_cache_from_states(
    states: dict[str, dict[str, Any]],
    *,
    expected_layers: int,
    gptq_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a sidecar manifest from GPTQ's pre-BF16 codes and scales.

    Unlike :func:`build_packed_model_from_reconstructed_cache`, this path does
    not infer anything from a fake-quant checkpoint.  ``gptq_utils`` records
    each signed code and frozen FP32 group scale before the reconstructed
    matrix is cast to BF16, so real and fake arms differ only in decode and
    GEMM accumulation order.
    """
    if len(states) != expected_layers:
        raise RuntimeError(
            f"exact packed GPTQ coverage mismatch: {len(states)} != {expected_layers}"
        )
    methods: dict[str, int] = {}
    payload_bytes = scale_bytes = high_bytes = bias_bytes = 0
    max_pack_error = 0.0
    for name, state in states.items():
        if state["qweight"].dtype != torch.uint8:
            raise TypeError(f"{name}: GPTQ payload must be packed uint8")
        if state["weight_scales"].dtype != torch.float32:
            raise TypeError(f"{name}: exact GPTQ scales must remain FP32")
        method = str(state["quantization_method"])
        methods[method] = methods.get(method, 0) + 1
        payload_bytes += state["qweight"].numel()
        scale_bytes += state["weight_scales"].numel() * 4
        if state.get("high_weight") is not None:
            high_bytes += state["high_weight"].numel() * state["high_weight"].element_size()
        if state.get("bias") is not None:
            bias_bytes += state["bias"].numel() * state["bias"].element_size()
        max_pack_error = max(
            max_pack_error, float(state.get("packing_max_abs_error_fp32", 0.0))
        )
    if gptq_summary.get("rtn_fallback_layers") != 0:
        raise RuntimeError("exact packed cache refuses GPTQ/RTN fallback")
    aggregate = {
        "active_layers": len(states),
        "gptq_summary": gptq_summary,
        "quantization_methods": methods,
        "payload_bytes": payload_bytes,
        "scale_bytes": scale_bytes,
        "high_weight_bytes": high_bytes,
        "bias_bytes": bias_bytes,
        "persistent_layer_bytes": payload_bytes + scale_bytes + high_bytes + bias_bytes,
        "scale_dtype": "torch.float32",
        "max_abs_packing_error_fp32": max_pack_error,
        "source": "exact-pre-bf16-gptq-codes-and-frozen-scales",
    }
    return (
        {"schema": SCHEMA, "version": VERSION, "layers": states},
        {"aggregate": aggregate, "layers": {}},
    )


@torch.inference_mode()
def validate_states_against_fake_quant_cache(
    states: dict[str, dict[str, Any]],
    fake_cache_path: Path,
) -> dict[str, Any]:
    """Prove that packed state reconstructs the matched BF16 fake cache."""
    fake = torch.load(
        Path(fake_cache_path), map_location="cpu", weights_only=False, mmap=True
    )
    compared = elements = unequal = 0
    max_abs_error = 0.0
    for name, state in states.items():
        key = f"{name}.module.weight"
        if key not in fake:
            raise RuntimeError(f"fake-quant provenance is missing {key}")
        target = fake[key]
        low = decode_weight_int4(
            state["qweight"],
            state["weight_scales"],
            int(state["logical_low_k"]),
            int(state["group_size"]),
            target.dtype,
        )
        high = state.get("high_weight")
        layout = state.get("stored_layout", "low-only")
        if layout == "low-only":
            reconstructed = low
        elif layout == "low-then-high":
            reconstructed = torch.cat((low, high.to(target.dtype)), dim=1)
        elif layout == "per-head-interleaved":
            heads = int(state["num_heads"])
            head_dim = int(state["head_dim"])
            high_per_head = int(state["high_per_head"])
            low_per_head = head_dim - high_per_head
            reconstructed = torch.cat(
                (
                    low.reshape(low.shape[0], heads, low_per_head),
                    high.to(target.dtype).reshape(high.shape[0], heads, high_per_head),
                ),
                dim=2,
            ).reshape(low.shape[0], heads * head_dim)
        else:
            raise ValueError(f"{name}: unknown stored layout {layout!r}")
        if reconstructed.shape != target.shape:
            raise RuntimeError(
                f"{name}: reconstructed/fake shape mismatch "
                f"{tuple(reconstructed.shape)} != {tuple(target.shape)}"
            )
        diff = (reconstructed.float() - target.float()).abs()
        layer_unequal = int((reconstructed != target).sum().item())
        unequal += layer_unequal
        elements += target.numel()
        max_abs_error = max(max_abs_error, float(diff.max().item()))
        compared += 1
        if layer_unequal:
            raise RuntimeError(
                f"{name}: exact packed build does not reproduce matched fake cache; "
                f"unequal={layer_unequal}/{target.numel()}, max={float(diff.max()):.8g}"
            )
    return {
        "layers": compared,
        "elements": elements,
        "unequal_elements": unequal,
        "max_abs_error": max_abs_error,
        "bitwise_bf16_equal": unequal == 0,
    }


def install_packed_states(
    transformer: torch.nn.Module,
    states: dict[str, dict[str, Any]],
    *,
    require_cuda: bool = True,
) -> dict[str, Any]:
    qlayers = find_qlayers(transformer, layers=[ActQuantWrapper])
    if set(states) != set(qlayers):
        raise ValueError("packed-state layer coverage does not match wrapped model")
    total = 0
    for name, layer in qlayers.items():
        total += _install_state(layer, states[name], require_cuda=require_cuda).persistent_bytes
    return {"active_layers": len(qlayers), "persistent_layer_bytes": total}


def real_int4_storage_report(transformer: torch.nn.Module) -> dict[str, int]:
    """Byte accounting for the materialized packed transformer.

    Parameter/buffer storages are deduplicated by storage identity.  Online
    PCA/residual frames are ordinary attributes rather than registered
    buffers, so they are counted separately, also by unique storage.
    """
    categories = {
        "packed_low_payload": 0,
        "low_group_scales_fp32": 0,
        "protected_high_bf16": 0,
        "active_bias": 0,
        "other_model_parameters_and_buffers": 0,
        "online_pca_residual_frames": 0,
    }
    seen_model: set[tuple[Any, ...]] = set()

    def _storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
        storage = tensor.untyped_storage()
        return (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())

    packed_tensor_ids: set[int] = set()
    for module in transformer.modules():
        if not isinstance(module, PackedSplitInt4Linear):
            continue
        for key, tensor in (
            ("packed_low_payload", module.qweight),
            ("low_group_scales_fp32", module.weight_scales),
            ("protected_high_bf16", module.high_weight),
            ("active_bias", module.bias),
        ):
            if tensor is None:
                continue
            packed_tensor_ids.add(id(tensor))
            storage_key = _storage_key(tensor)
            if storage_key not in seen_model:
                categories[key] += tensor.untyped_storage().nbytes()
                seen_model.add(storage_key)

    for tensor in list(transformer.parameters()) + list(transformer.buffers()):
        if id(tensor) in packed_tensor_ids:
            continue
        storage_key = _storage_key(tensor)
        if storage_key not in seen_model:
            categories["other_model_parameters_and_buffers"] += tensor.untyped_storage().nbytes()
            seen_model.add(storage_key)

    seen_rotations: set[tuple[Any, ...]] = set()
    for module in transformer.modules():
        if not isinstance(module, ActQuantWrapper):
            continue
        for tensor in (module.rotation, module.rotation_per_head):
            if tensor is None:
                continue
            key = _storage_key(tensor)
            if key not in seen_rotations:
                categories["online_pca_residual_frames"] += tensor.untyped_storage().nbytes()
                seen_rotations.add(key)
    categories["persistent_transformer_and_frames"] = sum(categories.values())
    return categories


@torch.inference_mode()
def build_packed_model_from_reconstructed_cache(
    transformer: torch.nn.Module,
    *,
    group_size: int = 64,
    scale_dtype: torch.dtype = torch.float32,
    require_cuda: bool = True,
    max_decode_error: float = 0.02,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace active wrappers and return a portable packed sidecar/report."""
    qlayers = find_qlayers(transformer, layers=[ActQuantWrapper])
    layer_states: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    total_bytes = 0
    active = 0
    for name, layer in qlayers.items():
        if layer.quantizer.bits >= 16:
            continue
        if not isinstance(layer.module, torch.nn.Linear):
            raise TypeError(f"{name}: expected dense reconstructed Linear before packing")
        low, high = _split_cached_weight(layer)
        payload, scales, report = fit_reconstructed_int4_weight(
            low, group_size=group_size, scale_dtype=scale_dtype
        )
        if report.max_abs_decode_error > max_decode_error:
            raise RuntimeError(
                f"{name}: reconstructed-cache packing error "
                f"{report.max_abs_decode_error:.6g} exceeds {max_decode_error:.6g}"
            )
        state = {
            "qweight": payload.cpu(),
            "weight_scales": scales.cpu(),
            "high_weight": None if high is None else high.cpu(),
            "bias": None if layer.module.bias is None else layer.module.bias.detach().cpu(),
            "logical_low_k": low.shape[1],
            "group_size": group_size,
            "in_features": layer.module.in_features,
            "out_features": layer.module.out_features,
        }
        packed = _install_state(layer, state, require_cuda=require_cuda)
        layer_states[name] = state
        reports[name] = {
            "elements": report.elements,
            "exact_bf16_elements": report.exact_bf16_elements,
            "exact_fraction": report.exact_fraction,
            "max_abs_decode_error": report.max_abs_decode_error,
            "mean_abs_decode_error": report.mean_abs_decode_error,
            "payload_bytes": report.payload_bytes,
            "scale_bytes": report.scale_bytes,
            "high_weight_bytes": 0 if high is None else high.numel() * high.element_size(),
            "persistent_bytes": packed.persistent_bytes,
        }
        total_bytes += packed.persistent_bytes
        active += 1
    if active != len(qlayers):
        raise RuntimeError(
            f"expected every wrapped FLUX Linear to be active: packed={active}, wrappers={len(qlayers)}"
        )
    aggregate = {
        "active_layers": active,
        "persistent_layer_bytes": total_bytes,
        "scale_dtype": str(scale_dtype),
        "max_abs_decode_error": max(r["max_abs_decode_error"] for r in reports.values()),
        "mean_abs_decode_error_weighted": sum(
            r["mean_abs_decode_error"] * r["elements"] for r in reports.values()
        ) / sum(r["elements"] for r in reports.values()),
        "exact_fraction": sum(r["exact_bf16_elements"] for r in reports.values())
        / sum(r["elements"] for r in reports.values()),
    }
    cache = {"schema": SCHEMA, "version": VERSION, "layers": layer_states}
    return cache, {"aggregate": aggregate, "layers": reports}


def save_packed_cache(
    cache: dict[str, Any],
    report: dict[str, Any],
    path: Path,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(cache)
    payload["provenance"] = provenance
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "cache": str(path),
        "cache_sha256": sha256_file(path),
        "cache_size": path.stat().st_size,
        "provenance": provenance,
        **report["aggregate"],
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_path = path.with_suffix(path.suffix + ".packing.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return manifest


def load_packed_cache(
    transformer: torch.nn.Module,
    path: Path,
    *,
    require_cuda: bool = True,
    expected_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise ValueError("unsupported FLUX packed INT4 cache schema")
    provenance = payload.get("provenance", {})
    for key, value in (expected_provenance or {}).items():
        if provenance.get(key) != value:
            raise ValueError(
                f"packed-cache provenance mismatch for {key}: "
                f"{provenance.get(key)!r} != {value!r}"
            )
    qlayers = find_qlayers(transformer, layers=[ActQuantWrapper])
    states = payload["layers"]
    if set(states) != set(qlayers):
        missing = sorted(set(qlayers) - set(states))[:5]
        extra = sorted(set(states) - set(qlayers))[:5]
        raise ValueError(f"packed-cache layer coverage mismatch: missing={missing}, extra={extra}")
    total = 0
    for name, layer in qlayers.items():
        packed = _install_state(layer, states[name], require_cuda=require_cuda)
        total += packed.persistent_bytes
    return {
        "active_layers": len(qlayers),
        "persistent_layer_bytes": total,
        "cache_sha256": sha256_file(path),
        "cache_size": path.stat().st_size,
        "provenance": provenance,
    }


def _real_int4_forward(self: ActQuantWrapper, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(self.module, PackedSplitInt4Linear):
        raise RuntimeError("real-INT4 forward reached a non-packed active wrapper")
    if not getattr(self, "_unrot_fused", False) and (
        self.rotation is not None or self.rotation_per_head is not None
    ):
        raise RuntimeError("rotated real-INT4 weight was not fused into its basis")

    if self.rotation is not None:
        shape = x.shape
        x_rot = (x.reshape(-1, shape[-1]) @ self.rotation).reshape(shape)
        high = int(self.quantizer.high_bits_length)
        x_low = x_rot[..., :-high] if high else x_rot
        x_high = x_rot[..., -high:] if high else None
        return self.module(x_low, x_high)

    if self.rotation_per_head is not None:
        shape = x.shape[:-1]
        heads, head_dim = int(self.num_heads), int(self.head_dim)
        x_heads = x.reshape(*shape, heads, head_dim)
        x_rot = torch.einsum("...hd,hde->...he", x_heads, self.rotation_per_head)
        high = int(self.quantizer.high_bits_length)
        low_dim = head_dim - high
        x_low = x_rot[..., :low_dim].reshape(*shape, heads * low_dim)
        x_high = x_rot[..., low_dim:].reshape(*shape, heads * high) if high else None
        return self.module(x_low, x_high)

    if getattr(self, "use_hadamard", False) or getattr(self, "perm_idx", None) is not None:
        raise RuntimeError("FLUX real-INT4 audit does not silently support alternate rotations")
    return self.module(x, None)


def patch_real_int4_forward() -> None:
    ActQuantWrapper.forward = _real_int4_forward
    print("Patched ActQuantWrapper.forward for packed integer low GEMM + BF16 protected tail.")
