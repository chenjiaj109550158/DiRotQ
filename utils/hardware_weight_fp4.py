"""Packing-valid hardware-faithful fixed E2M1/E0M3 weight GPTQ.

The runtime remains reconstructed-weight fake quantization.  Unlike the
legacy DiRotQ weight path, each transformed low-weight matrix uses one FP32
global scale, and every stored-row 1x16 group uses a frozen E4M3 block scale.
The two formats therefore differ only in their payload codebook and natural
maximum (6 for E2M1, 7 for E0M3).
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .e0joint_gptq import extract_fused_low_weight, sha256_file
from .quant_utils import ActQuantWrapper, _rotate_and_split_W, find_qlayers, round_to_nf4_codebook
from .weight_mixfp4 import (
    REQUIRED_ACTIVE_LAYERS,
    WEIGHT_GROUP_SIZE,
    _active_sana_layers,
    _tile_preserving_permutation,
    hessian_trace_loss,
    metadata_path as hessian_metadata_path,
    skip_layer_hash,
)


OBJECTIVE_VERSION = "e0a-hardware-fixed-weight-gptq-v1"
FORMATS = ("hardware-fixed-e2", "hardware-fixed-e0")
GLOBAL_MAX = 2688.0
E4M3_MAX = 448.0
E2_VALUES = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
             0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E0_VALUES = (-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
             0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
SCALE_SEMANTICS = (
    "layer-low-only-fp32-amax-over-2688;stored-row-k-g16;"
    "frozen-e4m3fn-amax-over-format-max;zero-safe"
)


def _format_short(fmt: str) -> str:
    if fmt == "hardware-fixed-e2":
        return "e2"
    if fmt == "hardware-fixed-e0":
        return "e0"
    raise ValueError(f"unsupported hardware weight format {fmt!r}")


def _codebook(fmt: str, *, device=None) -> torch.Tensor:
    values = E2_VALUES if _format_short(fmt) == "e2" else E0_VALUES
    return torch.tensor(values, dtype=torch.float32, device=device)


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def packing_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".packing.pt")


def metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".metadata.json")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hardware_global_scale(source: torch.Tensor) -> torch.Tensor:
    """One FP32 scale over the actual transformed low-weight region only."""
    if source.ndim != 2 or not source.is_floating_point():
        raise ValueError("weight source must be floating stored [N,K]")
    if not torch.isfinite(source).all():
        raise ValueError("non-finite transformed low weight")
    amax = source.float().abs().amax()
    return torch.where(amax == 0, torch.ones_like(amax), amax / GLOBAL_MAX)


def frozen_block_scales(
    source: torch.Tensor, fmt: str, global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 scales and pre-rounding FP32 scales for stored-row K groups."""
    short = _format_short(fmt)
    alpha = hardware_global_scale(source) if global_scale is None else global_scale.float()
    n, k = source.shape
    pad = (-k) % WEIGHT_GROUP_SIZE
    scaled = source.float() / alpha
    padded = torch.nn.functional.pad(scaled, (0, pad)) if pad else scaled
    blocks = padded.reshape(n, -1, WEIGHT_GROUP_SIZE)
    maximum = 6.0 if short == "e2" else 7.0
    raw = blocks.abs().amax(dim=-1) / maximum
    zero = raw == 0
    # abs() canonicalizes a possible negative zero after float8 round-trip.
    rounded = raw.to(torch.float8_e4m3fn).float().abs()
    if not torch.isfinite(rounded).all():
        raise OverflowError(f"{fmt}: E4M3 block-scale overflow")
    # Preserve a legal, finite divisor when a nonzero raw scale underflows.
    rounded = torch.where(
        (~zero) & (rounded == 0), torch.full_like(rounded, 2.0 ** -9), rounded
    )
    rounded = torch.where(zero, torch.ones_like(rounded), rounded)
    limit = 448.0 if short == "e2" else 384.0
    if float(rounded.max()) > limit:
        raise AssertionError(f"{fmt}: block scale exceeds {limit}")
    return rounded, raw


def _round_payload(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if _format_short(fmt) == "e2":
        return round_to_nf4_codebook(x)
    levels = torch.arange(0, 8, dtype=torch.float32, device=x.device)
    indices = torch.bucketize(x.abs().contiguous(), (levels[:-1] + levels[1:]) * .5)
    return torch.sign(x) * levels[indices]


def quantize_with_frozen_scales(
    source: torch.Tensor,
    fmt: str,
    global_scale: torch.Tensor,
    block_scales: torch.Tensor,
) -> torch.Tensor:
    n, k = source.shape
    pad = (-k) % WEIGHT_GROUP_SIZE
    padded = torch.nn.functional.pad(source.float(), (0, pad)) if pad else source.float()
    blocks = padded.reshape(n, -1, WEIGHT_GROUP_SIZE)
    alpha = global_scale.float()
    scales = block_scales.float()
    effective = alpha * scales
    codes = _round_payload(blocks / effective.unsqueeze(-1), fmt)
    # Keep the reconstruction order identical to the packing decoder.
    return (codes * scales.unsqueeze(-1) * alpha).reshape(n, -1)[:, :k]


def _quantize_column(
    column: torch.Tensor,
    original_column: int,
    fmt: str,
    global_scale: torch.Tensor,
    block_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_scale = block_scales[:, original_column // WEIGHT_GROUP_SIZE]
    scale = global_scale * block_scale
    normalized = column / scale
    code = _round_payload(normalized, fmt)
    return code * block_scale * global_scale, normalized


@torch.no_grad()
def gptq_quantize_hardware_fixed(
    source: torch.Tensor,
    hessian: torch.Tensor,
    fmt: str,
    *,
    damp_pct: float = .01,
    num_inv_tries: int = 8,
) -> tuple[torch.Tensor | None, dict, dict]:
    """GPTQ with source-derived global/E4M3 scales frozen before processing."""
    _format_short(fmt)
    if source.ndim != 2 or hessian.shape != (source.shape[1], source.shape[1]):
        raise ValueError("source/Hessian shape mismatch")
    if source.device.type != "cuda" or hessian.device.type != "cuda":
        raise RuntimeError("hardware weight GPTQ forbids silent CPU fallback")
    if not torch.isfinite(source).all() or not torch.isfinite(hessian).all():
        raise ValueError("non-finite hardware weight GPTQ input")

    source = source.float()
    hessian = hessian.float().clone()
    n, k = source.shape
    alpha = hardware_global_scale(source)
    scales, raw_scales = frozen_block_scales(source, fmt, alpha)
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    source_proc = source.clone()
    source_proc[:, dead] = 0
    permutation, tile_lengths = _tile_preserving_permutation(hessian)
    inverse = torch.argsort(permutation)
    source_proc = source_proc[:, permutation]
    hessian = hessian[permutation][:, permutation]
    diagonal = hessian.diagonal()
    base_damping = damp_pct * diagonal.mean()
    diagonal.add_(base_damping)

    result = None
    failure = None
    attempts = 0
    saturation_count = payload_count = 0
    maximum = 6.0 if _format_short(fmt) == "e2" else 7.0
    for attempt in range(1, num_inv_tries + 1):
        attempts = attempt
        try:
            chol = torch.linalg.cholesky(hessian)
            h_inv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
        except RuntimeError as exc:
            failure = f"Cholesky failure: {exc}"
            diagonal.add_(max(float(base_damping), 1e-8) * (10 ** (attempt - 1)))
            continue
        inv_diag = h_inv.diagonal()
        if inv_diag.min() < 1e-4 * inv_diag.mean():
            failure = "near-zero inverse-Hessian diagonal"
            diagonal.add_(max(float(base_damping), 1e-8) * (10 ** attempt))
            continue

        work = source_proc.clone()
        quantized = torch.zeros_like(work)
        saturation_count = payload_count = 0
        cursor = 0
        for span_length in tile_lengths:
            end = cursor + span_length
            for offset in range(cursor, end):
                original_column = int(permutation[offset])
                q_column, normalized = _quantize_column(
                    work[:, offset], original_column, fmt, alpha, scales
                )
                saturation_count += int((normalized.abs() > maximum).sum().item())
                payload_count += normalized.numel()
                error = (work[:, offset] - q_column) / h_inv[offset, offset]
                quantized[:, offset] = q_column
                if offset + 1 < end:
                    work[:, offset + 1:end].sub_(
                        error.unsqueeze(1) * h_inv[offset, offset + 1:end].unsqueeze(0)
                    )
                if end < k:
                    work[:, end:].sub_(
                        error.unsqueeze(1) * h_inv[offset, end:].unsqueeze(0)
                    )
            cursor = end
        if not torch.isfinite(quantized).all():
            failure = "non-finite GPTQ output"
            diagonal.add_(max(float(base_damping), 1e-8) * (10 ** attempt))
            continue
        result = quantized[:, inverse]
        break

    scale_nonzero = raw_scales > 0
    relative = torch.zeros_like(raw_scales)
    relative[scale_nonzero] = (
        (scales[scale_nonzero] - raw_scales[scale_nonzero]).abs()
        / raw_scales[scale_nonzero]
    )
    stats = {
        "format": fmt,
        "gptq_status": "gptq" if result is not None else "failed",
        "attempts": attempts,
        "failure": failure if result is None else None,
        "global_scale": float(alpha),
        "block_scale_min": float(scales.min()),
        "block_scale_max": float(scales.max()),
        "block_scale_mean": float(scales.mean()),
        "scale_rounding_relative_mean": float(relative[scale_nonzero].mean()) if scale_nonzero.any() else 0.0,
        "scale_rounding_relative_max": float(relative.max()),
        "saturation_count": saturation_count,
        "payload_count": payload_count,
        "saturation_rate": saturation_count / payload_count if payload_count else 0.0,
    }
    frozen = {
        "global_scale": alpha.detach(),
        "block_scales": scales.detach().to(torch.float8_e4m3fn),
        "raw_block_scales": raw_scales.detach(),
    }
    return result, stats, frozen


def payload_indices(weight: torch.Tensor, fmt: str, alpha: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    pad = (-k) % WEIGHT_GROUP_SIZE
    padded = torch.nn.functional.pad(weight.float(), (0, pad)) if pad else weight.float()
    effective = alpha.float() * scales.float()
    normalized = padded.reshape(n, -1, WEIGHT_GROUP_SIZE) / effective.unsqueeze(-1)
    codes = _round_payload(normalized, fmt)
    book = _codebook(fmt, device=weight.device)
    distance = (codes.unsqueeze(-1) - book).abs()
    return distance.argmin(dim=-1).to(torch.uint8).reshape(n, -1)


def pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    if indices.dtype != torch.uint8 or indices.ndim != 2:
        raise ValueError("payload indices must be uint8 [N,K]")
    if int(indices.max()) > 15:
        raise ValueError("4-bit payload index exceeds 15")
    if indices.shape[1] % 2:
        indices = torch.nn.functional.pad(indices, (0, 1))
    return indices[:, 0::2] | (indices[:, 1::2] << 4)


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed payload must be uint8 [N,K/2]")
    output = torch.empty(
        packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8, device=packed.device
    )
    output[:, 0::2] = packed & 0x0F
    output[:, 1::2] = packed >> 4
    return output[:, :count]


def decode_packing_record(record: dict, *, dtype=torch.float32, device="cpu") -> torch.Tensor:
    n, k = record["stored_shape"]
    group = int(record["group_size"])
    if group != WEIGHT_GROUP_SIZE:
        raise RuntimeError(f"packing record group size is {group}, expected 16")
    padded_k = math.ceil(k / group) * group
    indices = unpack_nibbles(record["packed_payload"].to(device), padded_k)
    book = _codebook(record["format"], device=device)
    codes = book[indices.long()].reshape(n, -1, group)
    scales = record["block_scales"].to(device).float()
    alpha = record["global_scale"].to(device).float()
    decoded = (codes * scales.unsqueeze(-1) * alpha).reshape(n, -1)[:, :k]
    return decoded.to(dtype=dtype)


def make_packing_record(
    quantized: torch.Tensor,
    fmt: str,
    frozen: dict,
    *,
    high_branch_hash: str,
) -> dict:
    indices = payload_indices(
        quantized, fmt, frozen["global_scale"], frozen["block_scales"]
    )
    packed = pack_nibbles(indices).cpu()
    record = {
        "format": fmt,
        "global_scale": frozen["global_scale"].float().cpu(),
        "block_scales": frozen["block_scales"].cpu(),
        "packed_payload": packed,
        "logical_shape": [quantized.shape[1], quantized.shape[0]],
        "stored_shape": list(quantized.shape),
        "group_size": WEIGHT_GROUP_SIZE,
        "high_branch_hash": high_branch_hash,
    }
    decoded = decode_packing_record(record, dtype=torch.float32)
    record["reconstructed_low_hash_fp32"] = tensor_sha256(decoded)
    return record


def _base_cache_state(transformer) -> dict:
    transient = (".quantizer.scale", ".quantizer.zero")
    return {
        key: value.detach().cpu()
        for key, value in transformer.state_dict().items()
        if not key.endswith(transient)
    }


def expected_metadata(
    *, model: str, fmt: str, calibration_count: int, damp_pct: float,
    basis_path: Path, rotation_path: Path, hessian_cache: Path,
    skip_layers: list[str],
) -> dict:
    _format_short(fmt)
    return {
        "model": model,
        "objective_version": OBJECTIVE_VERSION,
        "activation_hessian": "hardware-e0m3-gscale2688-per-calibration-chunk",
        "weight_format": fmt,
        "weight_group_size": WEIGHT_GROUP_SIZE,
        "weight_scale_semantics": SCALE_SEMANTICS,
        "global_scale_denominator": GLOBAL_MAX,
        "residual_rotation": "random",
        "calibration_count": calibration_count,
        "damp_pct": damp_pct,
        "skip_layer_hash": skip_layer_hash(skip_layers),
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
        "hessian_cache_sha256": sha256_file(hessian_cache),
        "active_layers": REQUIRED_ACTIVE_LAYERS,
    }


def validate_metadata(cache_path: Path, expected: dict) -> dict:
    path = metadata_path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"missing hardware weight metadata: {path}")
    metadata = json.loads(path.read_text())
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"hardware weight metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    sidecar = packing_path(cache_path)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing hardware weight packing sidecar: {sidecar}")
    if sha256_file(cache_path) != metadata.get("cache_sha256"):
        raise RuntimeError("hardware weight reconstructed cache SHA-256 mismatch")
    if sha256_file(sidecar) != metadata.get("packing_sha256"):
        raise RuntimeError("hardware weight packing sidecar SHA-256 mismatch")
    return metadata


def validate_runtime_state(
    transformer, state: dict, cache_path: Path, fmt: str,
) -> dict:
    """Decode every packed low weight and compare to the runtime BF16 cache."""
    layers = _active_sana_layers(transformer)
    sidecar = torch.load(packing_path(cache_path), map_location="cpu", weights_only=False)
    if sidecar.get("format") != fmt or set(sidecar.get("layers", {})) != set(layers):
        raise RuntimeError("hardware weight packing coverage/format mismatch")
    verified = 0
    for name, layer in layers.items():
        key = f"{name}.module.weight"
        if key not in state:
            raise RuntimeError(f"hardware weight cache missing {key}")
        low, high = extract_fused_low_weight(layer, state[key])
        record = sidecar["layers"][name]
        decoded = decode_packing_record(record, dtype=state[key].dtype)
        if not torch.equal(decoded, low.to(decoded.dtype)):
            max_diff = float((decoded.float() - low.float()).abs().max())
            raise RuntimeError(f"{name}: packed/reconstructed low mismatch max={max_diff}")
        high_hash = tensor_sha256(high) if high is not None else hashlib.sha256(b"").hexdigest()
        if high_hash != record["high_branch_hash"]:
            raise RuntimeError(f"{name}: high branch hash mismatch")
        if tensor_sha256(low) != record["reconstructed_low_hash"]:
            raise RuntimeError(f"{name}: reconstructed low hash mismatch")
        verified += 1
    return {"verified_layers": verified, "format": fmt, "fallback": False}


@torch.no_grad()
def build_hardware_fixed_caches(
    transformer,
    *,
    hessian_cache: Path,
    cache_paths: dict[str, Path],
    legacy_cache_paths: dict[str, Path],
    report_dir: Path,
    basis_path: Path,
    rotation_path: Path,
    skip_layers: list[str],
    num_calib_files: int = 5120,
    damp_pct: float = .01,
    device: str = "cuda",
) -> dict:
    """Reuse H_Z and build isolated packing-valid fixed E2/E0 GPTQ caches."""
    if set(cache_paths) != set(FORMATS):
        raise ValueError(f"cache paths must cover {FORMATS}")
    if set(legacy_cache_paths) != set(FORMATS):
        raise ValueError(f"legacy cache paths must cover {FORMATS}")
    if not hessian_cache.exists() or not hessian_metadata_path(hessian_cache).exists():
        raise FileNotFoundError("existing E0 activation Hessian and metadata are required")
    for path in (*cache_paths.values(),):
        if path.exists() or packing_path(path).exists() or metadata_path(path).exists():
            raise FileExistsError(f"refusing to overwrite hardware weight cache: {path}")
        if "hardware-fixed" not in path.name:
            raise ValueError(f"unsafe hardware weight cache path: {path}")
    for path in legacy_cache_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"missing legacy diagnostic cache: {path}")

    layers = _active_sana_layers(transformer)
    hmeta = json.loads(hessian_metadata_path(hessian_cache).read_text())
    if hmeta.get("cache_sha256") != sha256_file(hessian_cache):
        raise RuntimeError("H_Z SHA-256 does not match its provenance metadata")
    hessian_expected = {
        "model": "sana-1.6b",
        "cache_kind": "e0-activation-hessian",
        "activation_hessian": "hardware-e0m3-gscale2688-per-calibration-chunk",
        "active_layers": REQUIRED_ACTIVE_LAYERS,
        "weight_group_size": WEIGHT_GROUP_SIZE,
        "residual_rotation": "random",
        "calibration_count": num_calib_files,
        "damp_pct": damp_pct,
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
        "skip_layer_hash": skip_layer_hash(skip_layers),
    }
    for key, value in hessian_expected.items():
        if hmeta.get(key) != value:
            raise RuntimeError(
                f"H_Z provenance mismatch for {key}: expected {value!r}, "
                f"got {hmeta.get(key)!r}"
            )
    hessians = torch.load(hessian_cache, map_location="cpu", weights_only=False)
    if set(hessians) != set(layers):
        raise RuntimeError("H_Z layer coverage does not match SANA active layers")
    legacy_states = {
        fmt: torch.load(path, map_location="cpu", weights_only=False)
        for fmt, path in legacy_cache_paths.items()
    }
    base_state = _base_cache_state(transformer)
    weights = {fmt: {} for fmt in FORMATS}
    packings = {fmt: {} for fmt in FORMATS}
    aggregate = {fmt: 0.0 for fmt in FORMATS}
    aggregate_payload_fp32 = {fmt: 0.0 for fmt in FORMATS}
    occupancy = {fmt: [0] * 15 for fmt in FORMATS}
    rows = []
    failures = []
    started = time.perf_counter()
    peak_bytes = 0

    for name, layer in tqdm(layers.items(), desc="hardware E2/E0 weight GPTQ"):
        original_low, original_high, stitch, _ = _rotate_and_split_W(
            layer, layer.module.weight.data
        )
        high_hash = (
            tensor_sha256(
                original_high.to(layer.module.weight.dtype).float().cpu()
            )
            if original_high is not None else hashlib.sha256(b"").hexdigest()
        )
        source = original_low.to(device=device, dtype=torch.float32)
        hessian = hessians[name].to(device=device, dtype=torch.float32)
        for fmt in FORMATS:
            quantized, stats, frozen = gptq_quantize_hardware_fixed(
                source, hessian, fmt, damp_pct=damp_pct
            )
            if quantized is None:
                failures.append({"layer": name, "format": fmt, "reason": stats["failure"]})
                continue
            payload_loss = float(hessian_trace_loss(source, quantized, hessian).item())
            record = make_packing_record(
                quantized, fmt, frozen, high_branch_hash=high_hash
            )
            decoded_fp32 = decode_packing_record(record, dtype=torch.float32)
            fused = stitch(decoded_fp32.to(device)).to(layer.module.weight.dtype).cpu()
            final_low, final_high = extract_fused_low_weight(layer, fused)
            runtime_low_fp32 = decoded_fp32.to(layer.module.weight.dtype).float().cpu()
            if not torch.equal(runtime_low_fp32, final_low):
                raise RuntimeError(f"{name}/{fmt}: BF16 reconstructed low mismatch")
            final_high_hash = (
                tensor_sha256(final_high) if final_high is not None else hashlib.sha256(b"").hexdigest()
            )
            if final_high_hash != high_hash:
                raise RuntimeError(f"{name}/{fmt}: high branch changed")
            # Primary loss reflects the BF16 reconstructed tensor actually loaded
            # by the fake-quant runtime; preserve the pre-cast payload loss too.
            loss = float(
                hessian_trace_loss(source, final_low.to(device).float(), hessian).item()
            )
            aggregate[fmt] += loss
            aggregate_payload_fp32[fmt] += payload_loss
            record["reconstructed_low_hash"] = tensor_sha256(final_low)
            weights[fmt][name] = fused
            packings[fmt][name] = record
            indices = unpack_nibbles(record["packed_payload"], source.shape[1])
            counts = torch.bincount(indices.flatten().long(), minlength=15).tolist()
            occupancy[fmt] = [a + int(b) for a, b in zip(occupancy[fmt], counts)]

            legacy_key = f"{name}.module.weight"
            legacy_low, legacy_high = extract_fused_low_weight(layer, legacy_states[fmt][legacy_key])
            legacy_high_hash = (
                tensor_sha256(legacy_high)
                if legacy_high is not None else hashlib.sha256(b"").hexdigest()
            )
            if legacy_high_hash != high_hash:
                raise RuntimeError(f"{name}/{fmt}: legacy high branch mismatch")
            delta = final_low.float() - legacy_low.float()
            rounded_rtn = quantize_with_frozen_scales(
                source, fmt, frozen["global_scale"], frozen["block_scales"].float()
            )
            raw_safe = torch.where(
                frozen["raw_block_scales"] == 0,
                torch.ones_like(frozen["raw_block_scales"]),
                frozen["raw_block_scales"],
            )
            exact_rtn = quantize_with_frozen_scales(
                source, fmt, frozen["global_scale"], raw_safe
            )
            rounded_sse = float((source - rounded_rtn).double().square().sum())
            exact_sse = float((source - exact_rtn).double().square().sum())
            rows.append({
                "layer": name, "format": fmt, "n": source.shape[0], "k": source.shape[1],
                "calibration_loss": loss, "gptq_status": stats["gptq_status"],
                "payload_fp32_calibration_loss": payload_loss,
                "attempts": stats["attempts"], "fallback_reason": stats["failure"] or "",
                "global_scale": stats["global_scale"],
                "block_scale_min": stats["block_scale_min"],
                "block_scale_max": stats["block_scale_max"],
                "block_scale_mean": stats["block_scale_mean"],
                "scale_rounding_relative_mean": stats["scale_rounding_relative_mean"],
                "scale_rounding_relative_max": stats["scale_rounding_relative_max"],
                "rtn_exact_scale_sse": exact_sse, "rtn_rounded_scale_sse": rounded_sse,
                "rounding_sse_fraction": ((rounded_sse - exact_sse) / rounded_sse if rounded_sse else 0.0),
                "saturation_count": stats["saturation_count"],
                "payload_count": stats["payload_count"], "saturation_rate": stats["saturation_rate"],
                "legacy_hw_mse": float(delta.double().square().mean()),
                "legacy_hw_max_abs": float(delta.abs().max()),
                "legacy_hw_relative_l2": float(delta.double().norm() / legacy_low.double().norm().clamp_min(1e-30)),
            })
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
        del hessians[name], source, hessian
        torch.cuda.empty_cache()

    if failures or any(len(weights[fmt]) != REQUIRED_ACTIVE_LAYERS for fmt in FORMATS):
        _write_json(report_dir / "hardware_weight_failures.json", {"failures": failures})
        raise RuntimeError("hardware weight GPTQ incomplete; no cache saved")

    common = {
        "model": "sana-1.6b", "objective_version": OBJECTIVE_VERSION,
        "activation_hessian": "hardware-e0m3-gscale2688-per-calibration-chunk",
        "weight_group_size": WEIGHT_GROUP_SIZE,
        "weight_scale_semantics": SCALE_SEMANTICS,
        "global_scale_denominator": GLOBAL_MAX,
        "residual_rotation": "random", "calibration_count": num_calib_files,
        "damp_pct": damp_pct, "skip_layer_hash": skip_layer_hash(skip_layers),
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
        "hessian_cache": str(hessian_cache),
        "hessian_cache_sha256": sha256_file(hessian_cache),
        "hessian_metadata_sha256": sha256_file(hessian_metadata_path(hessian_cache)),
        "active_layers": REQUIRED_ACTIVE_LAYERS,
        "gptq_layers": REQUIRED_ACTIVE_LAYERS,
        "rtn_fallbacks": [], "high_branch_unchanged_layers": REQUIRED_ACTIVE_LAYERS,
    }
    cache_records = {}
    for fmt in FORMATS:
        state = dict(base_state)
        for name, fused in weights[fmt].items():
            state[f"{name}.weight"] = fused
            state[f"{name}.module.weight"] = fused
        sidecar = {"format": fmt, "metadata": common, "layers": packings[fmt]}
        cache_path = cache_paths[fmt]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, cache_path)
        torch.save(sidecar, packing_path(cache_path))
        metadata = {
            **common, "weight_format": fmt, "cache_path": str(cache_path),
            "packing_path": str(packing_path(cache_path)),
            "cache_sha256": sha256_file(cache_path),
            "packing_sha256": sha256_file(packing_path(cache_path)),
            "aggregate_calibration_loss": aggregate[fmt],
            "payload_occupancy": {
                str(value): occupancy[fmt][i]
                for i, value in enumerate(E2_VALUES if fmt.endswith("e2") else E0_VALUES)
            },
        }
        _write_json(metadata_path(cache_path), metadata)
        cache_records[fmt] = metadata
        del state

    elapsed = time.perf_counter() - started
    summary = {
        **common, "quantization_seconds": elapsed,
        "peak_gpu_memory_gib": peak_bytes / 1024**3,
        "aggregate_losses": aggregate,
        "aggregate_payload_fp32_losses": aggregate_payload_fp32,
        "hardware_e0_relative_loss_change_vs_e2": (
            (aggregate["hardware-fixed-e0"] - aggregate["hardware-fixed-e2"])
            / aggregate["hardware-fixed-e2"]
        ),
        "scale_statistics": {
            fmt: {
                "global_scale_min": min(r["global_scale"] for r in rows if r["format"] == fmt),
                "global_scale_max": max(r["global_scale"] for r in rows if r["format"] == fmt),
                "block_scale_min": min(r["block_scale_min"] for r in rows if r["format"] == fmt),
                "block_scale_max": max(r["block_scale_max"] for r in rows if r["format"] == fmt),
                "mean_layer_saturation_rate": sum(
                    r["saturation_rate"] for r in rows if r["format"] == fmt
                ) / REQUIRED_ACTIVE_LAYERS,
                "mean_layer_rounding_sse_fraction": sum(
                    r["rounding_sse_fraction"] for r in rows if r["format"] == fmt
                ) / REQUIRED_ACTIVE_LAYERS,
            }
            for fmt in FORMATS
        },
        "cache_records": cache_records,
        "legacy_cache_paths": {k: str(v) for k, v in legacy_cache_paths.items()},
        "legacy_cache_sha256": {k: sha256_file(v) for k, v in legacy_cache_paths.items()},
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(report_dir / "hardware_weight_layer_objectives.csv", rows)
    _write_json(report_dir / "hardware_weight_summary.json", summary)
    del base_state, weights, packings, legacy_states, hessians
    gc.collect()
    torch.cuda.empty_cache()
    return summary


__all__ = [
    "E0_VALUES", "E2_VALUES", "FORMATS", "GLOBAL_MAX", "OBJECTIVE_VERSION",
    "SCALE_SEMANTICS", "build_hardware_fixed_caches", "decode_packing_record",
    "expected_metadata", "frozen_block_scales", "gptq_quantize_hardware_fixed",
    "hardware_global_scale", "make_packing_record", "metadata_path",
    "pack_nibbles", "packing_path", "quantize_with_frozen_scales",
    "tensor_sha256", "unpack_nibbles", "validate_metadata", "validate_runtime_state",
]
