"""Fit-only FP8 high-weight RTN/GPTQ utilities.

The functions in this module operate on transformed high weights in PyTorch
stored layout ``[N, K_high]``.  Payloads remain real E4M3 bytes; reconstructed
BF16 tensors are produced only at the accuracy-runtime boundary.  No kernel,
QAT, STE, optimizer, or CPU fallback is hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Literal, Sequence

import torch

from .fp8_high_e0_low import (
    E4M3_MAX,
    MX_BLOCK_SIZE,
    MX_MAX_SCALE_EXP,
    MX_MIN_SCALE_EXP,
    MX_SCALE_BIAS,
    _e4m3_payload_from_scaled,
    decode_e4m3_bytes,
    decode_ue8m0,
)


E4_SCALE_MULTIPLIERS = (0.875, 1.0, 1.125)
MXRecipe = Literal["current", "nosat", "neighbor"]


@dataclass
class HighWeightQuantResult:
    payload: torch.Tensor
    reconstructed: torch.Tensor
    scales: torch.Tensor | None
    scale_bytes: torch.Tensor | None
    original_shape: tuple[int, int]
    padded_k: int
    saturation_count: int
    payload_mismatch_vs_rtn: int = 0
    gptq_status: str = "rtn"
    attempts: int = 0
    failure: str | None = None


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def hessian_weighted_error(
    source: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor,
) -> torch.Tensor:
    """Return ``tr((W-Q) H (W-Q)^T)`` with float64 reduction."""
    if source.ndim != 2 or source.shape != quantized.shape:
        raise ValueError("stored source/quantized weights must have identical [N,K] shape")
    if hessian.shape != (source.shape[1], source.shape[1]):
        raise ValueError("high Hessian shape does not match stored weight K")
    if not all(torch.isfinite(value).all() for value in (source, quantized, hessian)):
        raise ValueError("non-finite high-weight objective input")
    error = source.float() - quantized.float()
    return (error * (error @ hessian.float())).double().sum().clamp_min(0)


def _validate_source(source: torch.Tensor) -> None:
    if source.ndim != 2 or not source.is_floating_point():
        raise ValueError("high weight must be floating stored [N,K]")
    if not torch.isfinite(source).all():
        raise ValueError("high weight contains non-finite values")


def _e4_per_channel_payload(
    source: torch.Tensor, scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if scales.shape != (source.shape[0],):
        raise ValueError("plain E4M3 scales must be one FP32 value per output channel")
    if scales.dtype != torch.float32 or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("plain E4M3 per-channel scales must be finite positive FP32")
    payload, saturation = _e4m3_payload_from_scaled(source.float() / scales[:, None])
    decoded = decode_e4m3_bytes(payload) * scales[:, None]
    if not torch.isfinite(decoded).all():
        raise RuntimeError("plain E4M3 reconstruction is non-finite")
    return payload, decoded, saturation


@torch.no_grad()
def choose_e4_per_channel_scales(
    source: torch.Tensor,
    hessian: torch.Tensor | None = None,
    multipliers: Sequence[float] = E4_SCALE_MULTIPLIERS,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Freeze one scale per output channel using only the fit objective.

    With one multiplier this is ordinary absmax RTN.  With multiple fixed
    candidates, strict row-wise Hessian loss selects a scale; ties prefer the
    unmodified 1.0 multiplier and then the declared order.  This selection is
    performed before GPTQ and never reads Dev labels.
    """
    _validate_source(source)
    if not multipliers or any((not math.isfinite(x) or x <= 0) for x in multipliers):
        raise ValueError("E4M3 scale multipliers must be finite and positive")
    if hessian is not None and hessian.shape != (source.shape[1], source.shape[1]):
        raise ValueError("scale-selection Hessian shape mismatch")
    amax = source.float().abs().amax(dim=1)
    base = torch.where(amax == 0, torch.ones_like(amax), amax / E4M3_MAX).float()
    order = list(range(len(multipliers)))
    if 1.0 in multipliers:
        one = list(multipliers).index(1.0)
        order = [one, *[index for index in order if index != one]]
    losses = []
    payloads = []
    for multiplier in multipliers:
        scales = (base * float(multiplier)).float()
        payload, decoded, _ = _e4_per_channel_payload(source, scales)
        error = source.float() - decoded
        if hessian is None:
            loss = error.double().square().sum(dim=1)
        else:
            loss = (error * (error @ hessian.float())).double().sum(dim=1)
        losses.append(loss)
        payloads.append(payload)
    loss_matrix = torch.stack(losses)
    best = torch.full((source.shape[0],), order[0], dtype=torch.long, device=source.device)
    best_loss = loss_matrix[order[0]].clone()
    for index in order[1:]:
        better = loss_matrix[index] < best_loss
        best = torch.where(better, torch.full_like(best, index), best)
        best_loss = torch.where(better, loss_matrix[index], best_loss)
    selected_scales = base * torch.tensor(
        list(multipliers), device=source.device, dtype=torch.float32
    )[best]
    selected_payload = torch.gather(
        torch.stack(payloads), 0,
        best[None, :, None].expand(1, source.shape[0], source.shape[1]),
    ).squeeze(0)
    counts = {str(float(multiplier)): int((best == index).sum())
              for index, multiplier in enumerate(multipliers)}
    return selected_scales.float(), selected_payload, {
        "multipliers": [float(value) for value in multipliers],
        "selection_counts": counts,
        "selected_objective": float(best_loss.sum()),
        "absmax_objective": float(loss_matrix[list(multipliers).index(1.0)].sum())
        if 1.0 in multipliers else None,
    }


@torch.no_grad()
def quantize_e4_per_channel_rtn(
    source: torch.Tensor,
    *,
    scales: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
    multipliers: Sequence[float] = (1.0,),
) -> tuple[HighWeightQuantResult, dict]:
    _validate_source(source)
    selection = None
    if scales is None:
        scales, _, selection = choose_e4_per_channel_scales(
            source, hessian=hessian, multipliers=multipliers
        )
    payload, reconstructed, saturation = _e4_per_channel_payload(source, scales)
    return HighWeightQuantResult(
        payload=payload,
        reconstructed=reconstructed.to(source.dtype),
        scales=scales.float(),
        scale_bytes=None,
        original_shape=tuple(source.shape),
        padded_k=source.shape[1],
        saturation_count=saturation,
    ), (selection or {"multipliers": None, "selection_counts": None})


def _mx_exponents(block_amax: torch.Tensor, recipe: MXRecipe) -> torch.Tensor:
    nonzero = block_amax > 0
    safe = torch.where(nonzero, block_amax.float(), torch.ones_like(block_amax).float())
    if recipe == "current":
        exponent = torch.floor(torch.log2(safe)) - 8.0
    elif recipe in {"nosat", "neighbor"}:
        exponent = torch.ceil(torch.log2(safe / E4M3_MAX))
    else:
        raise ValueError(f"unsupported MXFP8 recipe {recipe!r}")
    exponent = exponent.clamp(MX_MIN_SCALE_EXP, MX_MAX_SCALE_EXP).to(torch.int16)
    return torch.where(nonzero, exponent, torch.zeros_like(exponent))


def _mx_blocks(source: torch.Tensor) -> tuple[torch.Tensor, int]:
    k = source.shape[1]
    padded_k = math.ceil(k / MX_BLOCK_SIZE) * MX_BLOCK_SIZE
    padded = torch.nn.functional.pad(source.float(), (0, padded_k - k)) \
        if padded_k != k else source.float()
    return padded.reshape(source.shape[0], -1, MX_BLOCK_SIZE), padded_k


@torch.no_grad()
def choose_mx_scale_bytes(
    source: torch.Tensor, recipe: MXRecipe,
) -> tuple[torch.Tensor, dict]:
    """Freeze legal UE8M0 K32 scales from the original high weight.

    ``neighbor`` compares the no-saturation exponent and its two adjacent
    exponents by deterministic block reconstruction SSE.  This fit-only rule
    is frozen before GPTQ and deliberately does not use Dev labels.
    """
    _validate_source(source)
    blocks, _ = _mx_blocks(source)
    amax = blocks.abs().amax(dim=-1)
    base = _mx_exponents(amax, recipe)
    if recipe != "neighbor":
        return (base + MX_SCALE_BIAS).to(torch.uint8), {
            "recipe": recipe, "neighbor_selection_counts": None,
        }
    candidates = torch.stack([
        (base - 1).clamp(MX_MIN_SCALE_EXP, MX_MAX_SCALE_EXP),
        base,
        (base + 1).clamp(MX_MIN_SCALE_EXP, MX_MAX_SCALE_EXP),
    ])
    # Tie order is the ideal no-saturation exponent, then lower, then higher.
    order = (1, 0, 2)
    losses = []
    for index in range(3):
        scale = torch.pow(torch.tensor(2.0, device=source.device), candidates[index].float())
        payload, _ = _e4m3_payload_from_scaled(blocks / scale.unsqueeze(-1))
        decoded = decode_e4m3_bytes(payload) * scale.unsqueeze(-1)
        losses.append((blocks - decoded).double().square().sum(dim=-1))
    loss_matrix = torch.stack(losses)
    best = torch.full(base.shape, order[0], dtype=torch.long, device=source.device)
    best_loss = loss_matrix[order[0]].clone()
    for index in order[1:]:
        better = loss_matrix[index] < best_loss
        best = torch.where(better, torch.full_like(best, index), best)
        best_loss = torch.where(better, loss_matrix[index], best_loss)
    exponent = torch.gather(candidates, 0, best.unsqueeze(0)).squeeze(0)
    return (exponent + MX_SCALE_BIAS).to(torch.uint8), {
        "recipe": recipe,
        "neighbor_selection_counts": {
            "e-1": int((best == 0).sum()),
            "e": int((best == 1).sum()),
            "e+1": int((best == 2).sum()),
        },
        "selected_reconstruction_sse": float(best_loss.sum()),
    }


def _mx_payload(
    source: torch.Tensor, scale_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    blocks, padded_k = _mx_blocks(source)
    expected = (source.shape[0], padded_k // MX_BLOCK_SIZE)
    if scale_bytes.shape != expected or scale_bytes.dtype != torch.uint8:
        raise ValueError(f"MXFP8 scale bytes must be uint8 {expected}")
    scales = decode_ue8m0(scale_bytes)
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("MXFP8 scales must decode to finite positive powers of two")
    payload, saturation = _e4m3_payload_from_scaled(blocks / scales.unsqueeze(-1))
    decoded = decode_e4m3_bytes(payload) * scales.unsqueeze(-1)
    decoded = decoded.reshape(source.shape[0], padded_k)[:, :source.shape[1]]
    return payload.reshape(source.shape[0], padded_k), decoded, saturation, padded_k


@torch.no_grad()
def quantize_mx_weight_rtn(
    source: torch.Tensor, *, recipe: MXRecipe = "current",
    scale_bytes: torch.Tensor | None = None,
) -> tuple[HighWeightQuantResult, dict]:
    _validate_source(source)
    selection = None
    if scale_bytes is None:
        scale_bytes, selection = choose_mx_scale_bytes(source, recipe)
    payload, reconstructed, saturation, padded_k = _mx_payload(source, scale_bytes)
    return HighWeightQuantResult(
        payload=payload,
        reconstructed=reconstructed.to(source.dtype),
        scales=None,
        scale_bytes=scale_bytes,
        original_shape=tuple(source.shape),
        padded_k=padded_k,
        saturation_count=saturation,
    ), (selection or {"recipe": recipe})


def _prepare_gptq_hessian(
    source: torch.Tensor, hessian: torch.Tensor, damp_pct: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hessian.shape != (source.shape[1], source.shape[1]):
        raise ValueError("high Hessian shape mismatch")
    if not torch.isfinite(hessian).all():
        raise ValueError("high Hessian contains non-finite values")
    hessian = hessian.float().clone()
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1.0
    work = source.float().clone()
    work[:, dead] = 0.0
    permutation = torch.argsort(hessian.diagonal(), descending=True)
    inverse = torch.argsort(permutation)
    hessian = hessian[permutation][:, permutation]
    diagonal = hessian.diagonal()
    diagonal.add_(damp_pct * diagonal.mean())
    return work[:, permutation], hessian, permutation, inverse


@torch.no_grad()
def _sequential_gptq(
    source: torch.Tensor,
    hessian: torch.Tensor,
    quantize_column: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    damp_pct: float = .01,
    num_inv_tries: int = 8,
    require_cuda: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:
    _validate_source(source)
    if require_cuda and (source.device.type != "cuda" or hessian.device.type != "cuda"):
        raise RuntimeError("formal high-weight GPTQ forbids silent CPU fallback")
    work_initial, damped, permutation, inverse = _prepare_gptq_hessian(
        source, hessian, damp_pct
    )
    diagonal = damped.diagonal()
    base_damping = max(float(damp_pct * diagonal.mean()), 1e-8)
    failure = None
    attempts = 0
    for attempt in range(1, num_inv_tries + 1):
        attempts = attempt
        try:
            chol = torch.linalg.cholesky(damped)
            h_inv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
        except RuntimeError as error:
            failure = f"Cholesky failure: {error}"
            diagonal.add_(base_damping * (10 ** (attempt - 1)))
            continue
        inv_diag = h_inv.diagonal()
        if float(inv_diag.min()) < 1e-4 * float(inv_diag.mean()):
            failure = "near-zero inverse-Hessian diagonal"
            diagonal.add_(base_damping * (10 ** attempt))
            continue
        work = work_initial.clone()
        quantized = torch.zeros_like(work)
        payload = torch.zeros_like(work, dtype=torch.uint8)
        for offset in range(source.shape[1]):
            original_column = int(permutation[offset])
            q_column, code = quantize_column(work[:, offset], original_column)
            if q_column.shape != work[:, offset].shape or code.shape != q_column.shape:
                raise RuntimeError("GPTQ column quantizer returned an invalid shape")
            error = (work[:, offset] - q_column) / h_inv[offset, offset]
            quantized[:, offset] = q_column
            payload[:, offset] = code
            if offset + 1 < source.shape[1]:
                work[:, offset + 1:].sub_(
                    error[:, None] * h_inv[offset, offset + 1:][None, :]
                )
        if not torch.isfinite(quantized).all():
            failure = "non-finite GPTQ output"
            diagonal.add_(base_damping * (10 ** attempt))
            continue
        return quantized[:, inverse], payload[:, inverse], {
            "gptq_status": "gptq", "attempts": attempts, "failure": None,
            "saturation_count": 0,
            "column_order_sha256": tensor_sha256(permutation),
        }
    return None, None, {
        "gptq_status": "failed", "attempts": attempts, "failure": failure,
        "saturation_count": 0, "column_order_sha256": tensor_sha256(permutation),
    }


@torch.no_grad()
def quantize_e4_per_channel_gptq(
    source: torch.Tensor,
    hessian: torch.Tensor,
    scales: torch.Tensor,
    *,
    damp_pct: float = .01,
    num_inv_tries: int = 8,
    require_cuda: bool = True,
) -> HighWeightQuantResult:
    rtn, _ = quantize_e4_per_channel_rtn(source, scales=scales)

    def quantize_column(column: torch.Tensor, _original_column: int):
        code, _ = _e4m3_payload_from_scaled(column.float() / scales)
        return decode_e4m3_bytes(code) * scales, code

    quantized, payload, stats = _sequential_gptq(
        source, hessian, quantize_column, damp_pct=damp_pct,
        num_inv_tries=num_inv_tries, require_cuda=require_cuda,
    )
    if quantized is None or payload is None:
        return HighWeightQuantResult(
            payload=torch.empty(0, dtype=torch.uint8, device=source.device),
            reconstructed=torch.empty(0, dtype=source.dtype, device=source.device),
            scales=scales, scale_bytes=None, original_shape=tuple(source.shape),
            padded_k=source.shape[1], saturation_count=0,
            gptq_status="failed", attempts=stats["attempts"], failure=stats["failure"],
        )
    # Re-decode payload rather than trusting the transient quantized buffer.
    reconstructed = decode_e4m3_bytes(payload) * scales[:, None]
    saturation = int((source.float().abs() / scales[:, None] > E4M3_MAX).sum())
    return HighWeightQuantResult(
        payload=payload, reconstructed=reconstructed.to(source.dtype), scales=scales,
        scale_bytes=None, original_shape=tuple(source.shape), padded_k=source.shape[1],
        saturation_count=saturation,
        payload_mismatch_vs_rtn=int((payload != rtn.payload).sum()),
        gptq_status="gptq", attempts=stats["attempts"], failure=None,
    )


@torch.no_grad()
def quantize_mx_weight_gptq(
    source: torch.Tensor,
    hessian: torch.Tensor,
    scale_bytes: torch.Tensor,
    *,
    damp_pct: float = .01,
    num_inv_tries: int = 8,
    require_cuda: bool = True,
) -> HighWeightQuantResult:
    rtn, _ = quantize_mx_weight_rtn(source, scale_bytes=scale_bytes)
    scales = decode_ue8m0(scale_bytes)

    def quantize_column(column: torch.Tensor, original_column: int):
        scale = scales[:, original_column // MX_BLOCK_SIZE]
        code, _ = _e4m3_payload_from_scaled(column.float() / scale)
        return decode_e4m3_bytes(code) * scale, code

    quantized, payload_valid, stats = _sequential_gptq(
        source, hessian, quantize_column, damp_pct=damp_pct,
        num_inv_tries=num_inv_tries, require_cuda=require_cuda,
    )
    if quantized is None or payload_valid is None:
        return HighWeightQuantResult(
            payload=torch.empty(0, dtype=torch.uint8, device=source.device),
            reconstructed=torch.empty(0, dtype=source.dtype, device=source.device),
            scales=None, scale_bytes=scale_bytes, original_shape=tuple(source.shape),
            padded_k=rtn.padded_k, saturation_count=0,
            gptq_status="failed", attempts=stats["attempts"], failure=stats["failure"],
        )
    payload = torch.zeros(
        source.shape[0], rtn.padded_k, dtype=torch.uint8, device=source.device
    )
    payload[:, :source.shape[1]] = payload_valid
    reconstructed = decode_e4m3_bytes(payload_valid) * torch.gather(
        scales, 1,
        (torch.arange(source.shape[1], device=source.device) // MX_BLOCK_SIZE)
        .unsqueeze(0).expand(source.shape[0], -1),
    )
    saturation = int(sum(
        (source[:, start:start + MX_BLOCK_SIZE].float().abs()
         / scales[:, block:block + 1] > E4M3_MAX).sum().item()
        for block, start in enumerate(range(0, source.shape[1], MX_BLOCK_SIZE))
    ))
    return HighWeightQuantResult(
        payload=payload, reconstructed=reconstructed.to(source.dtype), scales=None,
        scale_bytes=scale_bytes, original_shape=tuple(source.shape), padded_k=rtn.padded_k,
        saturation_count=saturation,
        payload_mismatch_vs_rtn=int((payload != rtn.payload).sum()),
        gptq_status="gptq", attempts=stats["attempts"], failure=None,
    )


def make_high_weight_record(
    result: HighWeightQuantResult, *, fmt: str, recipe: str, metadata: dict,
) -> dict:
    if result.gptq_status == "failed":
        raise RuntimeError(f"refusing to serialize failed GPTQ result: {result.failure}")
    record = {
        "schema": "dirotq.fp8_high_weight.v1",
        "format": fmt,
        "recipe": recipe,
        "payload": result.payload.detach().cpu(),
        "scales": result.scales.detach().cpu() if result.scales is not None else None,
        "scale_bytes": (
            result.scale_bytes.detach().cpu() if result.scale_bytes is not None else None
        ),
        "stored_shape": list(result.original_shape),
        "padded_k": result.padded_k,
        "saturation_count": result.saturation_count,
        "payload_mismatch_vs_rtn": result.payload_mismatch_vs_rtn,
        "gptq_status": result.gptq_status,
        "gptq_attempts": result.attempts,
        "metadata": dict(metadata),
    }
    record["payload_sha256"] = tensor_sha256(record["payload"])
    scale_tensor = record["scales"] if record["scales"] is not None else record["scale_bytes"]
    record["scale_sha256"] = tensor_sha256(scale_tensor)
    return record


def decode_high_weight_record(
    record: dict, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if record.get("schema") != "dirotq.fp8_high_weight.v1":
        raise RuntimeError("unknown high-weight packing schema")
    n, k = map(int, record["stored_shape"])
    payload = record["payload"].to(device)
    if tensor_sha256(record["payload"]) != record["payload_sha256"]:
        raise RuntimeError("high-weight payload hash mismatch")
    if record["format"] == "e4m3-per-channel":
        scales = record["scales"].to(device).float()
        decoded = decode_e4m3_bytes(payload[:, :k]) * scales[:, None]
    elif record["format"] == "mxfp8-e4m3-k32":
        scale_bytes = record["scale_bytes"].to(device)
        scales = decode_ue8m0(scale_bytes)
        blocks = decode_e4m3_bytes(payload.reshape(n, -1, MX_BLOCK_SIZE))
        decoded = (blocks * scales.unsqueeze(-1)).reshape(n, -1)[:, :k]
    else:
        raise RuntimeError(f"unsupported high-weight format {record['format']!r}")
    if tuple(decoded.shape) != (n, k) or not torch.isfinite(decoded).all():
        raise RuntimeError("invalid decoded high-weight record")
    return decoded.to(dtype)


def serialized_high_record_bytes(record: dict) -> dict[str, int]:
    payload = int(record["payload"].numel() * record["payload"].element_size())
    scale_tensor = record["scales"] if record["scales"] is not None else record["scale_bytes"]
    scales = int(scale_tensor.numel() * scale_tensor.element_size())
    return {"payload": payload, "scales": scales, "total": payload + scales}
