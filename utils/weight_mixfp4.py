"""SANA-only fixed-E0 activation x static E2/E0 weight MixFP4 experiment.

This module is deliberately fake-quant only.  It preserves the repository's
existing weight scale semantics (FP32 per-output-row 1x16 ``amax/max_level``;
no tensor-global scale and no E4M3 scale rounding), and changes only the
4-bit payload codebook.  GPTQ candidates for a logical 64x8 weight tile start
from identical work state; the selected candidate alone is committed to the
subsequent error-propagation state.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from .e0joint_gptq import (
    extract_fused_low_weight,
    project_low_activation,
    quantize_e0_per_chunk,
    sha256_file,
)
from .quant_utils import (
    ActQuantWrapper,
    NF4_MAX,
    _quant_group_nvfp4,
    _rotate_and_split_W,
    find_qlayers,
    round_to_nf4_codebook,
)


OBJECTIVE_VERSION = "e0a-weight-mix-gptq-v1"
REQUIRED_ACTIVE_LAYERS = 120
WEIGHT_GROUP_SIZE = 16
WEIGHT_TILE_K = 64
WEIGHT_TILE_N = 8
E0_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
WEIGHT_SCALE_SEMANTICS = (
    "per-output-row-g16-fp32-amax-over-format-max-clamp1e-5;"
    "no-global-scale;no-e4m3-rounding"
)
PRODUCTION_MODES = ("fixed-e2", "fixed-e0", "tilemix")


def _active_sana_layers(transformer) -> dict[str, ActQuantWrapper]:
    layers = {
        name: layer
        for name, layer in find_qlayers(
            transformer, layers=[ActQuantWrapper]
        ).items()
        if layer.quantizer.bits < 16
    }
    if len(layers) != REQUIRED_ACTIVE_LAYERS:
        raise RuntimeError(
            f"weight Mix requires {REQUIRED_ACTIVE_LAYERS} active SANA layers, "
            f"found {len(layers)}"
        )
    for name, layer in layers.items():
        if layer.rotation is None and layer.rotation_per_head is None:
            raise RuntimeError(f"{name}: missing random residual rotation")
    return layers


def skip_layer_hash(skip_layers: list[str]) -> str:
    canonical = "\n".join(sorted(skip_layers))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


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


def expected_metadata(
    *, model: str, mode: str, calibration_count: int, damp_pct: float,
    basis_path: Path, rotation_path: Path, skip_layers: list[str],
) -> dict:
    if mode not in PRODUCTION_MODES:
        raise ValueError(f"unsupported production weight format {mode!r}")
    return {
        "model": model,
        "objective_version": OBJECTIVE_VERSION,
        "activation_hessian": "hardware-e0m3-gscale2688-per-calibration-chunk",
        "weight_format": mode,
        "weight_group_size": WEIGHT_GROUP_SIZE,
        "logical_weight_tile": [WEIGHT_TILE_K, WEIGHT_TILE_N],
        "stored_weight_tile": [WEIGHT_TILE_N, WEIGHT_TILE_K],
        "residual_rotation": "random",
        "calibration_count": calibration_count,
        "damp_pct": damp_pct,
        "skip_layer_hash": skip_layer_hash(skip_layers),
        "quantizer_scale_semantics": WEIGHT_SCALE_SEMANTICS,
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
    }


def validate_cache_metadata(cache_path: Path, expected: dict) -> dict:
    path = metadata_path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"missing weight-Mix cache metadata: {path}")
    metadata = json.loads(path.read_text())
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"weight-Mix metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    return metadata


def logical_tile_to_stored_slice(k_tile: int, n_tile: int):
    """Map logical ``[K,N]`` 64x8 tile to stored ``[N,K]`` 8x64 slices."""
    if k_tile < 0 or n_tile < 0:
        raise ValueError("tile indices must be non-negative")
    return (
        slice(n_tile * WEIGHT_TILE_N, (n_tile + 1) * WEIGHT_TILE_N),
        slice(k_tile * WEIGHT_TILE_K, (k_tile + 1) * WEIGHT_TILE_K),
    )


def _round_e0m3(x: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(E0_LEVELS, dtype=torch.float32, device=x.device)
    midpoints = (levels[:-1] + levels[1:]) * .5
    indices = torch.bucketize(x.abs().contiguous(), midpoints)
    return torch.sign(x) * levels[indices]


def _weight_scales(source: torch.Tensor, fmt: str) -> torch.Tensor:
    """Return fixed legacy weight scales ``[N, ceil(K/16)]`` in FP32."""
    if source.ndim != 2 or not source.is_floating_point():
        raise ValueError("weight source must be floating [N,K]")
    if fmt not in {"e2", "e0"}:
        raise ValueError(f"unsupported weight format {fmt!r}")
    source = source.float()
    n, k = source.shape
    pad = (-k) % WEIGHT_GROUP_SIZE
    padded = torch.nn.functional.pad(source, (0, pad)) if pad else source
    blocks = padded.reshape(n, -1, WEIGHT_GROUP_SIZE)
    maximum = NF4_MAX if fmt == "e2" else E0_LEVELS[-1]
    return blocks.abs().amax(dim=-1).clamp(min=1e-5) / maximum


def quantize_weight_blocks(source: torch.Tensor, fmt: str) -> torch.Tensor:
    """RTN reference primitive using the exact existing weight scale hierarchy."""
    source_fp32 = source.float()
    n, k = source_fp32.shape
    pad = (-k) % WEIGHT_GROUP_SIZE
    padded = torch.nn.functional.pad(source_fp32, (0, pad)) if pad else source_fp32
    blocks = padded.reshape(n, -1, WEIGHT_GROUP_SIZE)
    scales = _weight_scales(source_fp32, fmt).unsqueeze(-1)
    normalized = blocks / scales
    codes = round_to_nf4_codebook(normalized) if fmt == "e2" else _round_e0m3(normalized)
    return (codes * scales).reshape(n, -1)[:, :k]


def _quantize_column(
    column: torch.Tensor, original_column: int, scales: torch.Tensor, fmt: str
) -> torch.Tensor:
    scale = scales[:, original_column // WEIGHT_GROUP_SIZE]
    normalized = column / scale
    code = round_to_nf4_codebook(normalized) if fmt == "e2" else _round_e0m3(normalized)
    return code * scale


def _simulate_span(
    work: torch.Tensor,
    h_inv_span: torch.Tensor,
    original_columns: torch.Tensor,
    scales: torch.Tensor,
    fmt: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate one candidate from an isolated GPTQ work-state snapshot."""
    candidate = work.clone()
    quantized = torch.zeros_like(candidate)
    errors = torch.zeros_like(candidate)
    loss_per_row = torch.zeros(candidate.shape[0], device=candidate.device)
    for offset in range(candidate.shape[1]):
        column = candidate[:, offset]
        q_column = _quantize_column(
            column, int(original_columns[offset]), scales, fmt
        )
        diagonal = h_inv_span[offset, offset]
        error = (column - q_column) / diagonal
        quantized[:, offset] = q_column
        errors[:, offset] = error
        loss_per_row.add_((column - q_column).square() / diagonal.square())
        if offset + 1 < candidate.shape[1]:
            candidate[:, offset + 1:].sub_(
                error.unsqueeze(1) * h_inv_span[offset, offset + 1:].unsqueeze(0)
            )
    return quantized, errors, loss_per_row


def _tile_preserving_permutation(hessian: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Importance-order K tiles, preserving each logical 64-column tile as a unit."""
    diagonal = hessian.diagonal()
    k = hessian.shape[0]
    tiles = list(range(math.ceil(k / WEIGHT_TILE_K)))
    tiles.sort(
        key=lambda tile: float(
            diagonal[tile * WEIGHT_TILE_K:min((tile + 1) * WEIGHT_TILE_K, k)].sum()
        ),
        reverse=True,
    )
    columns = []
    lengths = []
    for tile in tiles:
        start, end = tile * WEIGHT_TILE_K, min((tile + 1) * WEIGHT_TILE_K, k)
        # Preserve 1x16 scale blocks as sub-units; importance-sort within each
        # block.  This permits the BlockMix ceiling to commit a legal block at
        # a time while TileMix still commits all four blocks together.
        block_ids = list(range(start // WEIGHT_GROUP_SIZE, math.ceil(end / WEIGHT_GROUP_SIZE)))
        block_ids.sort(
            key=lambda block: float(
                diagonal[block * WEIGHT_GROUP_SIZE:min((block + 1) * WEIGHT_GROUP_SIZE, end)].sum()
            ),
            reverse=True,
        )
        tile_columns = []
        for block in block_ids:
            b0 = max(start, block * WEIGHT_GROUP_SIZE)
            b1 = min(end, (block + 1) * WEIGHT_GROUP_SIZE)
            local = torch.arange(b0, b1, device=hessian.device)
            order = torch.argsort(diagonal[local], descending=True)
            tile_columns.extend(local[order].tolist())
        columns.extend(tile_columns)
        lengths.append(len(tile_columns))
    return torch.tensor(columns, device=hessian.device, dtype=torch.long), lengths


@torch.no_grad()
def gptq_quantize_weight_tiles(
    source: torch.Tensor,
    hessian: torch.Tensor,
    mode: str,
    damp_pct: float = .01,
    num_inv_tries: int = 8,
) -> tuple[torch.Tensor | None, dict]:
    """GPTQ with fixed, 64x8 TileMix, or 1x16 BlockMix weight formats.

    ``source`` is stored layout ``[N,K]``.  TileMix constructs E2 and E0
    simulations from the same cloned 64-column work state, sums the existing
    GPTQ incremental weighted loss over each valid 8-row tile, chooses E0 only
    on a strict improvement, and propagates only the chosen error.  BlockMix
    uses the same rule per stored 1x16 block and is stats-only.
    """
    if mode not in {"fixed-e2", "fixed-e0", "tilemix", "blockmix"}:
        raise ValueError(f"unsupported GPTQ weight mode {mode!r}")
    if source.ndim != 2 or hessian.shape != (source.shape[1], source.shape[1]):
        raise ValueError("source/Hessian shape mismatch")
    if source.device.type != "cuda" or hessian.device.type != "cuda":
        raise RuntimeError("weight Mix GPTQ forbids silent CPU fallback")
    if not torch.isfinite(source).all() or not torch.isfinite(hessian).all():
        raise ValueError("non-finite weight Mix input")

    source = source.float()
    hessian = hessian.float().clone()
    n, k = source.shape
    scales_e2 = _weight_scales(source, "e2")
    scales_e0 = _weight_scales(source, "e0")
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
    attempts = 0
    selected_e2 = selected_e0 = 0
    incremental_e2 = incremental_e0 = incremental_selected = 0.0
    failure = None
    for attempt in range(1, num_inv_tries + 1):
        attempts = attempt
        try:
            chol = torch.linalg.cholesky(hessian)
            h_inv = torch.linalg.cholesky(
                torch.cholesky_inverse(chol), upper=True
            )
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
        selected_e2 = selected_e0 = 0
        incremental_e2 = incremental_e0 = incremental_selected = 0.0
        cursor = 0
        failed = False
        for tile_length in tile_lengths:
            tile_end = cursor + tile_length
            if mode == "blockmix":
                spans = [(start, min(start + WEIGHT_GROUP_SIZE, tile_end))
                         for start in range(cursor, tile_end, WEIGHT_GROUP_SIZE)]
            else:
                spans = [(cursor, tile_end)]

            for span_start, span_end in spans:
                span = work[:, span_start:span_end].clone()
                h_span = h_inv[span_start:span_end, span_start:span_end]
                original_columns = permutation[span_start:span_end]
                q_e2 = err_e2 = loss_e2 = None
                q_e0 = err_e0 = loss_e0 = None
                if mode != "fixed-e0":
                    q_e2, err_e2, loss_e2 = _simulate_span(
                        span, h_span, original_columns, scales_e2, "e2"
                    )
                if mode != "fixed-e2":
                    q_e0, err_e0, loss_e0 = _simulate_span(
                        span, h_span, original_columns, scales_e0, "e0"
                    )

                if mode == "fixed-e2":
                    chosen_q, chosen_err = q_e2, err_e2
                    choose_e0_rows = torch.zeros(n, dtype=torch.bool, device=source.device)
                elif mode == "fixed-e0":
                    chosen_q, chosen_err = q_e0, err_e0
                    choose_e0_rows = torch.ones(n, dtype=torch.bool, device=source.device)
                else:
                    unit_rows = 1 if mode == "blockmix" else WEIGHT_TILE_N
                    pad_n = (-n) % unit_rows
                    e2p = torch.nn.functional.pad(loss_e2, (0, pad_n)) if pad_n else loss_e2
                    e0p = torch.nn.functional.pad(loss_e0, (0, pad_n)) if pad_n else loss_e0
                    choose_units = e0p.reshape(-1, unit_rows).sum(1) < e2p.reshape(-1, unit_rows).sum(1)
                    choose_e0_rows = choose_units.repeat_interleave(unit_rows)[:n]
                    chosen_q = torch.where(choose_e0_rows[:, None], q_e0, q_e2)
                    chosen_err = torch.where(choose_e0_rows[:, None], err_e0, err_e2)
                    selected_e0 += int(choose_units.sum().item())
                    selected_e2 += choose_units.numel() - int(choose_units.sum().item())
                    incremental_e2 += float(loss_e2.sum().item())
                    incremental_e0 += float(loss_e0.sum().item())
                    chosen_loss = torch.where(choose_e0_rows, loss_e0, loss_e2)
                    incremental_selected += float(chosen_loss.sum().item())

                quantized[:, span_start:span_end] = chosen_q
                if span_end < k:
                    work[:, span_end:].sub_(
                        chosen_err @ h_inv[span_start:span_end, span_end:]
                    )
            cursor = tile_end
        if not torch.isfinite(quantized).all():
            failure = "non-finite GPTQ output"
            diagonal.add_(max(float(base_damping), 1e-8) * (10 ** attempt))
            continue
        result = quantized[:, inverse]
        break

    stats = {
        "mode": mode,
        "gptq_status": "gptq" if result is not None else "failed",
        "attempts": attempts,
        "failure": failure if result is None else None,
        "e2_count": selected_e2,
        "e0_count": selected_e0,
        "total_count": selected_e2 + selected_e0,
        "e0_ratio": (
            selected_e0 / (selected_e2 + selected_e0)
            if selected_e2 + selected_e0 else (1.0 if mode == "fixed-e0" else 0.0)
        ),
        "incremental_e2": incremental_e2,
        "incremental_e0": incremental_e0,
        "incremental_selected": incremental_selected,
    }
    return result, stats


def hessian_trace_loss(source: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Return ``tr((W-Q) H (W-Q)^T)`` with float64 accumulation."""
    if source.shape != quantized.shape or hessian.shape != (source.shape[1], source.shape[1]):
        raise ValueError("weight/Hessian loss shape mismatch")
    error = source.float() - quantized.float()
    return (error * (error @ hessian.float())).double().sum().clamp_min(0)


@torch.no_grad()
def collect_e0_activation_hessians(
    transformer,
    calib_dir: str | Path,
    num_calib_files: int,
    batch_size: int,
    device: str = "cuda",
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stream cached chunks and accumulate ``H_Z=2/n Z^T Z`` online."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("E0 activation Hessian collection requires CUDA")
    layers = _active_sana_layers(transformer)
    model_dtype = next(transformer.parameters()).dtype
    hessians = {}
    rows = {}
    chunks = {}
    total_bytes = 0
    for name, layer in layers.items():
        if layer.rotation is not None:
            k = layer.module.in_features - int(layer.quantizer.high_bits_length)
        else:
            k = int(layer.num_heads) * (
                int(layer.head_dim) - int(layer.quantizer.high_bits_length)
            )
        hessians[name] = torch.zeros(k, k, dtype=torch.float32, device=device)
        rows[name] = chunks[name] = 0
        total_bytes += k * k * 4

    hooks = []
    for name, layer in layers.items():
        def make_hook(layer_name, qlayer):
            def hook(_module, inputs, _output):
                activation = project_low_activation(qlayer, inputs[0].detach())
                quantized = quantize_e0_per_chunk(activation)
                flat = quantized.reshape(-1, quantized.shape[-1]).float()
                if not torch.isfinite(flat).all():
                    raise RuntimeError(f"{layer_name}: non-finite E0 calibration activation")
                hessians[layer_name].addmm_(flat.T, flat)
                rows[layer_name] += flat.shape[0]
                chunks[layer_name] += activation.shape[0]
            return hook
        hooks.append(layer.register_forward_hook(make_hook(name, layer)))

    files = sorted(Path(calib_dir).glob("*.pt"))[:num_calib_files]
    if len(files) != num_calib_files:
        raise RuntimeError(f"expected {num_calib_files} calibration chunks, found {len(files)}")
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    def load_batch(paths):
        return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]

    def run_batch(batch):
        arguments = []
        keyword_lists = {}
        for data in batch:
            arguments.append([
                value.to(model_dtype) if value.is_floating_point() else value
                for value in data["input_args"]
            ])
            for key, value in data["input_kwargs"].items():
                keyword_lists.setdefault(key, []).append(value)
        latents = torch.cat([values[0] for values in arguments], dim=0).to(device)
        keywords = {}
        for key, values in keyword_lists.items():
            value = values[0]
            if isinstance(value, torch.Tensor):
                if value.ndim >= 1 and value.shape[0] == 1:
                    value = torch.cat(values, dim=0)
                if value.is_floating_point():
                    value = value.to(model_dtype)
                value = value.to(device)
            keywords[key] = value
        transformer(latents, **keywords)

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            queue = deque()
            iterator = iter(batches)
            for _ in range(2):
                try:
                    queue.append(pool.submit(load_batch, next(iterator)))
                except StopIteration:
                    break
            for _ in tqdm(range(len(batches)), desc="E0 activation Hessian calibration"):
                try:
                    queue.append(pool.submit(load_batch, next(iterator)))
                except StopIteration:
                    pass
                run_batch(queue.popleft().result())
    finally:
        for hook in hooks:
            hook.remove()
        for layer in layers.values():
            if hasattr(layer, "_e0joint_runtime_rotation"):
                delattr(layer, "_e0joint_runtime_rotation")

    output = {}
    for name in layers:
        if chunks[name] != num_calib_files or rows[name] <= 0:
            raise RuntimeError(
                f"{name}: incomplete Hessian chunks={chunks[name]}, rows={rows[name]}"
            )
        hessians[name].mul_(2.0 / rows[name])
        if not torch.isfinite(hessians[name]).all():
            raise RuntimeError(f"{name}: non-finite E0 activation Hessian")
        output[name] = hessians[name].cpu()
    elapsed = time.perf_counter() - started
    return output, {
        "calibration_files": len(files), "calibration_batches": len(batches),
        "calibration_seconds": elapsed, "hessian_gpu_gib": total_bytes / 1024**3,
        "rows_by_layer": rows, "chunks_by_layer": chunks,
    }


def _base_cache_state(transformer) -> dict:
    state = transformer.state_dict()
    transient = (".quantizer.scale", ".quantizer.zero")
    return {key: value.detach().cpu() for key, value in state.items()
            if not key.endswith(transient)}


@torch.no_grad()
def build_weight_mix_caches(
    transformer,
    *,
    calib_dir: str | Path,
    hessian_cache: Path,
    cache_paths: dict[str, Path],
    standard_cache: Path,
    report_dir: Path,
    basis_path: Path,
    rotation_path: Path,
    skip_layers: list[str],
    num_calib_files: int = 5120,
    batch_size: int = 4,
    damp_pct: float = .01,
    device: str = "cuda",
) -> dict:
    """Build common H_Z, three independent GPTQ caches, and BlockMix stats."""
    if set(cache_paths) != set(PRODUCTION_MODES):
        raise ValueError(f"cache paths must cover {PRODUCTION_MODES}")
    if not standard_cache.exists():
        raise FileNotFoundError(f"missing standard GPTQ cache: {standard_cache}")
    for mode, path in cache_paths.items():
        if path == standard_cache or "weightmix" not in path.name:
            raise ValueError(f"unsafe {mode} cache path: {path}")
    report_dir.mkdir(parents=True, exist_ok=True)
    layers = _active_sana_layers(transformer)
    common = {
        "model": "sana-1.6b", "objective_version": OBJECTIVE_VERSION,
        "activation_hessian": "hardware-e0m3-gscale2688-per-calibration-chunk",
        "weight_group_size": WEIGHT_GROUP_SIZE,
        "logical_weight_tile": [WEIGHT_TILE_K, WEIGHT_TILE_N],
        "stored_weight_tile": [WEIGHT_TILE_N, WEIGHT_TILE_K],
        "residual_rotation": "random", "calibration_count": num_calib_files,
        "calibration_batch_size": batch_size, "damp_pct": damp_pct,
        "skip_layer_hash": skip_layer_hash(skip_layers),
        "quantizer_scale_semantics": WEIGHT_SCALE_SEMANTICS,
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
        "standard_cache": str(standard_cache),
        "standard_cache_sha256": sha256_file(standard_cache),
        "active_layers": len(layers),
    }

    hessian_meta_path = metadata_path(hessian_cache)
    hessian_expected = {**common, "cache_kind": "e0-activation-hessian"}
    if hessian_cache.exists():
        meta = json.loads(hessian_meta_path.read_text())
        for key, value in hessian_expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"H_Z cache metadata mismatch for {key}")
        hessians = torch.load(hessian_cache, map_location="cpu", weights_only=False)
        if set(hessians) != set(layers):
            raise RuntimeError("H_Z cache layer set does not match active model")
        calibration = {"hessian_cache_reused": True, **meta["calibration"]}
    else:
        hessians, calibration_raw = collect_e0_activation_hessians(
            transformer, calib_dir, num_calib_files, batch_size, device
        )
        hessian_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(hessians, hessian_cache)
        calibration = {"hessian_cache_reused": False, **calibration_raw}
        _write_json(hessian_meta_path, {
            **hessian_expected, "calibration": calibration_raw,
            "cache_sha256": sha256_file(hessian_cache),
        })

    standard_state = torch.load(standard_cache, map_location="cpu", weights_only=False)
    base_state = _base_cache_state(transformer)
    weights_by_mode: dict[str, dict[str, torch.Tensor]] = {
        mode: {} for mode in PRODUCTION_MODES
    }
    layer_rows = []
    aggregate = {mode: 0.0 for mode in (*PRODUCTION_MODES, "blockmix")}
    aggregate_counts = {
        mode: {"e2": 0, "e0": 0} for mode in ("tilemix", "blockmix")
    }
    fallbacks = []
    started = time.perf_counter()
    for name, layer in tqdm(layers.items(), desc="E0-Hessian weight GPTQ formats"):
        standard_key = f"{name}.module.weight"
        if standard_key not in standard_state:
            raise RuntimeError(f"standard cache missing {standard_key}")
        original_low, original_high, stitch, _ = _rotate_and_split_W(
            layer, layer.module.weight.data
        )
        # Audit that the existing cache leaves the native high branch exact.
        _std_low, std_high = extract_fused_low_weight(layer, standard_state[standard_key])
        if original_high is not None:
            cache_dtype = standard_state[standard_key].dtype
            if not torch.equal(
                original_high.to(dtype=cache_dtype).cpu(),
                std_high.to(dtype=cache_dtype).cpu(),
            ):
                raise RuntimeError(f"{name}: existing standard cache changed high branch")
        source = original_low.to(device=device, dtype=torch.float32)
        hessian = hessians[name].to(device=device, dtype=torch.float32)
        outputs = {}
        stats_by_mode = {}
        for mode in (*PRODUCTION_MODES, "blockmix"):
            quantized, stats = gptq_quantize_weight_tiles(
                source, hessian, mode, damp_pct=damp_pct
            )
            if quantized is None:
                fallbacks.append({"layer": name, "mode": mode, "reason": stats["failure"]})
                continue
            outputs[mode] = quantized
            stats_by_mode[mode] = stats
            loss = float(hessian_trace_loss(source, quantized, hessian).item())
            aggregate[mode] += loss
            if mode in aggregate_counts:
                aggregate_counts[mode]["e2"] += stats["e2_count"]
                aggregate_counts[mode]["e0"] += stats["e0_count"]
            if mode in PRODUCTION_MODES:
                fused = stitch(quantized).to(layer.module.weight.dtype).cpu()
                final_low, final_high = extract_fused_low_weight(layer, fused)
                expected_fused = stitch(original_low).to(layer.module.weight.dtype).cpu()
                _expected_low, expected_high = extract_fused_low_weight(layer, expected_fused)
                if final_low.shape != quantized.shape:
                    raise RuntimeError(f"{name}/{mode}: stitched low shape mismatch")
                if not ((final_high is None and expected_high is None) or
                        (final_high is not None and expected_high is not None and
                         torch.equal(final_high, expected_high))):
                    raise RuntimeError(f"{name}/{mode}: high branch changed")
                weights_by_mode[mode][name] = fused
            layer_rows.append({
                "layer": name, "mode": mode, "n": source.shape[0], "k": source.shape[1],
                "calibration_loss": loss, "gptq_status": stats["gptq_status"],
                "attempts": stats["attempts"], "fallback_reason": stats["failure"] or "",
                "e2_count": stats["e2_count"], "e0_count": stats["e0_count"],
                "e0_ratio": stats["e0_ratio"],
            })
        del hessians[name], source, hessian, outputs
        torch.cuda.empty_cache()

    if fallbacks:
        _write_json(report_dir / "weight_mix_fallbacks.json", {"fallbacks": fallbacks})
        raise RuntimeError(f"weight Mix GPTQ had {len(fallbacks)} failures; no cache saved")

    quantization_seconds = time.perf_counter() - started
    best_fixed = min(aggregate["fixed-e2"], aggregate["fixed-e0"])
    denominator = best_fixed - aggregate["blockmix"]
    retained = ((best_fixed - aggregate["tilemix"]) / denominator
                if denominator > 0 else None)
    tile_counts = aggregate_counts["tilemix"]
    tile_total = tile_counts["e2"] + tile_counts["e0"]
    tile_e0_ratio = tile_counts["e0"] / tile_total if tile_total else 0.0

    # Save one state at a time.  Wrapper and module weight keys deliberately
    # point to the same tensor object, matching the existing state-dict alias.
    cache_records = {}
    for mode in PRODUCTION_MODES:
        state = dict(base_state)
        for name, fused in weights_by_mode[mode].items():
            state[f"{name}.weight"] = fused
            state[f"{name}.module.weight"] = fused
        cache_path = cache_paths[mode]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, cache_path)
        metadata = {
            **common, "weight_format": mode, "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "gptq_layers": len(layers), "rtn_fallbacks": [],
            "high_branch_unchanged_layers": len(layers),
            "aggregate_calibration_loss": aggregate[mode],
        }
        _write_json(metadata_path(cache_path), metadata)
        cache_records[mode] = metadata
        del state

    summary = {
        **common, **calibration,
        "hessian_cache": str(hessian_cache),
        "hessian_cache_sha256": sha256_file(hessian_cache),
        "quantization_seconds": quantization_seconds,
        "aggregate_losses": aggregate,
        "best_fixed_loss": best_fixed,
        "tilemix_reduction_vs_best_fixed": (
            (best_fixed - aggregate["tilemix"]) / best_fixed
        ),
        "blockmix_reduction_vs_best_fixed": (
            (best_fixed - aggregate["blockmix"]) / best_fixed
        ),
        "tilemix_blockmix_gain_retained": retained,
        "tile_counts": tile_counts,
        "tile_e0_ratio": tile_e0_ratio,
        "block_counts": aggregate_counts["blockmix"],
        "gptq_layers": len(layers), "rtn_fallbacks": [],
        "high_branch_unchanged_layers": len(layers),
        "cache_records": cache_records,
    }
    gate = {
        "loss_reduction_pass": summary["tilemix_reduction_vs_best_fixed"] >= .02,
        "format_ratio_pass": .05 <= tile_e0_ratio <= .95,
        "retained_gain_pass": retained is not None and retained >= .5,
        "all_layers_gptq_pass": len(layers) == REQUIRED_ACTIVE_LAYERS,
        "high_branch_pass": True,
    }
    summary["calibration_gate"] = {**gate, "passed": all(gate.values())}
    _write_rows(report_dir / "weight_mix_layer_objectives.csv", layer_rows)
    _write_json(report_dir / "weight_mix_summary.json", summary)
    del standard_state, base_state, weights_by_mode, hessians
    gc.collect()
    torch.cuda.empty_cache()
    return summary


__all__ = [
    "OBJECTIVE_VERSION", "PRODUCTION_MODES", "WEIGHT_GROUP_SIZE",
    "WEIGHT_SCALE_SEMANTICS", "WEIGHT_TILE_K", "WEIGHT_TILE_N",
    "build_weight_mix_caches", "collect_e0_activation_hessians",
    "expected_metadata", "gptq_quantize_weight_tiles", "hessian_trace_loss",
    "logical_tile_to_stored_slice", "metadata_path", "quantize_weight_blocks",
    "validate_cache_metadata",
]
