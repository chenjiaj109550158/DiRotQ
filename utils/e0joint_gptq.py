"""SANA-only E0-aware joint activation/weight GPTQ feasibility tooling.

This experiment keeps the runtime contract fixed: hardware-faithful E0M3
activations multiply dequantized NVFP4 E2M1 GPTQ weights.  Only the offline
low-residual weight objective changes from ``A(W-Q)`` to ``AW-ZQ``.
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

from .gptq_utils import _gptq_quantize_layer
from .quant_utils import (
    ActQuantWrapper,
    NF4_MAX,
    _quant_group_nvfp4,
    _rotate_and_split_W,
    find_qlayers,
    round_to_nf4_codebook,
)
from .tilemixfp4_utils import E2M1_MAGNITUDES, fake_quantize_e0m3


OBJECTIVE_VERSION = "e0joint-v1"
REQUIRED_ACTIVE_LAYERS = 120


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def e0joint_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".metadata.json")


def validate_e0joint_cache_path(cache_path: Path, enabled: bool) -> None:
    tagged = "e0joint" in cache_path.name
    if enabled and not tagged:
        raise ValueError("E0-aware joint GPTQ requires an e0joint-tagged cache path")
    if tagged and not enabled:
        raise ValueError("an e0joint cache may only be loaded with --e0joint-gptq")


def validate_e0joint_metadata(cache_path: Path, expected: dict) -> dict:
    metadata_path = e0joint_metadata_path(cache_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing E0-joint cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"E0-joint cache metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    return metadata


def write_e0joint_metadata(cache_path: Path, metadata: dict) -> Path:
    output = e0joint_metadata_path(cache_path)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output


def quantize_e0_per_chunk(a: torch.Tensor) -> torch.Tensor:
    """Quantize each leading batch entry with its own runtime E0 global scale."""
    if a.ndim < 2:
        raise ValueError("chunked activation must have a leading chunk dimension")
    if not a.is_floating_point():
        raise TypeError("chunked activation must be floating point")
    if not torch.isfinite(a).all():
        raise ValueError("non-finite full-precision residual activation")
    return torch.stack([fake_quantize_e0m3(chunk) for chunk in a.unbind(0)], dim=0)


def project_low_activation(qlayer: ActQuantWrapper, x: torch.Tensor) -> torch.Tensor:
    """Project pre-wrapper input to the exact low residual layout used at runtime."""
    if x.device != qlayer.module.weight.device:
        raise RuntimeError("joint calibration does not permit a CPU/device fallback")
    dtype = qlayer.module.weight.dtype
    batch = x.shape[0]
    if qlayer.rotation is not None:
        cache_key = (x.device, dtype, "hidden")
        cache = getattr(qlayer, "_e0joint_runtime_rotation", None)
        if cache is None or cache[0] != cache_key:
            cache = (cache_key, qlayer.rotation.to(device=x.device, dtype=dtype))
            qlayer._e0joint_runtime_rotation = cache
        rotation = cache[1]
        flat = x.to(dtype).reshape(-1, x.shape[-1])
        rotated = flat @ rotation
        low = rotated.shape[-1] - int(qlayer.quantizer.high_bits_length)
        if low <= 0:
            raise ValueError("joint calibration found no low residual channels")
        return rotated[:, :low].reshape(batch, -1, low)

    if qlayer.rotation_per_head is not None:
        heads, head_dim = int(qlayer.num_heads), int(qlayer.head_dim)
        high = int(qlayer.quantizer.high_bits_length)
        low_per_head = head_dim - high
        if low_per_head <= 0:
            raise ValueError("joint calibration found no per-head low channels")
        cache_key = (x.device, dtype, "per-head")
        cache = getattr(qlayer, "_e0joint_runtime_rotation", None)
        if cache is None or cache[0] != cache_key:
            cache = (
                cache_key,
                qlayer.rotation_per_head.to(device=x.device, dtype=dtype),
            )
            qlayer._e0joint_runtime_rotation = cache
        rotation = cache[1]
        shaped = x.to(dtype).reshape(batch, -1, heads, head_dim)
        rotated = torch.einsum("bmhd,hde->bmhe", shaped, rotation)
        return rotated[..., :low_per_head].reshape(
            batch, -1, heads * low_per_head
        )

    raise ValueError("E0-joint SANA calibration requires random residual rotation")


def extract_fused_low_weight(
    qlayer: ActQuantWrapper, fused_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract low/high regions from an already transformed cache weight."""
    weight = fused_weight.float()
    if qlayer.rotation is not None:
        high = int(qlayer.quantizer.high_bits_length)
        low = weight.shape[1] - high
        return weight[:, :low].contiguous(), (
            weight[:, low:].contiguous() if high else None
        )
    if qlayer.rotation_per_head is not None:
        heads, head_dim = int(qlayer.num_heads), int(qlayer.head_dim)
        high = int(qlayer.quantizer.high_bits_length)
        low = head_dim - high
        shaped = weight.reshape(weight.shape[0], heads, head_dim)
        low_weight = shaped[..., :low].reshape(weight.shape[0], heads * low)
        high_weight = (
            shaped[..., low:].reshape(weight.shape[0], heads * high)
            if high else None
        )
        return low_weight.contiguous(), (
            high_weight.contiguous() if high_weight is not None else None
        )
    raise ValueError("unsupported fused weight layout for E0-joint SANA")


def direct_joint_loss(
    a: torch.Tensor, z: torch.Tensor, w: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Direct ``||AW-ZQ||_F^2`` for tests and small diagnostics (W/Q are KxN)."""
    return (a @ w - z @ q).double().square().sum()


def quadratic_joint_loss(
    s: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Compute ``||AW-ZQ||²`` from S=A'A, H=Z'Z, C=Z'A."""
    if not all(torch.isfinite(t).all() for t in (s, h, c, w, q)):
        raise ValueError("non-finite joint quadratic input")
    target = (w * (s @ w)).double().sum()
    cross = (q * (c @ w)).double().sum()
    predicted = (q * (h @ q)).double().sum()
    value = target - 2.0 * cross + predicted
    tolerance = 1e-7 * max(float(target.abs()), 1.0)
    if float(value) < -tolerance:
        raise RuntimeError(f"joint quadratic loss is materially negative: {float(value)}")
    return value.clamp_min(0.0)


def solve_compensated_weight(
    h: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor,
    damp_pct: float,
    max_cholesky_tries: int = 6,
) -> tuple[torch.Tensor, float, str, int]:
    """Solve ``(H+lambda I) Wc = C W`` with reported numerical fallback."""
    if damp_pct < 0:
        raise ValueError("damp_pct must be non-negative")
    if not all(torch.isfinite(t).all() for t in (h, c, w)):
        raise ValueError("non-finite continuous compensation input")
    rhs = c @ w
    eye = torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    base = float((h.diagonal().mean() * damp_pct).clamp_min(0).item())
    floor = max(float(h.diagonal().abs().mean().item()) * 1e-8, 1e-12)
    damping = base
    for attempt in range(1, max_cholesky_tries + 1):
        effective = damping if damping > 0 else (0.0 if attempt == 1 else floor)
        matrix = h + effective * eye
        chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info.max().item()) == 0:
            result = torch.cholesky_solve(rhs, chol)
            if torch.isfinite(result).all():
                return result, effective, "cholesky", attempt
        damping = max(floor, (effective if effective > 0 else floor) * 10.0)

    effective = max(damping, floor)
    try:
        result = torch.linalg.solve(h + effective * eye, rhs)
    except RuntimeError as exc:
        raise RuntimeError("both Cholesky and direct solve failed") from exc
    if not torch.isfinite(result).all():
        raise RuntimeError("direct solve produced non-finite compensated weight")
    return result, effective, "solve-fallback", max_cholesky_tries


def validate_nvfp4_e2m1_weight(
    quantized: torch.Tensor,
    source: torch.Tensor,
    groupsize: int,
    atol: float = 2e-5,
) -> bool:
    """Confirm a GPTQ output uses the original core's legal E2M1 group scales."""
    if quantized.shape != source.shape or quantized.ndim != 2:
        return False
    out_features, in_features = source.shape
    pad = (-in_features) % groupsize
    source_pad = torch.nn.functional.pad(source, (0, pad)) if pad else source
    quant_pad = torch.nn.functional.pad(quantized, (0, pad)) if pad else quantized
    src_groups = source_pad.reshape(out_features, -1, groupsize)
    q_groups = quant_pad.reshape(out_features, -1, groupsize)
    scales = src_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / NF4_MAX
    normalized = q_groups / scales
    reconstructed = round_to_nf4_codebook(normalized) * scales
    if pad:
        reconstructed = reconstructed.reshape(out_features, -1)[:, :in_features]
    else:
        reconstructed = reconstructed.reshape_as(quantized)
    return bool(torch.allclose(reconstructed, quantized, rtol=0.0, atol=atol))


def _active_sana_layers(transformer) -> dict[str, ActQuantWrapper]:
    layers = {
        name: layer for name, layer in find_qlayers(
            transformer, layers=[ActQuantWrapper]
        ).items()
        if layer.quantizer.bits < 16
    }
    if len(layers) != REQUIRED_ACTIVE_LAYERS:
        raise RuntimeError(
            f"E0-joint requires {REQUIRED_ACTIVE_LAYERS} active SANA layers, "
            f"found {len(layers)}"
        )
    for name, layer in layers.items():
        if layer.rotation is None and layer.rotation_per_head is None:
            raise RuntimeError(f"{name}: missing random residual rotation")
    return layers


@torch.no_grad()
def collect_joint_moments(
    transformer,
    calib_dir: str | Path,
    num_calib_files: int,
    batch_size: int,
    device: str = "cuda",
) -> tuple[dict[str, dict], dict]:
    """Stream cached chunks once and accumulate S/H/C without saving activations."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("E0-joint calibration requires CUDA; CPU fallback is forbidden")
    layers = _active_sana_layers(transformer)
    model_dtype = next(transformer.parameters()).dtype
    moments: dict[str, dict] = {}
    total_bytes = 0
    for name, layer in layers.items():
        if layer.rotation is not None:
            k = layer.module.in_features - int(layer.quantizer.high_bits_length)
        else:
            k = int(layer.num_heads) * (
                int(layer.head_dim) - int(layer.quantizer.high_bits_length)
            )
        moments[name] = {
            "S": torch.zeros(k, k, dtype=torch.float32, device=device),
            "H": torch.zeros(k, k, dtype=torch.float32, device=device),
            "C": torch.zeros(k, k, dtype=torch.float32, device=device),
            "rows": 0,
            "chunks": 0,
            "k": k,
        }
        total_bytes += 3 * k * k * 4
    print(
        f"Allocated S/H/C for {len(layers)} layers on CUDA "
        f"(~{total_bytes / 1024**3:.2f} GiB); no CPU fallback."
    )

    hooks = []
    for name, layer in layers.items():
        def make_hook(layer_name, qlayer):
            def hook(_module, inputs, _output):
                x = inputs[0].detach()
                a = project_low_activation(qlayer, x)
                z = quantize_e0_per_chunk(a)
                a_flat = a.reshape(-1, a.shape[-1]).float()
                z_flat = z.reshape(-1, z.shape[-1]).float()
                if not torch.isfinite(z_flat).all():
                    raise RuntimeError(f"{layer_name}: E0 quantization produced non-finite Z")
                stats = moments[layer_name]
                stats["S"].addmm_(a_flat.T, a_flat)
                stats["H"].addmm_(z_flat.T, z_flat)
                stats["C"].addmm_(z_flat.T, a_flat)
                stats["rows"] += a_flat.shape[0]
                stats["chunks"] += a.shape[0]
            return hook
        hooks.append(layer.register_forward_hook(make_hook(name, layer)))

    files = sorted(Path(calib_dir).glob("*.pt"))[:num_calib_files]
    if len(files) != num_calib_files:
        raise RuntimeError(
            f"expected {num_calib_files} calibration chunks, found {len(files)}"
        )
    file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    def load_batch(paths):
        return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]

    def run_batch(batch_data):
        args_lists: list[list[torch.Tensor]] = []
        kwargs_lists: dict[str, list] = {}
        for data in batch_data:
            args_lists.append([
                value.to(model_dtype) if value.is_floating_point() else value
                for value in data["input_args"]
            ])
            for key, value in data["input_kwargs"].items():
                kwargs_lists.setdefault(key, []).append(value)
        latents = torch.cat([args[0] for args in args_lists], dim=0).to(device)
        kwargs = {}
        for key, values in kwargs_lists.items():
            value = values[0]
            if isinstance(value, torch.Tensor):
                if value.ndim >= 1 and value.shape[0] == 1:
                    value = torch.cat(values, dim=0)
                if value.is_floating_point():
                    value = value.to(model_dtype)
                kwargs[key] = value.to(device)
            else:
                kwargs[key] = value
        transformer(latents, **kwargs)

    transformer.eval()
    started = time.perf_counter()
    prefetch = 2
    try:
        with ThreadPoolExecutor(max_workers=prefetch) as pool:
            queue = deque()
            iterator = iter(file_batches)
            for _ in range(prefetch):
                try:
                    queue.append(pool.submit(load_batch, next(iterator)))
                except StopIteration:
                    break
            for _ in tqdm(range(len(file_batches)), desc="E0-joint S/H/C calibration"):
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

    for name, stats in moments.items():
        if stats["chunks"] != num_calib_files or stats["rows"] <= 0:
            raise RuntimeError(
                f"{name}: incomplete moments chunks={stats['chunks']} rows={stats['rows']}"
            )
        scale = 1.0 / stats["rows"]
        for key in ("S", "H", "C"):
            stats[key].mul_(scale)
            if not torch.isfinite(stats[key]).all():
                raise RuntimeError(f"{name}: non-finite accumulated {key}")
    elapsed = time.perf_counter() - started
    return moments, {
        "calibration_files": len(files),
        "calibration_batches": len(file_batches),
        "calibration_seconds": elapsed,
        "moment_gpu_gib": total_bytes / 1024**3,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def build_e0joint_gptq(
    transformer,
    calib_dir: str | Path,
    standard_cache: Path,
    report_dir: Path,
    basis_path: Path,
    rotation_path: Path,
    num_calib_files: int = 5120,
    batch_size: int = 4,
    damp_pct: float = 0.01,
    block_size: int = 128,
    groupsize: int = 16,
    device: str = "cuda",
    early_stop_threshold: float = 0.10,
) -> dict:
    """Calibrate, gate, and optionally mutate SANA weights to joint GPTQ values."""
    report_dir.mkdir(parents=True, exist_ok=True)
    if not standard_cache.exists() or "e0joint" in standard_cache.name:
        raise FileNotFoundError(
            f"standard NVFP4 E2M1 GPTQ cache is missing/invalid: {standard_cache}"
        )
    layers = _active_sana_layers(transformer)
    moments, calibration = collect_joint_moments(
        transformer, calib_dir, num_calib_files, batch_size, device
    )
    standard_state = torch.load(standard_cache, map_location="cpu", weights_only=False)
    continuous_rows = []
    continuous_weights: dict[str, torch.Tensor] = {}
    solvers = {"cholesky": 0, "solve-fallback": 0}
    aggregate_std = 0.0
    aggregate_continuous = 0.0

    for name, layer in tqdm(layers.items(), desc="E0-joint continuous compensation"):
        key = f"{name}.module.weight"
        if key not in standard_state:
            raise RuntimeError(f"standard cache missing active weight {key}")
        original_low, original_high, _stitch, _ = _rotate_and_split_W(
            layer, layer.module.weight.data
        )
        cached_weight = standard_state[key]
        standard_low, standard_high = extract_fused_low_weight(layer, cached_weight)
        # Standard GPTQ stitches the FP32-rotated tail and then saves the full
        # layer back in the model's native dtype.  Compare after that exact
        # cache-dtype round trip; comparing the pre-cast FP32 tail against a
        # BF16 cache with a fixed tolerance rejects legitimate BF16 rounding.
        if original_high is not None:
            expected_high = original_high.to(
                device="cpu", dtype=cached_weight.dtype
            )
            cached_high = standard_high.to(dtype=cached_weight.dtype)
            if not torch.equal(expected_high, cached_high):
                raise RuntimeError(
                    f"{name}: standard cache changed the high-precision branch"
                )
        stats = moments[name]
        s, h, c = stats["S"], stats["H"], stats["C"]
        w = original_low.to(device=device, dtype=torch.float32).T.contiguous()
        q_std = standard_low.to(device=device, dtype=torch.float32).T.contiguous()
        wc, damping, solver, attempts = solve_compensated_weight(
            h, c, w, damp_pct
        )
        solvers[solver] += 1
        e_std = float(quadratic_joint_loss(s, h, c, w, q_std).item())
        e_cont = float(quadratic_joint_loss(s, h, c, w, wc).item())
        if e_std <= 0 or not math.isfinite(e_std + e_cont):
            raise RuntimeError(f"{name}: invalid calibration loss")
        rc = (e_std - e_cont) / e_std
        aggregate_std += e_std
        aggregate_continuous += e_cont
        continuous_weights[name] = wc.T.cpu()
        continuous_rows.append({
            "layer": name, "rows": stats["rows"], "chunks": stats["chunks"],
            "k": stats["k"], "e_std": e_std, "e_continuous": e_cont,
            "r_continuous": rc, "lambda": damping, "solver": solver,
            "solver_attempts": attempts, "e_joint": "", "r_quantized": "",
            "continuous_gain_retained": "", "gptq_status": "not_run",
        })

    aggregate_rc = (aggregate_std - aggregate_continuous) / aggregate_std
    base_report = {
        "objective_version": OBJECTIVE_VERSION,
        "activation_format": "e0m3",
        "weight_format": "nvfp4-e2m1",
        "groupsize": groupsize,
        "calibration_count": num_calib_files,
        "calibration_batch_size": batch_size,
        "damp_pct": damp_pct,
        "residual_rotation": "random",
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
        "standard_cache": str(standard_cache),
        "standard_cache_sha256": sha256_file(standard_cache),
        "active_layers": len(layers),
        "aggregate_e_std": aggregate_std,
        "aggregate_e_continuous": aggregate_continuous,
        "aggregate_r_continuous": aggregate_rc,
        "early_stop_threshold": early_stop_threshold,
        "continuous_solvers": solvers,
        **calibration,
    }
    _write_rows(report_dir / "e0joint_layer_objectives.csv", continuous_rows)
    if aggregate_rc < early_stop_threshold:
        result = {
            **base_report,
            "status": "JOINT COMPENSATION NOT VIABLE",
            "cache_created": False,
            "aggregate_e_joint": None,
            "aggregate_r_quantized": None,
            "aggregate_continuous_gain_retained": None,
        }
        (report_dir / "e0joint_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        del moments, continuous_weights, standard_state
        gc.collect()
        torch.cuda.empty_cache()
        return result

    aggregate_joint = 0.0
    gptq_count = 0
    fallbacks = []
    for row in tqdm(continuous_rows, desc="E0-joint NVFP4 GPTQ"):
        name = row["layer"]
        layer = layers[name]
        original_low, _original_high, stitch, gs_override = _rotate_and_split_W(
            layer, layer.module.weight.data
        )
        gs = gs_override if gs_override is not None else groupsize
        wc_source = continuous_weights[name].to(device=device, dtype=torch.float32)
        h = moments[name]["H"]
        q_joint = _gptq_quantize_layer(
            wc_source, h, bits=4, groupsize=gs, sym=True,
            damp_pct=damp_pct, block_size=block_size, num_inv_tries=250,
            device=device, nvfp4=True,
        )
        if q_joint is None:
            reason = "GPTQ Cholesky/inversion/non-finite retry exhaustion"
            q_joint = _quant_group_nvfp4(wc_source, gs)
            fallbacks.append({"layer": name, "reason": reason, "fallback": "RTN"})
            row["gptq_status"] = "rtn-fallback"
        else:
            gptq_count += 1
            row["gptq_status"] = "gptq"
        if not validate_nvfp4_e2m1_weight(q_joint, wc_source, gs):
            raise RuntimeError(f"{name}: joint low weight is not legal NVFP4 E2M1")
        stats = moments[name]
        original_w = original_low.to(device=device, dtype=torch.float32).T.contiguous()
        e_joint = float(quadratic_joint_loss(
            stats["S"], stats["H"], stats["C"], original_w, q_joint.T
        ).item())
        e_std = float(row["e_std"])
        e_cont = float(row["e_continuous"])
        rq = (e_std - e_joint) / e_std
        retained = (
            (e_std - e_joint) / (e_std - e_cont)
            if e_std > e_cont else float("nan")
        )
        row["e_joint"] = e_joint
        row["r_quantized"] = rq
        row["continuous_gain_retained"] = retained
        aggregate_joint += e_joint
        final_weight = stitch(q_joint).to(layer.module.weight.dtype)
        final_low, final_high = extract_fused_low_weight(layer, final_weight)
        expected_high = extract_fused_low_weight(
            layer, stitch(original_low).to(layer.module.weight.dtype)
        )[1]
        high_matches = (
            final_high is None and expected_high is None
        ) or (
            final_high is not None and expected_high is not None and
            torch.equal(final_high, expected_high)
        )
        if not high_matches:
            raise RuntimeError(f"{name}: high branch changed during joint stitching")
        if final_low.shape != q_joint.shape:
            raise RuntimeError(f"{name}: stitched joint low weight shape mismatch")
        layer.module.weight.data = final_weight
        layer._unrot_fused = True
        del moments[name], continuous_weights[name]

    aggregate_rq = (aggregate_std - aggregate_joint) / aggregate_std
    aggregate_retained = (
        (aggregate_std - aggregate_joint) /
        (aggregate_std - aggregate_continuous)
    )
    _write_rows(report_dir / "e0joint_layer_objectives.csv", continuous_rows)
    result = {
        **base_report,
        "status": "gate_passed",
        "cache_created": True,
        "aggregate_e_joint": aggregate_joint,
        "aggregate_r_quantized": aggregate_rq,
        "aggregate_continuous_gain_retained": aggregate_retained,
        "gptq_layers": gptq_count,
        "rtn_fallbacks": fallbacks,
        "legal_e2m1_layers": len(layers),
        "high_branch_unchanged_layers": len(layers),
    }
    (report_dir / "e0joint_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    del moments, continuous_weights, standard_state
    gc.collect()
    torch.cuda.empty_cache()
    return result


__all__ = [
    "OBJECTIVE_VERSION",
    "build_e0joint_gptq",
    "collect_joint_moments",
    "direct_joint_loss",
    "e0joint_metadata_path",
    "extract_fused_low_weight",
    "project_low_activation",
    "quadratic_joint_loss",
    "quantize_e0_per_chunk",
    "sha256_file",
    "solve_compensated_weight",
    "validate_e0joint_cache_path",
    "validate_e0joint_metadata",
    "validate_nvfp4_e2m1_weight",
    "write_e0joint_metadata",
]
