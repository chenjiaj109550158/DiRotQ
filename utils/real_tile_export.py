"""Read-only export of real DiRotQ FP4 operands for the Blackwell receiver.

This module does not quantize weights and does not define a second activation
quantizer.  The activation hook receives the codebook value and E4M3 scale
selected inside the existing hardware-E0 fake-quant path.  Weight payloads,
scales and global scales are read from the already-built packing sidecars and
are checked against their reconstructed BF16 cache before capture is armed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch

from .e0joint_gptq import extract_fused_low_weight, sha256_file
from .hardware_weight_fp4 import (
    decode_packing_record,
    packing_path,
    unpack_nibbles as unpack_weight_nibbles,
    validate_runtime_state,
)
from .quant_utils import ActQuantWrapper
from .tilemixfp4_utils import E0M3_MAGNITUDES


SCHEMA_NAME = "dirotq.blackwell.real_fp4_tile"
SCHEMA_VERSION = 1
GROUP_SIZE = 16
UE4M3_ONE_BYTE = 0x38
A_LAYOUT = "row_major_mk_k_contiguous"
B_LAYOUT = "column_major_kn_each_storage_row_is_one_logical_b_column"
NIBBLE_ORDER = "earlier_k_element_in_low_nibble"
BLOCK_SCALE_ENCODING = "ue4m3_e4m3fn_literal_byte"
GLOBAL_SCALE_SEMANTICS = (
    "C=FP32(alpha_A*alpha_B)*sum_k((D_A*Q_A)*(D_B*Q_B));"
    " global product is applied once after FP32 K accumulation"
)
PACKED_EXPECTED_SEMANTICS = (
    "valid_MxN sequential-K FP32 reference decoded from canonical packed"
    " payload/scales with the global product applied once"
)
FAKEQUANT_EXPECTED_SEMANTICS = (
    "valid_MxN float32 materialization of the producer runtime output;"
    " operands are separately dequantized and cast to runtime_dtype,"
    " torch.matmul is performed on those runtime tensors, output is cast"
    " to runtime_dtype, then materialized as float32 for portable NPY storage"
)
CASE_FILES = {
    "a_payload": "a_payload.npy",
    "a_scales": "a_scales.npy",
    "b_payload": "b_payload.npy",
    "b_scales": "b_scales.npy",
    "expected_packed_fp32": "expected_packed_fp32.npy",
    "expected_fakequant_runtime": "expected_fakequant_runtime.npy",
}


def _ceil(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _float8_bytes(values: torch.Tensor) -> torch.Tensor:
    rounded = values.detach().float().abs().to(torch.float8_e4m3fn)
    if not torch.isfinite(rounded.float()).all():
        raise OverflowError("non-finite E4M3 block scale during real-tile export")
    return rounded.contiguous().view(torch.uint8)


def encode_sign_magnitude(values: torch.Tensor, magnitudes: tuple[float, ...]) -> torch.Tensor:
    """Encode already-selected codebook values; this function never rounds."""
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("logical FP4 code values must be finite floating tensors")
    levels = torch.tensor(magnitudes, dtype=torch.float32, device=values.device)
    absolute = values.float().abs()
    matches = absolute.unsqueeze(-1) == levels
    if not bool(matches.any(dim=-1).all()):
        raise ValueError("capture contains a value outside the selected FP4 codebook")
    index = matches.to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    sign = torch.signbit(values).to(torch.uint8) << 3
    # The runtime fake quantizer canonicalizes zero through torch.sign(), so
    # captured zero is always the receiver's positive-zero nibble.
    sign = torch.where(index == 0, torch.zeros_like(sign), sign)
    return index | sign


def transcode_sidecar_indices_to_sign_magnitude(indices: torch.Tensor) -> torch.Tensor:
    """Losslessly transcode sidecar codebook indices to receiver nibbles.

    Hardware-weight sidecars store indices in the sorted signed codebook
    ``[-max,...,0,...,+max]``.  Receiver v1 stores the same selected logical
    codepoint as sign bit plus magnitude index.  This is an encoding change,
    not re-quantization, and it never reads the reconstructed BF16 weight.
    """
    if indices.dtype != torch.uint8 or bool((indices > 14).any()):
        raise ValueError("hardware sidecar contains an invalid codebook index")
    negative = indices < 7
    positive = indices > 7
    return torch.where(
        negative,
        (15 - indices).to(torch.uint8),
        torch.where(positive, (indices - 7).to(torch.uint8), torch.zeros_like(indices)),
    )


def pack_rows_low_nibble(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[1] % 2:
        raise ValueError("codes must be uint8 rows with an even K")
    if bool((codes > 15).any()):
        raise ValueError("code exceeds one nibble")
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def canonical_a(codes: torch.Tensor, scales: torch.Tensor, m: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    mp, kp = _ceil(m, 16), _ceil(k, 64)
    blocks = (k + 15) // 16
    if tuple(codes.shape) != (m, k) or tuple(scales.shape) != (m, blocks):
        raise ValueError("A capture shape does not match M/K")
    padded_codes = torch.zeros((mp, kp), dtype=torch.uint8)
    padded_codes[:m, :k] = codes.cpu()
    padded_scales = torch.full((mp, kp // 16), UE4M3_ONE_BYTE, dtype=torch.uint8)
    padded_scales[:m, :blocks] = scales.cpu()
    return (
        np.ascontiguousarray(pack_rows_low_nibble(padded_codes).numpy(), dtype=np.uint8),
        np.ascontiguousarray(padded_scales.numpy(), dtype=np.uint8),
    )


def canonical_b(codes_stored: torch.Tensor, scales_stored: torch.Tensor,
                n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Canonical B rows are logical output columns with K contiguous."""
    npad, kp = _ceil(n, 8), _ceil(k, 64)
    blocks = (k + 15) // 16
    if tuple(codes_stored.shape) != (n, k) or tuple(scales_stored.shape) != (n, blocks):
        raise ValueError("B sidecar slice shape does not match N/K")
    padded_codes = torch.zeros((npad, kp), dtype=torch.uint8)
    padded_codes[:n, :k] = codes_stored.cpu()
    padded_scales = torch.full((npad, kp // 16), UE4M3_ONE_BYTE, dtype=torch.uint8)
    padded_scales[:n, :blocks] = scales_stored.cpu()
    return (
        np.ascontiguousarray(pack_rows_low_nibble(padded_codes).numpy(), dtype=np.uint8),
        np.ascontiguousarray(padded_scales.numpy(), dtype=np.uint8),
    )


def sidecar_case_arrays(record: dict, *, column_start: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Read payload/scales/alpha directly from an existing packing record."""
    stored_n, stored_k = map(int, record["stored_shape"])
    if int(record["group_size"]) != GROUP_SIZE or stored_k != k:
        raise ValueError("weight sidecar K/group does not match captured activation")
    if column_start < 0 or column_start + n > stored_n:
        raise ValueError("requested output-column slice is outside the weight sidecar")
    padded_k = _ceil(k, GROUP_SIZE)
    packed_rows = record["packed_payload"][column_start:column_start + n].cpu()
    sidecar_indices = unpack_weight_nibbles(packed_rows, padded_k)[:, :k]
    receiver_codes = transcode_sidecar_indices_to_sign_magnitude(sidecar_indices)
    scale_values = record["block_scales"][
        column_start:column_start + n, : (k + 15) // 16
    ]
    scale_bytes = _float8_bytes(scale_values)
    b_payload, b_scales = canonical_b(receiver_codes, scale_bytes, n, k)
    alpha = float(np.float32(record["global_scale"].float().item()))
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("weight global scale must be finite and positive")
    return b_payload, b_scales, alpha


@dataclass(frozen=True)
class CaptureSpec:
    case_id: str
    layer_name: str
    timestep_index: int
    timestep_occurrence: int
    row_start: int
    column_start: int
    m: int
    n: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CaptureSpec":
        required = {
            "case_id", "layer_name", "timestep_index", "timestep_occurrence",
            "row_start", "column_start", "M", "N",
        }
        if set(value) != required:
            raise ValueError(
                f"capture spec keys differ: missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        result = cls(
            case_id=str(value["case_id"]),
            layer_name=str(value["layer_name"]),
            timestep_index=int(value["timestep_index"]),
            timestep_occurrence=int(value["timestep_occurrence"]),
            row_start=int(value["row_start"]),
            column_start=int(value["column_start"]),
            m=int(value["M"]),
            n=int(value["N"]),
        )
        if min(result.timestep_index, result.timestep_occurrence, result.row_start,
               result.column_start) < 0 or result.m <= 0 or result.n <= 0:
            raise ValueError("capture spec indices and dimensions are invalid")
        return result


@dataclass
class CapturedCase:
    spec: CaptureSpec
    scheduler_timestep: float
    wrapper_call_index: int
    full_call_m: int
    k: int
    alpha_a: float
    global_amax: float
    a_codes: torch.Tensor
    a_scale_bytes: torch.Tensor
    reconstructed_a: torch.Tensor
    runtime_outputs: dict[str, np.ndarray]


def _import_receiver_reference(receiver_root: Path):
    root = str(receiver_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from kernels.blackwell_e0_probe.gemm_probe.packing import (  # noqa: PLC0415
        CanonicalInputs,
        GemmShape,
    )
    from kernels.blackwell_e0_probe.gemm_probe.reference import (  # noqa: PLC0415
        sequential_fp32_gemm,
    )
    return CanonicalInputs, GemmShape, sequential_fp32_gemm


def _receiver_expected(receiver_root: Path, pairing: str, arrays: dict[str, np.ndarray],
                       m: int, n: int, k: int, alpha_a: float, alpha_b: float) -> np.ndarray:
    CanonicalInputs, GemmShape, sequential = _import_receiver_reference(receiver_root)
    inputs = CanonicalInputs(
        shape=GemmShape(m, n, k),
        packed_a=torch.from_numpy(arrays["a_payload"].copy()),
        packed_b=torch.from_numpy(arrays["b_payload"].copy()),
        a_scales=torch.from_numpy(arrays["a_scales"].copy()),
        b_scales=torch.from_numpy(arrays["b_scales"].copy()),
        alpha_a=alpha_a,
        alpha_b=alpha_b,
    )
    b_format = "e2m1" if pairing == "e0xe2" else "e0m3"
    _, expected = sequential(inputs, "e0m3", b_format)
    return np.ascontiguousarray(expected.numpy(), dtype=np.float32)


def _file_record(path: Path, relative: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_size": path.stat().st_size,
    }


def _write_manifest_fixed_size(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "manifest.json"
    previous = None
    for _ in range(16):
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        actual = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        if manifest["package_size_bytes"] == actual and previous == actual:
            return
        manifest["package_size_bytes"] = actual
        previous = actual
    raise RuntimeError("package_size_bytes did not converge")


def verify_with_receiver(receiver_root: Path, package: Path, report: Path | None = None) -> dict:
    verifier = receiver_root / "kernels/blackwell_e0_probe/real_tile_handoff/verify_package.py"
    if not verifier.is_file():
        raise FileNotFoundError(f"receiver verifier not found: {verifier}")
    command = [
        sys.executable, "-m",
        "kernels.blackwell_e0_probe.real_tile_handoff.verify_package",
        str(package),
    ]
    if report is not None:
        command += ["--report", str(report)]
    completed = subprocess.run(
        command, cwd=receiver_root, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(
            f"receiver verifier failed for {package}:\n{completed.stdout}\n{completed.stderr}"
        )
    result = json.loads(completed.stdout)
    if not result.get("passed") or result.get("cuda_touched"):
        raise RuntimeError(f"receiver verifier returned an unsafe result: {result}")
    return result


class RealTileExportController:
    """Capture allowlisted E0 activations and materialize paired v1 packages."""

    def __init__(self, *, config: dict[str, Any], transformer, pipeline,
                 loaded_e2_state: dict[str, torch.Tensor]) -> None:
        self.config = config
        self.transformer = transformer
        self.pipeline = pipeline
        self.receiver_root = Path(config["receiver_root"]).resolve()
        self.output_root = Path(config["output_root"]).resolve()
        self.specs = tuple(CaptureSpec.from_dict(value) for value in config["cases"])
        if not 3 <= len(self.specs) <= 6 or len({s.case_id for s in self.specs}) != len(self.specs):
            raise ValueError("real-tile export requires 3-6 uniquely named cases")
        self.prompt_id = str(config["prompt_image_id"])
        self.model_name = str(config["model_name"])
        self.model_revision = str(config["model_revision"])
        self.producer_commit = str(config["producer_commit"])
        self.basis_sha256 = str(config["pca_basis_sha256"])
        self.rotation_sha256 = str(config["rotation_sha256"])
        self.cache_paths = {
            "e0xe2": Path(config["e2_cache_path"]).resolve(),
            "e0xe0": Path(config["e0_cache_path"]).resolve(),
        }
        self.cache_hashes = {
            pairing: sha256_file(path) for pairing, path in self.cache_paths.items()
        }
        self.sidecar_hashes = {
            pairing: sha256_file(packing_path(path)) for pairing, path in self.cache_paths.items()
        }
        self._layers = {
            name: module for name, module in transformer.named_modules()
            if isinstance(module, ActQuantWrapper)
        }
        requested_layers = {spec.layer_name for spec in self.specs}
        missing = requested_layers - set(self._layers)
        if missing:
            raise ValueError(f"capture layers are not wrapped: {sorted(missing)}")
        if any(self._layers[name].quantizer.quant_dtype != "e0m3" for name in requested_layers):
            raise RuntimeError("capture requires the existing fixed hardware E0M3 activation path")

        self.records: dict[str, dict[str, dict]] = {}
        self.low_weights: dict[str, dict[str, torch.Tensor]] = {}
        self.validation: dict[str, dict[str, Any]] = {}
        self._load_and_validate_weights(loaded_e2_state, requested_layers)

        self.captured: dict[str, CapturedCase] = {}
        self.current_step: int | None = None
        self.current_timestep: float | None = None
        self.current_prompt: str | None = None
        self._layer_calls = {name: 0 for name in requested_layers}
        self._step_occurrences: dict[tuple[str, int], int] = {}
        self._transformer_handle = transformer.register_forward_pre_hook(
            self._transformer_pre_hook, with_kwargs=True
        )
        for name in requested_layers:
            self._layers[name].quantizer.real_tile_capture = self._hook_for(name)

    @classmethod
    def from_file(cls, path: Path | str, transformer, pipeline,
                  loaded_e2_state: dict[str, torch.Tensor]) -> "RealTileExportController":
        return cls(
            config=json.loads(Path(path).read_text()),
            transformer=transformer,
            pipeline=pipeline,
            loaded_e2_state=loaded_e2_state,
        )

    def _load_and_validate_weights(self, e2_state: dict[str, torch.Tensor],
                                   requested_layers: set[str]) -> None:
        states = {"e0xe2": e2_state}
        e0_state = torch.load(
            self.cache_paths["e0xe0"], map_location="cpu", weights_only=False
        )
        states["e0xe0"] = e0_state
        try:
            for pairing, fmt in (("e0xe2", "hardware-fixed-e2"),
                                 ("e0xe0", "hardware-fixed-e0")):
                state = states[pairing]
                metadata_path = self.cache_paths[pairing].with_suffix(
                    self.cache_paths[pairing].suffix + ".metadata.json"
                )
                metadata = json.loads(metadata_path.read_text())
                expected_metadata = {
                    "cache_sha256": self.cache_hashes[pairing],
                    "packing_sha256": self.sidecar_hashes[pairing],
                    "basis_sha256": self.basis_sha256,
                    "rotation_sha256": self.rotation_sha256,
                    "weight_format": fmt,
                    "weight_group_size": GROUP_SIZE,
                    "residual_rotation": "random",
                    "active_layers": 120,
                    "gptq_layers": 120,
                    "rtn_fallbacks": [],
                    "high_branch_unchanged_layers": 120,
                }
                for key, expected in expected_metadata.items():
                    if metadata.get(key) != expected:
                        raise RuntimeError(
                            f"{pairing} metadata mismatch for {key}: "
                            f"{metadata.get(key)!r} != {expected!r}"
                        )
                self.validation[pairing] = validate_runtime_state(
                    self.transformer, state, self.cache_paths[pairing], fmt
                )
                sidecar = torch.load(
                    packing_path(self.cache_paths[pairing]),
                    map_location="cpu", weights_only=False,
                )
                if sidecar.get("format") != fmt or len(sidecar.get("layers", {})) != 120:
                    raise RuntimeError(f"{pairing} sidecar format/coverage mismatch")
                self.records[pairing] = {
                    name: sidecar["layers"][name] for name in requested_layers
                }
                self.low_weights[pairing] = {}
                for name in requested_layers:
                    key = f"{name}.module.weight"
                    low, _ = extract_fused_low_weight(self._layers[name], state[key])
                    if low.dtype != torch.float32:
                        # extract_fused_low_weight intentionally computes in
                        # FP32; recover the exact cache runtime BF16 by decoding
                        # the verified sidecar and casting once, as the cache did.
                        raise AssertionError("unexpected fused-low extraction dtype")
                    decoded = decode_packing_record(
                        self.records[pairing][name], dtype=state[key].dtype
                    )
                    if state[key].dtype != torch.bfloat16:
                        raise RuntimeError("SANA hardware cache is not BF16 reconstructed weight")
                    if not torch.equal(decoded, low.to(torch.bfloat16)):
                        raise RuntimeError(f"{name}: selected cache/sidecar low mismatch")
                    self.low_weights[pairing][name] = decoded.contiguous()
                del sidecar
        finally:
            del e0_state

        for name in requested_layers:
            e2 = self.records["e0xe2"][name]
            e0 = self.records["e0xe0"][name]
            for key in ("logical_shape", "stored_shape", "group_size", "high_branch_hash"):
                if e2[key] != e0[key]:
                    raise RuntimeError(f"{name}: paired sidecars disagree on {key}")

    def _hook_for(self, layer_name: str):
        def hook(event: dict[str, Any]) -> None:
            self._capture(layer_name, event)
        return hook

    def start_batch(self, batch) -> None:
        if len(batch) != 1 or batch[0][0] != self.prompt_id:
            raise RuntimeError(
                f"real-tile capture expected only prompt {self.prompt_id}, got {[x[0] for x in batch]}"
            )
        if self.current_prompt is not None:
            raise RuntimeError("real-tile capture batch already active")
        self.current_prompt = batch[0][0]

    def end_batch(self) -> None:
        self.current_prompt = None
        self.current_step = None
        self.current_timestep = None

    def _transformer_pre_hook(self, module, args, kwargs) -> None:
        if self.current_prompt is None:
            raise RuntimeError("transformer ran without real-tile prompt context")
        if "timestep" not in kwargs:
            raise RuntimeError("transformer did not expose the true scheduler timestep")
        passed = kwargs["timestep"].detach().float().flatten()
        if passed.numel() == 0 or not torch.allclose(passed, passed[:1]):
            raise RuntimeError("timestep is empty or differs across the transformer batch")
        scale = float(getattr(module.config, "timestep_scale", 1.0))
        true_value = float((passed[0] / scale).cpu())
        schedule = self.pipeline.scheduler.timesteps.detach().float().cpu()
        index = int(torch.argmin((schedule - true_value).abs()))
        tolerance = 1e-4 * max(1.0, abs(float(schedule[index])))
        if abs(float(schedule[index]) - true_value) > tolerance:
            raise RuntimeError("passed timestep does not match the real scheduler trajectory")
        self.current_step = index
        self.current_timestep = float(schedule[index])

    @torch.no_grad()
    def _capture(self, layer_name: str, event: dict[str, Any]) -> None:
        call_index = self._layer_calls[layer_name]
        self._layer_calls[layer_name] += 1
        if self.current_step is None or self.current_timestep is None:
            raise RuntimeError("activation capture ran before true timestep attribution")
        step_key = (layer_name, self.current_step)
        occurrence = self._step_occurrences.get(step_key, 0)
        self._step_occurrences[step_key] = occurrence + 1
        candidates = [
            spec for spec in self.specs
            if spec.layer_name == layer_name
            and spec.timestep_index == self.current_step
            and spec.timestep_occurrence == occurrence
            and spec.case_id not in self.captured
        ]
        if not candidates:
            return
        if event["device"].type != "cuda":
            raise RuntimeError("real-tile capture forbids CPU fallback")
        if tuple(event["magnitudes"]) != E0M3_MAGNITUDES:
            raise RuntimeError("real-tile capture was not invoked by E0M3")
        logical = event["logical_codes"]
        scales = event["block_scales"]
        reconstructed = event["reconstructed"].reshape(-1, event["low_k"])
        full_m, k = reconstructed.shape
        logical = logical.reshape(full_m, -1, GROUP_SIZE)
        scales = scales.reshape(full_m, -1)
        alpha = float(np.float32(event["global_scale"].float().item()))
        amax = float(event["global_amax"].float().item())
        if not np.isclose(alpha, 1.0 if amax == 0 else amax / 2688.0, rtol=2e-7, atol=0):
            raise RuntimeError("captured E0 global alpha does not cover the complete low call")

        for spec in candidates:
            if spec.row_start + spec.m > full_m:
                raise RuntimeError(f"{spec.case_id}: requested rows exceed real activation M")
            rows = slice(spec.row_start, spec.row_start + spec.m)
            actual_codes = logical[rows].reshape(spec.m, -1)[:, :k]
            a_codes = encode_sign_magnitude(actual_codes, E0M3_MAGNITUDES).cpu()
            a_scale_bytes = _float8_bytes(scales[rows]).cpu()
            reconstructed_a = reconstructed[rows].to(torch.bfloat16)
            runtime_outputs: dict[str, np.ndarray] = {}
            for pairing in ("e0xe2", "e0xe0"):
                weight = self.low_weights[pairing][layer_name]
                if weight.shape[1] != k or spec.column_start + spec.n > weight.shape[0]:
                    raise RuntimeError(f"{spec.case_id}: activation and weight shapes disagree")
                weight_cuda = weight.to(device=event["device"], dtype=torch.bfloat16)
                # Compute the full real layer output N first, matching runtime
                # GEMM shape, then slice the package's continuous columns.
                full_output = torch.matmul(reconstructed_a, weight_cuda.T).to(torch.bfloat16)
                selected = full_output[:, spec.column_start:spec.column_start + spec.n]
                runtime_outputs[pairing] = np.ascontiguousarray(
                    selected.float().cpu().numpy(), dtype=np.float32
                )
                del weight_cuda, full_output, selected
            self.captured[spec.case_id] = CapturedCase(
                spec=spec,
                scheduler_timestep=self.current_timestep,
                wrapper_call_index=call_index,
                full_call_m=full_m,
                k=k,
                alpha_a=alpha,
                global_amax=amax,
                a_codes=a_codes,
                a_scale_bytes=a_scale_bytes,
                reconstructed_a=reconstructed_a.cpu(),
                runtime_outputs=runtime_outputs,
            )

        if len(self.captured) == len(self.specs):
            for name in {spec.layer_name for spec in self.specs}:
                self._layers[name].quantizer.real_tile_capture = None

    def _package_manifest(self, pairing: str, case_rows: list[dict], timestamp: str) -> dict:
        sidecar_marker = self.sidecar_hashes[pairing]
        return {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "producer": {"git_commit": self.producer_commit, "hostname": socket.gethostname()},
            "model": {"name": self.model_name, "revision": self.model_revision},
            "transform": {
                "pca_basis_sha256": self.basis_sha256,
                "residual_rotation_mode": "random",
                "residual_rotation_sha256": self.rotation_sha256,
            },
            "quantized_weight_cache_sha256": self.cache_hashes[pairing],
            # Schema v1 has no separate sidecar-provenance field; preserve it
            # losslessly in the unconstrained implementation identifier.
            "quantizer": {
                "implementation": (
                    "DiRotQ existing hardware-e0 activation + existing hardware weight "
                    f"packing sidecar sha256={sidecar_marker}; sorted-index payload "
                    "losslessly transcoded to receiver-v1 sign-magnitude"
                ),
                "version": "real-tile-export-v1",
            },
            "package_creation_timestamp": timestamp,
            "package_size_bytes": 1,
            "cases": case_rows,
        }

    def _write_package(self, pairing: str, package_name: str, timestamp: str) -> tuple[Path, dict]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        target = self.output_root / package_name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing package: {target}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{package_name}.tmp-", dir=self.output_root))
        case_rows = []
        packed_runtime_differences = []
        try:
            for spec in self.specs:
                captured = self.captured[spec.case_id]
                record = self.records[pairing][spec.layer_name]
                a_payload, a_scales = canonical_a(
                    captured.a_codes, captured.a_scale_bytes, spec.m, captured.k
                )
                b_payload, b_scales, alpha_b = sidecar_case_arrays(
                    record, column_start=spec.column_start, n=spec.n, k=captured.k
                )
                arrays = {
                    "a_payload": a_payload,
                    "a_scales": a_scales,
                    "b_payload": b_payload,
                    "b_scales": b_scales,
                }
                arrays["expected_packed_fp32"] = _receiver_expected(
                    self.receiver_root, pairing, arrays,
                    spec.m, spec.n, captured.k, captured.alpha_a, alpha_b,
                )
                arrays["expected_fakequant_runtime"] = captured.runtime_outputs[pairing]
                difference = np.abs(
                    arrays["expected_packed_fp32"] - arrays["expected_fakequant_runtime"]
                )
                packed_runtime_differences.append({
                    "case_id": spec.case_id,
                    "max_absolute_error": float(difference.max()),
                    "mean_absolute_error": float(difference.mean()),
                })
                case_dir = temporary / "cases" / spec.case_id
                case_dir.mkdir(parents=True)
                records = {}
                for role, filename in CASE_FILES.items():
                    path = case_dir / filename
                    array = np.ascontiguousarray(arrays[role])
                    np.save(path, array, allow_pickle=False)
                    relative = path.relative_to(temporary).as_posix()
                    records[role] = _file_record(path, relative, array)
                mp, npad, kp = _ceil(spec.m, 16), _ceil(spec.n, 8), _ceil(captured.k, 64)
                b_format = "e2m1" if pairing == "e0xe2" else "e0m3"
                case_rows.append({
                    "case_id": spec.case_id,
                    "layer_name": spec.layer_name,
                    "prompt_image_id": self.prompt_id,
                    "scheduler_timestep": captured.scheduler_timestep,
                    "wrapper_call_index": captured.wrapper_call_index,
                    "activation_original_dtype": "bfloat16",
                    "weight_reconstructed_dtype": "bfloat16",
                    "pairing": pairing,
                    "M": spec.m, "N": spec.n, "K": captured.k,
                    "Mp": mp, "Np": npad, "Kp": kp,
                    "group_size": GROUP_SIZE,
                    "a_logical_layout": A_LAYOUT,
                    "b_logical_layout": B_LAYOUT,
                    "nibble_order": NIBBLE_ORDER,
                    "a_format": "e0m3",
                    "b_format": b_format,
                    "alpha_A": captured.alpha_a,
                    "alpha_B": alpha_b,
                    "block_scale_encoding": BLOCK_SCALE_ENCODING,
                    "global_scale_semantics": GLOBAL_SCALE_SEMANTICS,
                    "expected_packed_fp32_semantics": PACKED_EXPECTED_SEMANTICS,
                    "expected_fakequant_runtime_semantics": FAKEQUANT_EXPECTED_SEMANTICS,
                    "fakequant_runtime": {
                        "runtime_dtype": "bfloat16",
                        "operand_dequantization": "alpha is applied per operand before runtime cast",
                        "matmul": "torch.matmul on producer using runtime_dtype inputs",
                        "output_cast": "cast matmul output to runtime_dtype",
                        "npy_storage": "float32 exact materialization of runtime output values",
                    },
                    "files": records,
                })
            manifest = self._package_manifest(pairing, case_rows, timestamp)
            _write_manifest_fixed_size(temporary, manifest)
            verification = verify_with_receiver(self.receiver_root, temporary)
            os.replace(temporary, target)
            return target, {
                "receiver": verification,
                "packed_vs_runtime": packed_runtime_differences,
            }
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def finalize(self) -> dict[str, Any]:
        missing = [spec.case_id for spec in self.specs if spec.case_id not in self.captured]
        if missing:
            raise RuntimeError(f"real-tile denoising ended before all cases were captured: {missing}")
        for name in {spec.layer_name for spec in self.specs}:
            self._layers[name].quantizer.real_tile_capture = None
        self._transformer_handle.remove()
        timestamp = datetime.now(timezone.utc).isoformat()
        outputs = {}
        for pairing, name in (("e0xe2", "sana_real_e0xe2_v1"),
                              ("e0xe0", "sana_real_e0xe0_v1")):
            package, diagnostics = self._write_package(pairing, name, timestamp)
            report_path = self.output_root / f"receiver_verification_{pairing}.json"
            strict = verify_with_receiver(self.receiver_root, package, report_path)
            outputs[pairing] = {
                "package": str(package),
                "package_sha256_manifest": sha256_file(package / "manifest.json"),
                "package_size_bytes": strict["package_size_bytes"],
                "case_count": strict["case_count"],
                "receiver_verification": strict,
                **diagnostics,
            }
        e2_root = Path(outputs["e0xe2"]["package"])
        e0_root = Path(outputs["e0xe0"]["package"])
        for spec in self.specs:
            for filename in ("a_payload.npy", "a_scales.npy"):
                if (
                    e2_root.joinpath("cases", spec.case_id, filename).read_bytes()
                    != e0_root.joinpath("cases", spec.case_id, filename).read_bytes()
                ):
                    raise RuntimeError(
                        f"paired packages do not share identical A capture: "
                        f"{spec.case_id}/{filename}"
                    )
        capture_rows = []
        for spec in self.specs:
            item = self.captured[spec.case_id]
            capture_rows.append({
                "case_id": spec.case_id,
                "layer_name": spec.layer_name,
                "scheduler_timestep": item.scheduler_timestep,
                "timestep_index": spec.timestep_index,
                "timestep_occurrence": spec.timestep_occurrence,
                "wrapper_call_index": item.wrapper_call_index,
                "M": spec.m, "N": spec.n, "K": item.k,
                "full_call_M": item.full_call_m,
                "selected_row_start": spec.row_start,
                "selected_column_start": spec.column_start,
                "alpha_A": item.alpha_a,
                "full_low_call_amax": item.global_amax,
                "alpha_scope": "complete pre-padding low-precision region of the quantizer call",
            })
        report = {
            "schema": f"{SCHEMA_NAME}/v{SCHEMA_VERSION}",
            "producer_commit": self.producer_commit,
            "prompt_image_id": self.prompt_id,
            "model_revision": self.model_revision,
            "weight_validation": self.validation,
            "cache_sha256": self.cache_hashes,
            "packing_sidecar_sha256": self.sidecar_hashes,
            "paired_activation_files_identical": True,
            "captures": capture_rows,
            "packages": outputs,
        }
        report_path = self.output_root / "capture_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report


__all__ = [
    "CaptureSpec", "RealTileExportController", "canonical_a", "canonical_b",
    "encode_sign_magnitude", "pack_rows_low_nibble", "sidecar_case_arrays",
    "transcode_sidecar_indices_to_sign_magnitude", "verify_with_receiver",
]
