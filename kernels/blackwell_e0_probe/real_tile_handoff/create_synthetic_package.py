#!/usr/bin/env python3
"""Create contract-only E0xE2 and E0xE0 packages; never model artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from kernels.blackwell_e0_probe.gemm_probe.reference import (
    decoded_operands_and_scales,
    sequential_fp32_gemm,
)
from kernels.blackwell_e0_probe.gemm_probe.run_gemm_probe import make_inputs
from kernels.blackwell_e0_probe.real_tile_handoff.schema import (
    A_LAYOUT,
    BLOCK_SCALE_ENCODING,
    B_LAYOUT,
    CASE_FILE_NAMES,
    FAKEQUANT_EXPECTED_SEMANTICS,
    GLOBAL_SCALE_SEMANTICS,
    GROUP_SIZE,
    NIBBLE_ORDER,
    PACKED_EXPECTED_SEMANTICS,
    PAIRING_FORMATS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_CASES = (
    ("aligned_global", (16, 8, 64), "global_scales"),
    ("tail_zero_block", (17, 9, 65), "zero_scales"),
    ("multistep_row_column_scales", (32, 16, 128), "row_column_scales"),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _runtime_expected(inputs, a_format: str, b_format: str,
                      runtime_dtype: str) -> np.ndarray:
    a, b, a_scale, b_scale = decoded_operands_and_scales(inputs, a_format, b_format)
    a_dequant = torch.from_numpy(
        np.multiply(a, a_scale, dtype=np.float32)
    ) * float(np.float32(inputs.alpha_a))
    b_dequant = torch.from_numpy(
        np.multiply(b, b_scale, dtype=np.float32)
    ) * float(np.float32(inputs.alpha_b))
    dtype = torch.bfloat16 if runtime_dtype == "bfloat16" else torch.float16
    runtime = torch.matmul(a_dequant.to(dtype), b_dequant.to(dtype)).to(dtype)
    return np.ascontiguousarray(runtime.float().numpy(), dtype=np.float32)


def _file_record(path: Path, relative: str, array: np.ndarray) -> dict:
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_size": path.stat().st_size,
    }


def _write_manifest_fixed_size(root: Path, manifest: dict) -> None:
    path = root / "manifest.json"
    previous = None
    for _ in range(12):
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        actual = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        if manifest["package_size_bytes"] == actual and previous == actual:
            return
        manifest["package_size_bytes"] = actual
        previous = actual
    raise RuntimeError("package_size_bytes did not converge")


def create_package(root: Path, pairing: str, runtime_dtype: str, *,
                   timestamp: str | None = None) -> Path:
    if pairing not in PAIRING_FORMATS:
        raise ValueError("synthetic package pairing must be e0xe2 or e0xe0")
    if runtime_dtype not in {"bfloat16", "float16"}:
        raise ValueError("runtime_dtype must be bfloat16 or float16")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty package root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    a_format, b_format, _ = PAIRING_FORMATS[pairing]
    cases = []
    for index, (suffix, dimensions, pattern) in enumerate(SYNTHETIC_CASES):
        case_id = f"synthetic_{pairing}_{suffix}"
        inputs = make_inputs(dimensions, pattern, seed=20260814 + index * 101)
        _, expected = sequential_fp32_gemm(inputs, a_format, b_format)
        arrays = {
            "a_payload": np.ascontiguousarray(inputs.packed_a.numpy(), dtype=np.uint8),
            "a_scales": np.ascontiguousarray(inputs.a_scales.numpy(), dtype=np.uint8),
            "b_payload": np.ascontiguousarray(inputs.packed_b.numpy(), dtype=np.uint8),
            "b_scales": np.ascontiguousarray(inputs.b_scales.numpy(), dtype=np.uint8),
            "expected_packed_fp32": np.ascontiguousarray(expected.numpy(), dtype=np.float32),
            "expected_fakequant_runtime": _runtime_expected(
                inputs, a_format, b_format, runtime_dtype
            ),
        }
        case_directory = root / "cases" / case_id
        case_directory.mkdir(parents=True)
        records = {}
        for role, filename in CASE_FILE_NAMES.items():
            path = case_directory / filename
            np.save(path, arrays[role], allow_pickle=False)
            relative = path.relative_to(root).as_posix()
            records[role] = _file_record(path, relative, arrays[role])
        shape = inputs.shape
        cases.append({
            "case_id": case_id,
            "layer_name": f"synthetic.contract.layer.{index}",
            "prompt_image_id": f"synthetic-contract-only-{index}",
            "scheduler_timestep": index,
            "wrapper_call_index": index,
            "activation_original_dtype": runtime_dtype,
            "weight_reconstructed_dtype": runtime_dtype,
            "pairing": pairing,
            "M": shape.m,
            "N": shape.n,
            "K": shape.k,
            "Mp": shape.mp,
            "Np": shape.np,
            "Kp": shape.kp,
            "group_size": GROUP_SIZE,
            "a_logical_layout": A_LAYOUT,
            "b_logical_layout": B_LAYOUT,
            "nibble_order": NIBBLE_ORDER,
            "a_format": a_format,
            "b_format": b_format,
            "alpha_A": float(np.float32(inputs.alpha_a)),
            "alpha_B": float(np.float32(inputs.alpha_b)),
            "block_scale_encoding": BLOCK_SCALE_ENCODING,
            "global_scale_semantics": GLOBAL_SCALE_SEMANTICS,
            "expected_packed_fp32_semantics": PACKED_EXPECTED_SEMANTICS,
            "expected_fakequant_runtime_semantics": FAKEQUANT_EXPECTED_SEMANTICS,
            "fakequant_runtime": {
                "runtime_dtype": runtime_dtype,
                "operand_dequantization": "alpha is applied per operand before runtime cast",
                "matmul": "torch.matmul on producer using runtime_dtype inputs",
                "output_cast": "cast matmul output to runtime_dtype",
                "npy_storage": "float32 exact materialization of runtime output values",
            },
            "files": records,
        })
    manifest = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "producer": {"git_commit": _git_commit()},
        "model": {
            "name": "__synthetic_contract_only__",
            "revision": "no-model-was-loaded",
        },
        "transform": {
            "pca_basis_sha256": _sha256_bytes(b"synthetic-pca-basis-marker"),
            "residual_rotation_mode": "synthetic_identity",
            "residual_rotation_sha256": _sha256_bytes(b"synthetic-identity-rotation-marker"),
        },
        "quantized_weight_cache_sha256": _sha256_bytes(b"synthetic-weight-cache-marker"),
        "quantizer": {
            "implementation": "real_tile_handoff.create_synthetic_package",
            "version": "1",
        },
        "package_creation_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "package_size_bytes": 1,
        "cases": cases,
    }
    _write_manifest_fixed_size(root, manifest)
    return root


def create_synthetic_packages(output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    return (
        create_package(output_root / "e0xe2", "e0xe2", "bfloat16", timestamp=timestamp),
        create_package(output_root / "e0xe0", "e0xe0", "float16", timestamp=timestamp),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    packages = create_synthetic_packages(args.output_root)
    print(json.dumps({"packages": [str(path.resolve()) for path in packages]}, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["SYNTHETIC_CASES", "create_package", "create_synthetic_packages"]
