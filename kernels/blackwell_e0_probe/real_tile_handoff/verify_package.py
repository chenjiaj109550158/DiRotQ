#!/usr/bin/env python3
"""Strict, CUDA-free verifier for real DiRotQ FP4 tile package v1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
import torch

from kernels.blackwell_e0_probe.gemm_probe.packing import CanonicalInputs, validate_canonical
from kernels.blackwell_e0_probe.gemm_probe.reference import (
    fp32_comparison_bound,
    sequential_fp32_gemm,
)
from kernels.blackwell_e0_probe.real_tile_handoff.schema import (
    CASE_FILE_NAMES,
    DEFAULT_MAX_PACKAGE_BYTES,
    CaseContract,
    PackageValidationError,
    expected_dtypes,
    expected_shapes,
    validate_manifest_structure,
)


@dataclass(frozen=True)
class VerifiedCase:
    contract: CaseContract
    metadata: dict[str, Any]
    inputs: CanonicalInputs
    expected_packed_fp32: np.ndarray
    expected_fakequant_runtime: np.ndarray
    packed_reference: np.ndarray
    packed_allowed_error: np.ndarray
    packed_reference_metrics: dict[str, Any]


@dataclass(frozen=True)
class VerifiedPackage:
    root: Path
    manifest: dict[str, Any]
    package_size_bytes: int
    cases: tuple[VerifiedCase, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_package_root(package_root: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(package_root)))
    try:
        root_stat = lexical.lstat()
    except FileNotFoundError as error:
        raise PackageValidationError(f"package root does not exist: {lexical}") from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise PackageValidationError("package root must not be a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise PackageValidationError("package root must be a directory")
    return lexical.resolve(strict=True)


def _inventory_regular_files(root: Path) -> dict[str, tuple[Path, int]]:
    files: dict[str, tuple[Path, int]] = {}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if entry.is_symlink():
                    raise PackageValidationError(f"symlink is forbidden: {relative}")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    visit(path)
                elif stat.S_ISREG(info.st_mode):
                    canonical = path.resolve(strict=True)
                    if canonical.parent != root and root not in canonical.parents:
                        raise PackageValidationError(f"path escapes package root: {relative}")
                    files[relative] = (canonical, info.st_size)
                else:
                    raise PackageValidationError(f"non-regular package entry: {relative}")

    visit(root)
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npy(path: Path, *, role: str, expected_dtype: str,
              expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise PackageValidationError(f"unsupported NPY version for {role}: {version}")
    except PackageValidationError:
        raise
    except Exception as error:
        raise PackageValidationError(f"cannot parse NPY header for {role}: {error}") from error
    if dtype.hasobject or dtype.kind == "O":
        raise PackageValidationError(f"object array is forbidden: {role}")
    if dtype.str != expected_dtype:
        raise PackageValidationError(
            f"{role} dtype mismatch: {dtype.str} != {expected_dtype}"
        )
    if tuple(shape) != expected_shape:
        raise PackageValidationError(
            f"{role} shape mismatch: {tuple(shape)} != {expected_shape}"
        )
    if fortran_order:
        raise PackageValidationError(f"{role} must be C-contiguous")
    try:
        with path.open("rb") as handle:
            array = np.load(handle, allow_pickle=False)
    except Exception as error:
        raise PackageValidationError(f"cannot safely load {role}: {error}") from error
    if not isinstance(array, np.ndarray):
        raise PackageValidationError(f"{role} is not a NumPy array")
    if array.dtype.hasobject or array.dtype.kind == "O":
        raise PackageValidationError(f"object array is forbidden: {role}")
    if array.dtype.str != expected_dtype:
        raise PackageValidationError(
            f"{role} dtype mismatch: {array.dtype.str} != {expected_dtype}"
        )
    if tuple(array.shape) != expected_shape:
        raise PackageValidationError(
            f"{role} shape mismatch: {tuple(array.shape)} != {expected_shape}"
        )
    if not array.flags.c_contiguous:
        raise PackageValidationError(f"{role} must be C-contiguous")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise PackageValidationError(f"{role} contains NaN or infinity")
    return array


def _comparison_metrics(actual: torch.Tensor, expected: torch.Tensor,
                        allowed: torch.Tensor) -> dict[str, Any]:
    error = (actual - expected).abs()
    relative = torch.where(
        expected != 0,
        error / expected.abs(),
        torch.where(error == 0, torch.zeros_like(error), torch.full_like(error, float("inf"))),
    )
    mismatch = error > allowed
    coordinates = torch.nonzero(mismatch, as_tuple=False)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float(relative.max()),
        "tolerance_mismatch_count": int(mismatch.sum()),
        "bitwise_mismatch_count": int((actual != expected).sum()),
        "first_mismatch_coordinates": coordinates[:8].tolist(),
        "max_gamma_k_tolerance": float(allowed.max()),
    }


def verify_package(package_root: Path | str, *,
                   max_package_bytes: int = DEFAULT_MAX_PACKAGE_BYTES) -> VerifiedPackage:
    if isinstance(max_package_bytes, bool) or max_package_bytes <= 0:
        raise ValueError("max_package_bytes must be positive")
    root = _canonical_package_root(Path(package_root))
    inventory = _inventory_regular_files(root)
    if "manifest.json" not in inventory:
        raise PackageValidationError("manifest.json is missing")
    actual_package_size = sum(size for _, size in inventory.values())
    if actual_package_size > max_package_bytes:
        raise PackageValidationError(
            f"package exceeds size limit: {actual_package_size} > {max_package_bytes}"
        )

    manifest_path, manifest_size = inventory["manifest.json"]
    if manifest_size > 4 * 1024 * 1024:
        raise PackageValidationError("manifest.json exceeds 4 MiB")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PackageValidationError(f"non-finite JSON number: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PackageValidationError(f"invalid manifest JSON: {error}") from error
    contracts = validate_manifest_structure(manifest)
    if manifest["package_size_bytes"] != actual_package_size:
        raise PackageValidationError(
            "manifest package_size_bytes does not equal the complete package byte size"
        )

    case_metadata = {case["case_id"]: case for case in manifest["cases"]}
    listed_paths: set[str] = set()
    for case in manifest["cases"]:
        for record in case["files"].values():
            path_text = record["path"]
            if path_text.startswith("/") or ".." in Path(path_text).parts:
                raise PackageValidationError(f"unsafe manifest path: {path_text}")
            if path_text in listed_paths:
                raise PackageValidationError(f"manifest path is listed twice: {path_text}")
            listed_paths.add(path_text)
    actual_case_paths = set(inventory) - {"manifest.json"}
    if actual_case_paths != listed_paths:
        raise PackageValidationError(
            f"unlisted or missing case data: extra={sorted(actual_case_paths - listed_paths)}, "
            f"missing={sorted(listed_paths - actual_case_paths)}"
        )

    verified_cases: list[VerifiedCase] = []
    dtypes = expected_dtypes()
    for contract in contracts:
        metadata = case_metadata[contract.case_id]
        shapes = expected_shapes(contract.shape)
        arrays: dict[str, np.ndarray] = {}
        for role in CASE_FILE_NAMES:
            record = metadata["files"][role]
            path, actual_size = inventory[record["path"]]
            if path.parent != root and root not in path.parents:
                raise PackageValidationError(f"case path escaped package root: {record['path']}")
            if actual_size != record["byte_size"]:
                raise PackageValidationError(f"byte size mismatch: {record['path']}")
            if _sha256_file(path) != record["sha256"]:
                raise PackageValidationError(f"SHA-256 mismatch: {record['path']}")
            arrays[role] = _load_npy(
                path,
                role=f"{contract.case_id}.{role}",
                expected_dtype=dtypes[role],
                expected_shape=shapes[role],
            )

        inputs = CanonicalInputs(
            shape=contract.shape,
            packed_a=torch.from_numpy(arrays["a_payload"].copy()),
            packed_b=torch.from_numpy(arrays["b_payload"].copy()),
            a_scales=torch.from_numpy(arrays["a_scales"].copy()),
            b_scales=torch.from_numpy(arrays["b_scales"].copy()),
            alpha_a=contract.alpha_a,
            alpha_b=contract.alpha_b,
        )
        try:
            validate_canonical(inputs)
        except ValueError as error:
            raise PackageValidationError(
                f"{contract.case_id} canonical payload/scale validation failed: {error}"
            ) from error
        _, recomputed = sequential_fp32_gemm(inputs, contract.a_format, contract.b_format)
        expected = torch.from_numpy(arrays["expected_packed_fp32"].copy())
        allowed = fp32_comparison_bound(
            inputs, contract.a_format, contract.b_format, recomputed
        )
        metrics = _comparison_metrics(expected, recomputed, allowed)
        if metrics["tolerance_mismatch_count"]:
            raise PackageValidationError(
                f"{contract.case_id} expected_packed_fp32 disagrees with recomputed reference: "
                f"{metrics}"
            )
        verified_cases.append(VerifiedCase(
            contract=contract,
            metadata=metadata,
            inputs=inputs,
            expected_packed_fp32=arrays["expected_packed_fp32"],
            expected_fakequant_runtime=arrays["expected_fakequant_runtime"],
            packed_reference=recomputed.numpy().copy(),
            packed_allowed_error=allowed.numpy().copy(),
            packed_reference_metrics=metrics,
        ))

    return VerifiedPackage(
        root=root,
        manifest=manifest,
        package_size_bytes=actual_package_size,
        cases=tuple(verified_cases),
    )


def verification_report(package: VerifiedPackage) -> dict[str, Any]:
    return {
        "package_root": str(package.root),
        "schema": package.manifest["schema"],
        "package_size_bytes": package.package_size_bytes,
        "case_count": len(package.cases),
        "cases": [
            {
                "case_id": case.contract.case_id,
                "pairing": case.contract.pairing,
                "shape": [case.contract.shape.m, case.contract.shape.n, case.contract.shape.k],
                "padded_shape": [case.contract.shape.mp, case.contract.shape.np, case.contract.shape.kp],
                "runtime_dtype": case.contract.runtime_dtype,
                "packed_reference": case.packed_reference_metrics,
            }
            for case in package.cases
        ],
        "passed": True,
        "cuda_touched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--max-package-bytes", type=int, default=DEFAULT_MAX_PACKAGE_BYTES)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        package = verify_package(args.package, max_package_bytes=args.max_package_bytes)
        report = verification_report(package)
    except PackageValidationError as error:
        report = {"passed": False, "cuda_touched": False, "error": str(error)}
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text)
        print(text, end="")
        raise SystemExit(1) from error
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()


__all__ = [
    "VerifiedCase", "VerifiedPackage", "verification_report", "verify_package",
]
