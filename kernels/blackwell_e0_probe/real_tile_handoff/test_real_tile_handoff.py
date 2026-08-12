from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from kernels.blackwell_e0_probe.gemm_probe.packing import (
    unpack_canonical_b,
)
from kernels.blackwell_e0_probe.packing import pack_nibbles
from kernels.blackwell_e0_probe.real_tile_handoff.create_synthetic_package import (
    create_synthetic_packages,
)
from kernels.blackwell_e0_probe.real_tile_handoff.run_real_tile import (
    DEFAULT_CUBINS,
    run_package,
    validate_allowlisted_cubin,
)
from kernels.blackwell_e0_probe.real_tile_handoff.schema import (
    DEFAULT_MAX_PACKAGE_BYTES,
    PackageValidationError,
)
from kernels.blackwell_e0_probe.real_tile_handoff.verify_package import verify_package


@pytest.fixture(scope="session")
def synthetic_packages(tmp_path_factory):
    root = tmp_path_factory.mktemp("real_tile_packages")
    e0xe2, e0xe0 = create_synthetic_packages(root)
    return {"e0xe2": e0xe2, "e0xe0": e0xe0}


def _copy_package(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def _write_manifest(root: Path, manifest: dict) -> None:
    path = root / "manifest.json"
    for _ in range(12):
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        actual = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        if manifest["package_size_bytes"] == actual:
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            if sum(item.stat().st_size for item in root.rglob("*") if item.is_file()) == actual:
                return
        manifest["package_size_bytes"] = actual
    raise RuntimeError("test manifest size did not converge")


def _case(manifest: dict, suffix: str) -> dict:
    return next(case for case in manifest["cases"] if case["case_id"].endswith(suffix))


def _refresh_file_record(root: Path, case: dict, role: str) -> None:
    record = case["files"][role]
    path = root / record["path"]
    data = path.read_bytes()
    record["sha256"] = hashlib.sha256(data).hexdigest()
    record["byte_size"] = len(data)


def _mutate_array(root: Path, suffix: str, role: str, mutate) -> None:
    manifest = _manifest(root)
    case = _case(manifest, suffix)
    path = root / case["files"][role]["path"]
    array = np.load(path, allow_pickle=False)
    changed = mutate(array.copy())
    np.save(path, changed, allow_pickle=False)
    _refresh_file_record(root, case, role)
    _write_manifest(root, manifest)


def test_synthetic_packages_cover_contract_and_verify(synthetic_packages):
    e0xe2 = verify_package(synthetic_packages["e0xe2"])
    e0xe0 = verify_package(synthetic_packages["e0xe0"])
    assert {case.contract.pairing for case in e0xe2.cases} == {"e0xe2"}
    assert {case.contract.pairing for case in e0xe0.cases} == {"e0xe0"}
    assert {case.contract.runtime_dtype for case in e0xe2.cases} == {"bfloat16"}
    assert {case.contract.runtime_dtype for case in e0xe0.cases} == {"float16"}
    for package in (e0xe2, e0xe0):
        shapes = {(case.contract.shape.m, case.contract.shape.n, case.contract.shape.k)
                  for case in package.cases}
        assert (16, 8, 64) in shapes
        assert (17, 9, 65) in shapes
        assert (32, 16, 128) in shapes
        assert all(case.packed_reference_metrics["tolerance_mismatch_count"] == 0
                   for case in package.cases)


def test_synthetic_packages_run_on_allowlisted_hardware(synthetic_packages):
    for pairing in ("e0xe2", "e0xe0"):
        report = run_package(synthetic_packages[pairing])
        assert report["passed"]
        assert report["strict_verifier_passed_before_cuda"]
        for case in report["cases"]:
            assert case["hardware_fp32_vs_expected_packed_fp32"]["mismatch_count"] == 0
            assert case["output_guard"]["prefix_canary_ok"]
            assert case["output_guard"]["suffix_canary_ok"]
            assert case["hardware_runtime_cast_vs_expected_fakequant_runtime"][
                "informational_only"
            ]


def test_manifest_hash_mismatch_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_hash")
    manifest = _manifest(root)
    _case(manifest, "aligned_global")["files"]["a_payload"]["sha256"] = "0" * 64
    _write_manifest(root, manifest)
    with pytest.raises(PackageValidationError, match="SHA-256 mismatch"):
        verify_package(root)


def test_actual_dtype_mismatch_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_dtype")

    def mutate(array):
        return array.astype(np.uint16)

    _mutate_array(root, "aligned_global", "a_payload", mutate)
    with pytest.raises(PackageValidationError, match="dtype mismatch"):
        verify_package(root)


def test_actual_shape_mismatch_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_shape")

    def mutate(array):
        return array.reshape(-1)

    _mutate_array(root, "aligned_global", "a_payload", mutate)
    with pytest.raises(PackageValidationError, match="shape mismatch"):
        verify_package(root)


def test_symlink_and_root_symlink_are_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_symlink")
    manifest = _manifest(root)
    case = _case(manifest, "aligned_global")
    victim = root / case["files"]["a_payload"]["path"]
    victim.unlink()
    os.symlink("b_payload.npy", victim)
    with pytest.raises(PackageValidationError, match="symlink"):
        verify_package(root)

    alias = tmp_path / "root_alias"
    os.symlink(synthetic_packages["e0xe2"], alias)
    with pytest.raises(PackageValidationError, match="root must not be a symlink"):
        verify_package(alias)


def test_manifest_path_traversal_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_path")
    manifest = _manifest(root)
    _case(manifest, "aligned_global")["files"]["a_payload"]["path"] = "../escape.npy"
    _write_manifest(root, manifest)
    with pytest.raises(PackageValidationError, match="path must be"):
        verify_package(root)


def test_pickle_object_numpy_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_pickle")
    manifest = _manifest(root)
    case = _case(manifest, "aligned_global")
    path = root / case["files"]["a_payload"]["path"]
    np.save(path, np.array([{"forbidden": True}], dtype=object), allow_pickle=True)
    _refresh_file_record(root, case, "a_payload")
    _write_manifest(root, manifest)
    with pytest.raises(PackageValidationError, match="object array is forbidden"):
        verify_package(root)


def test_nonzero_payload_padding_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_payload_padding")

    def mutate(array):
        array[0, 32] |= np.uint8(0x20)  # K=65: padded k=65 is high nibble.
        return array

    _mutate_array(root, "tail_zero_block", "a_payload", mutate)
    with pytest.raises(PackageValidationError, match="positive-zero"):
        verify_package(root)


def test_scale_padding_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_scale_padding")

    def mutate(array):
        array[0, 5] = np.uint8(0x30)  # K=65 has five valid scale blocks.
        return array

    _mutate_array(root, "tail_zero_block", "a_scales", mutate)
    with pytest.raises(PackageValidationError, match="padding scales must equal one"):
        verify_package(root)


def test_pairing_and_format_disagreement_is_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_format")
    manifest = _manifest(root)
    _case(manifest, "aligned_global")["a_format"] = "e2m1"
    _write_manifest(root, manifest)
    with pytest.raises(PackageValidationError, match="pairing and payload formats disagree"):
        verify_package(root)


def test_wrong_b_row_major_packing_is_numerically_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "bad_b_layout")
    package = verify_package(root)
    verified = next(case for case in package.cases
                    if case.contract.case_id.endswith("multistep_row_column_scales"))
    shape = verified.contract.shape
    logical_b = unpack_canonical_b(verified.inputs.packed_b, shape)[:shape.k, :shape.n]
    wrong = pack_nibbles(logical_b.contiguous().reshape(-1)).reshape(
        shape.np, shape.kp // 2
    ).numpy()

    def mutate(_array):
        return wrong

    _mutate_array(root, "multistep_row_column_scales", "b_payload", mutate)
    with pytest.raises(PackageValidationError, match="expected_packed_fp32 disagrees"):
        verify_package(root)


def test_nibble_swap_is_numerically_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe0"], tmp_path / "bad_nibble")

    def mutate(array):
        return ((array & np.uint8(0x0F)) << np.uint8(4)) | (array >> np.uint8(4))

    _mutate_array(root, "multistep_row_column_scales", "a_payload", mutate)
    with pytest.raises(PackageValidationError, match="expected_packed_fp32 disagrees"):
        verify_package(root)


@pytest.mark.parametrize("mode", ["missing", "nan"])
def test_missing_or_nan_global_scale_is_rejected(synthetic_packages, tmp_path, mode):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / f"bad_alpha_{mode}")
    manifest = _manifest(root)
    case = _case(manifest, "aligned_global")
    if mode == "missing":
        del case["alpha_A"]
        expected = "keys differ"
    else:
        case["alpha_A"] = float("nan")
        expected = "non-finite JSON number"
    _write_manifest(root, manifest)
    with pytest.raises(PackageValidationError, match=expected):
        verify_package(root)


def test_modified_expected_output_is_semantically_rejected(synthetic_packages, tmp_path):
    root = _copy_package(synthetic_packages["e0xe0"], tmp_path / "bad_expected")

    def mutate(array):
        array[0, 0] += np.float32(32.0)
        return array

    _mutate_array(root, "aligned_global", "expected_packed_fp32", mutate)
    with pytest.raises(PackageValidationError, match="expected_packed_fp32 disagrees"):
        verify_package(root)


def test_size_limit_and_unlisted_data_are_rejected(synthetic_packages, tmp_path):
    source = synthetic_packages["e0xe2"]
    size = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    assert size < DEFAULT_MAX_PACKAGE_BYTES
    with pytest.raises(PackageValidationError, match="exceeds size limit"):
        verify_package(source, max_package_bytes=size - 1)

    root = _copy_package(source, tmp_path / "unlisted")
    (root / "cases" / "unlisted.npy").write_bytes(b"not listed")
    manifest = _manifest(root)
    _write_manifest(root, manifest)
    with pytest.raises(PackageValidationError, match="unlisted or missing case data"):
        verify_package(root)


def test_non_allowlisted_cubin_is_rejected_before_load(tmp_path):
    corrupt = tmp_path / "corrupt.cubin"
    data = bytearray(DEFAULT_CUBINS["01"].read_bytes())
    data[100] ^= 1
    corrupt.write_bytes(data)
    with pytest.raises(ValueError, match="non-allowlisted"):
        validate_allowlisted_cubin(corrupt, "01")


def test_invalid_package_stops_before_cuda_initialization(
    synthetic_packages, tmp_path, monkeypatch
):
    root = _copy_package(synthetic_packages["e0xe2"], tmp_path / "pre_cuda_stop")
    manifest = _manifest(root)
    _case(manifest, "aligned_global")["alpha_A"] = 0.0
    _write_manifest(root, manifest)

    def forbidden_cuda():
        raise AssertionError("CUDA initialization was reached")

    monkeypatch.setattr(
        "kernels.blackwell_e0_probe.real_tile_handoff.run_real_tile.CudaDriver",
        forbidden_cuda,
    )
    with pytest.raises(PackageValidationError, match="finite and positive"):
        run_package(root)
