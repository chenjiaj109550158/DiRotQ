"""Schema constants and structural validation for real FP4 tile package v1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Any

import numpy as np

from kernels.blackwell_e0_probe.gemm_probe.packing import GemmShape


SCHEMA_NAME = "dirotq.blackwell.real_fp4_tile"
SCHEMA_VERSION = 1
DEFAULT_MAX_PACKAGE_BYTES = 256 * 1024 * 1024
GROUP_SIZE = 16
PAIRING_FORMATS = {
    "e0xe2": ("e0m3", "e2m1", "01"),
    "e0xe0": ("e0m3", "e0m3", "11"),
}
CASE_FILE_NAMES = {
    "a_payload": "a_payload.npy",
    "a_scales": "a_scales.npy",
    "b_payload": "b_payload.npy",
    "b_scales": "b_scales.npy",
    "expected_packed_fp32": "expected_packed_fp32.npy",
    "expected_fakequant_runtime": "expected_fakequant_runtime.npy",
}
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
RUNTIME_DTYPES = {"bfloat16", "float16"}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PackageValidationError(ValueError):
    """The package is unsafe or violates the frozen receiver contract."""


@dataclass(frozen=True)
class CaseContract:
    case_id: str
    shape: GemmShape
    pairing: str
    a_format: str
    b_format: str
    variant: str
    alpha_a: float
    alpha_b: float
    runtime_dtype: str


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackageValidationError(f"{where} must be an object")
    return value


def _exact_keys(value: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise PackageValidationError(
            f"{where} keys differ: missing={missing}, extra={extra}"
        )


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise PackageValidationError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PackageValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PackageValidationError(f"{where} must be a lowercase SHA-256")
    return value


def _fp32_positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageValidationError(f"{where} must be an FP32 number")
    converted = np.float32(value)
    if not math.isfinite(float(converted)) or float(converted) <= 0:
        raise PackageValidationError(f"{where} must be finite and positive")
    if float(converted) != float(value):
        raise PackageValidationError(f"{where} must be serialized as an exact FP32 value")
    return float(converted)


def expected_shapes(shape: GemmShape) -> dict[str, tuple[int, ...]]:
    return {
        "a_payload": (shape.mp, shape.kp // 2),
        "a_scales": (shape.mp, shape.kp // GROUP_SIZE),
        "b_payload": (shape.np, shape.kp // 2),
        "b_scales": (shape.np, shape.kp // GROUP_SIZE),
        "expected_packed_fp32": (shape.m, shape.n),
        "expected_fakequant_runtime": (shape.m, shape.n),
    }


def expected_dtypes() -> dict[str, str]:
    return {
        "a_payload": np.dtype(np.uint8).str,
        "a_scales": np.dtype(np.uint8).str,
        "b_payload": np.dtype(np.uint8).str,
        "b_scales": np.dtype(np.uint8).str,
        "expected_packed_fp32": np.dtype(np.float32).str,
        "expected_fakequant_runtime": np.dtype(np.float32).str,
    }


def validate_manifest_structure(manifest: Any) -> list[CaseContract]:
    root = _mapping(manifest, "manifest")
    root_keys = {
        "schema", "producer", "model", "transform", "quantized_weight_cache_sha256",
        "quantizer", "package_creation_timestamp", "package_size_bytes", "cases",
    }
    _exact_keys(root, root_keys, "manifest")

    schema = _mapping(root["schema"], "schema")
    _exact_keys(schema, {"name", "version"}, "schema")
    if schema["name"] != SCHEMA_NAME or schema["version"] != SCHEMA_VERSION:
        raise PackageValidationError("unsupported package schema name/version")

    producer = _mapping(root["producer"], "producer")
    if set(producer) not in ({"git_commit"}, {"git_commit", "hostname"}):
        raise PackageValidationError("producer permits only git_commit and optional hostname")
    commit = _nonempty_string(producer["git_commit"], "producer.git_commit")
    if not _COMMIT_RE.fullmatch(commit):
        raise PackageValidationError("producer.git_commit must be a full lowercase Git SHA")
    if "hostname" in producer:
        _nonempty_string(producer["hostname"], "producer.hostname")

    model = _mapping(root["model"], "model")
    _exact_keys(model, {"name", "revision"}, "model")
    _nonempty_string(model["name"], "model.name")
    _nonempty_string(model["revision"], "model.revision")

    transform = _mapping(root["transform"], "transform")
    _exact_keys(
        transform,
        {"pca_basis_sha256", "residual_rotation_mode", "residual_rotation_sha256"},
        "transform",
    )
    _sha256(transform["pca_basis_sha256"], "transform.pca_basis_sha256")
    _nonempty_string(transform["residual_rotation_mode"], "transform.residual_rotation_mode")
    _sha256(transform["residual_rotation_sha256"], "transform.residual_rotation_sha256")
    _sha256(root["quantized_weight_cache_sha256"], "quantized_weight_cache_sha256")

    quantizer = _mapping(root["quantizer"], "quantizer")
    _exact_keys(quantizer, {"implementation", "version"}, "quantizer")
    _nonempty_string(quantizer["implementation"], "quantizer.implementation")
    _nonempty_string(quantizer["version"], "quantizer.version")
    timestamp = _nonempty_string(
        root["package_creation_timestamp"], "package_creation_timestamp"
    )
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise PackageValidationError(
            "package_creation_timestamp must be ISO-8601"
        ) from error
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise PackageValidationError(
            "package_creation_timestamp must include a UTC offset"
        )
    _integer(root["package_size_bytes"], "package_size_bytes", minimum=1)

    cases = root["cases"]
    if not isinstance(cases, list) or not cases:
        raise PackageValidationError("cases must be a non-empty array")
    contracts: list[CaseContract] = []
    seen_ids: set[str] = set()
    case_keys = {
        "case_id", "layer_name", "prompt_image_id", "scheduler_timestep",
        "wrapper_call_index", "activation_original_dtype", "weight_reconstructed_dtype",
        "pairing", "M", "N", "K", "Mp", "Np", "Kp", "group_size",
        "a_logical_layout", "b_logical_layout", "nibble_order", "a_format", "b_format",
        "alpha_A", "alpha_B", "block_scale_encoding", "global_scale_semantics",
        "expected_packed_fp32_semantics", "expected_fakequant_runtime_semantics",
        "fakequant_runtime", "files",
    }
    for index, raw_case in enumerate(cases):
        where = f"cases[{index}]"
        case = _mapping(raw_case, where)
        _exact_keys(case, case_keys, where)
        case_id = _nonempty_string(case["case_id"], f"{where}.case_id")
        if not _CASE_ID_RE.fullmatch(case_id) or case_id in seen_ids:
            raise PackageValidationError(f"{where}.case_id is invalid or duplicated")
        seen_ids.add(case_id)
        _nonempty_string(case["layer_name"], f"{where}.layer_name")
        _nonempty_string(case["prompt_image_id"], f"{where}.prompt_image_id")
        timestep = case["scheduler_timestep"]
        if isinstance(timestep, bool) or not isinstance(timestep, (int, float)) or not math.isfinite(float(timestep)):
            raise PackageValidationError(f"{where}.scheduler_timestep must be finite")
        _integer(case["wrapper_call_index"], f"{where}.wrapper_call_index")

        activation_dtype = _nonempty_string(
            case["activation_original_dtype"], f"{where}.activation_original_dtype"
        )
        weight_dtype = _nonempty_string(
            case["weight_reconstructed_dtype"], f"{where}.weight_reconstructed_dtype"
        )
        pairing = case["pairing"]
        if pairing not in PAIRING_FORMATS:
            raise PackageValidationError(f"{where}.pairing must be e0xe2 or e0xe0")
        a_format, b_format, variant = PAIRING_FORMATS[pairing]
        if case["a_format"] != a_format or case["b_format"] != b_format:
            raise PackageValidationError(f"{where} pairing and payload formats disagree")

        shape = GemmShape(
            _integer(case["M"], f"{where}.M", minimum=1),
            _integer(case["N"], f"{where}.N", minimum=1),
            _integer(case["K"], f"{where}.K", minimum=1),
        )
        if (case["Mp"], case["Np"], case["Kp"]) != (shape.mp, shape.np, shape.kp):
            raise PackageValidationError(f"{where} padded dimensions are not canonical")
        if case["group_size"] != GROUP_SIZE:
            raise PackageValidationError(f"{where}.group_size must be {GROUP_SIZE}")
        if case["a_logical_layout"] != A_LAYOUT or case["b_logical_layout"] != B_LAYOUT:
            raise PackageValidationError(f"{where} logical layout is not canonical")
        if case["nibble_order"] != NIBBLE_ORDER:
            raise PackageValidationError(f"{where} nibble order is not canonical")
        if case["block_scale_encoding"] != BLOCK_SCALE_ENCODING:
            raise PackageValidationError(f"{where} block scale encoding is not UE4M3")
        if case["global_scale_semantics"] != GLOBAL_SCALE_SEMANTICS:
            raise PackageValidationError(f"{where} global scale semantics differ")
        if case["expected_packed_fp32_semantics"] != PACKED_EXPECTED_SEMANTICS:
            raise PackageValidationError(f"{where} packed reference semantics differ")
        if case["expected_fakequant_runtime_semantics"] != FAKEQUANT_EXPECTED_SEMANTICS:
            raise PackageValidationError(f"{where} fake-quant semantics differ")
        alpha_a = _fp32_positive(case["alpha_A"], f"{where}.alpha_A")
        alpha_b = _fp32_positive(case["alpha_B"], f"{where}.alpha_B")

        runtime = _mapping(case["fakequant_runtime"], f"{where}.fakequant_runtime")
        _exact_keys(
            runtime,
            {"runtime_dtype", "operand_dequantization", "matmul", "output_cast", "npy_storage"},
            f"{where}.fakequant_runtime",
        )
        runtime_dtype = runtime["runtime_dtype"]
        if runtime_dtype not in RUNTIME_DTYPES:
            raise PackageValidationError(f"{where} runtime dtype must be bfloat16 or float16")
        if activation_dtype != runtime_dtype or weight_dtype != runtime_dtype:
            raise PackageValidationError(f"{where} runtime and reconstructed operand dtypes disagree")
        if runtime["operand_dequantization"] != "alpha is applied per operand before runtime cast":
            raise PackageValidationError(f"{where} operand dequantization semantics differ")
        if runtime["matmul"] != "torch.matmul on producer using runtime_dtype inputs":
            raise PackageValidationError(f"{where} matmul semantics differ")
        if runtime["output_cast"] != "cast matmul output to runtime_dtype":
            raise PackageValidationError(f"{where} output cast semantics differ")
        if runtime["npy_storage"] != "float32 exact materialization of runtime output values":
            raise PackageValidationError(f"{where} runtime NPY storage semantics differ")

        files = _mapping(case["files"], f"{where}.files")
        _exact_keys(files, set(CASE_FILE_NAMES), f"{where}.files")
        shapes = expected_shapes(shape)
        dtypes = expected_dtypes()
        for role, filename in CASE_FILE_NAMES.items():
            record = _mapping(files[role], f"{where}.files.{role}")
            _exact_keys(record, {"path", "sha256", "dtype", "shape", "byte_size"}, f"{where}.files.{role}")
            expected_path = f"cases/{case_id}/{filename}"
            if record["path"] != expected_path:
                raise PackageValidationError(f"{where}.files.{role}.path must be {expected_path}")
            _sha256(record["sha256"], f"{where}.files.{role}.sha256")
            if record["dtype"] != dtypes[role]:
                raise PackageValidationError(f"{where}.files.{role}.dtype is not canonical")
            if record["shape"] != list(shapes[role]):
                raise PackageValidationError(f"{where}.files.{role}.shape is not canonical")
            _integer(record["byte_size"], f"{where}.files.{role}.byte_size", minimum=1)

        contracts.append(CaseContract(
            case_id=case_id,
            shape=shape,
            pairing=pairing,
            a_format=a_format,
            b_format=b_format,
            variant=variant,
            alpha_a=alpha_a,
            alpha_b=alpha_b,
            runtime_dtype=runtime_dtype,
        ))
    return contracts


__all__ = [
    "A_LAYOUT", "BLOCK_SCALE_ENCODING", "B_LAYOUT", "CASE_FILE_NAMES",
    "CaseContract", "DEFAULT_MAX_PACKAGE_BYTES", "FAKEQUANT_EXPECTED_SEMANTICS",
    "GLOBAL_SCALE_SEMANTICS", "GROUP_SIZE", "NIBBLE_ORDER", "PACKED_EXPECTED_SEMANTICS",
    "PAIRING_FORMATS", "PackageValidationError", "RUNTIME_DTYPES", "SCHEMA_NAME",
    "SCHEMA_VERSION", "expected_dtypes", "expected_shapes", "validate_manifest_structure",
]
