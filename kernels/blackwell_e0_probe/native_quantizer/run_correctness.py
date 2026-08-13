#!/usr/bin/env python3
"""Byte-level native quantizer and BF16 epilogue correctness runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any

import numpy as np
import torch

from .patch_bf16_gemm import BASELINE_SHA256 as BF16_BASELINE_SHA
from .reference import QuantizedActivation, decode_quantized, native_scales_from_canonical, scalar_quantize, vectorized_quantize
from ..gemm_probe.run_gemm_probe import make_inputs
from ..optimized_gemm.run_correctness import CUBINS as FP32_CUBINS, RUNNER as FP32_RUNNER, write_runner_input


PROBE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROBE_ROOT / "build" / "native_quantizer"
QUANTIZER = BUILD_ROOT / "cmake-build" / "native_fp4_quantizer"
PIPELINE = BUILD_ROOT / "cmake-build" / "native_fp4_pipeline_runner"
BF16_CUBINS = {key: BUILD_ROOT / f"bf16_variant_{key}.cubin" for key in ("00", "01", "11")}
BF16_SHA = {"00": BF16_BASELINE_SHA,
            "01": "e9685307414f0b4319f72e3a291a57bb209c7af8f9850bfb593711fabf6def57",
            "11": "e7a0cc8dfded9dda5417c5b4c117858fb7a3e0419bbd45b4109d2eaced1a305a"}
PAIRING = {("e2m1", "e2m1"): "00", ("e0m3", "e2m1"): "01", ("e0m3", "e0m3"): "11"}


def validate_bf16_cubin(variant: str) -> dict[str, Any]:
    path = BF16_CUBINS[variant]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != BF16_SHA[variant]:
        raise ValueError(f"non-allowlisted BF16 variant {variant}: {actual}")
    baseline, candidate = BF16_CUBINS["00"].read_bytes(), path.read_bytes()
    bits = sum((a ^ b).bit_count() for a, b in zip(baseline, candidate))
    byte_count = sum(a != b for a, b in zip(baseline, candidate))
    expected = (0, 0) if variant == "00" else (64, 64) if variant == "01" else (64, 128)
    if len(candidate) != len(baseline) or (byte_count, bits) != expected:
        raise ValueError("BF16 whole-CUBIN invariant failed")
    return {"variant": variant, "sha256": actual, "size": len(candidate),
            "omma_slots": 64, "changed_bytes": byte_count, "changed_bits": bits}


def source_bytes(source: torch.Tensor) -> bytes:
    return source.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def run_gpu_quantizer(source: torch.Tensor, fmt: str, root: Path, repeat: int = 1) -> tuple[QuantizedActivation, dict[str, Any]]:
    if source.device.type != "cpu" or not source.is_contiguous():
        raise ValueError("source must be contiguous CPU input")
    dtype = "bf16" if source.dtype == torch.bfloat16 else "fp16" if source.dtype == torch.float16 else None
    if dtype is None:
        raise ValueError("source must be BF16 or FP16")
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "activation.bin"
    source_path.write_bytes(source_bytes(source))
    outputs, report = [], {}
    for iteration in range(repeat):
        payload_path, scale_path = root / f"payload_{iteration}.bin", root / f"scale_{iteration}.bin"
        alpha_path, report_path = root / f"alpha_{iteration}.bin", root / f"runner_{iteration}.json"
        subprocess.run([str(QUANTIZER), "--input", str(source_path), "--dtype", dtype,
            "--m", str(source.shape[0]), "--k", str(source.shape[1]), "--format", fmt,
            "--payload-output", str(payload_path), "--scale-output", str(scale_path),
            "--alpha-output", str(alpha_path), "--report", str(report_path),
            "--warmup", "1", "--iterations", "1", "--rounds", "1"], check=True)
        mp, kp = (source.shape[0] + 15) // 16 * 16, (source.shape[1] + 63) // 64 * 64
        outputs.append(QuantizedActivation(source.shape[0], source.shape[1], mp, kp, fmt,
            struct.unpack("<f", alpha_path.read_bytes())[0],
            torch.from_numpy(np.fromfile(payload_path, dtype=np.uint8).copy()).reshape(mp, kp // 2),
            torch.from_numpy(np.fromfile(scale_path, dtype=np.uint8).copy())))
        report = json.loads(report_path.read_text())
    first = outputs[0]
    deterministic = all(struct.pack("<f", first.alpha) == struct.pack("<f", item.alpha)
        and torch.equal(first.payload, item.payload) and torch.equal(first.native_scales, item.native_scales)
        for item in outputs[1:])
    if not deterministic:
        raise AssertionError("native quantizer is nondeterministic")
    report["deterministic"] = deterministic
    return first, report


def cases() -> list[tuple[str, torch.Tensor, str, bool]]:
    generator = torch.Generator().manual_seed(20260813)
    return [
        ("all_zero", torch.zeros((1, 1), dtype=torch.bfloat16), "e0m3", True),
        ("signed_zero", torch.tensor([[0.0, -0.0] * 8]).to(torch.bfloat16), "e2m1", True),
        ("e2_codepoints", torch.tensor([[0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]]).to(torch.float16), "e2m1", True),
        ("e0_codepoints", torch.tensor([[0,1,2,3,4,5,6,7,0,-1,-2,-3,-4,-5,-6,-7]]).to(torch.bfloat16), "e0m3", True),
        ("e4m3_boundaries", torch.tensor([[0,2**-10,3*2**-10,.234375,.25,.265625,447,448,-2**-10,-.234375,-.25,-.265625,-447,-448,1,-1]]).to(torch.float16), "e0m3", True),
        ("tail_e2_bf16", torch.randn((17,65), generator=generator).to(torch.bfloat16), "e2m1", True),
        ("tail_e0_fp16", torch.randn((17,65), generator=generator).to(torch.float16), "e0m3", True),
        ("large_e0_bf16_1952", torch.randn((2048,1952), generator=generator).to(torch.bfloat16), "e0m3", False),
        ("large_e2_fp16_1960", torch.randn((2048,1960), generator=generator).to(torch.float16), "e2m1", False),
    ]


def compare_case(name: str, source: torch.Tensor, fmt: str, use_scalar: bool, root: Path) -> dict[str, Any]:
    vector = vectorized_quantize(source, fmt)
    scalar = scalar_quantize(source, fmt) if use_scalar else None
    gpu, runner = run_gpu_quantizer(source, fmt, root / name, 2 if "tail" in name else 1)
    alpha = struct.pack("<f", gpu.alpha) == struct.pack("<f", vector.alpha)
    payload, scale = torch.equal(gpu.payload, vector.payload), torch.equal(gpu.native_scales, vector.native_scales)
    scalar_match = scalar is None or (struct.pack("<f", scalar.alpha) == struct.pack("<f", vector.alpha)
        and torch.equal(scalar.payload, vector.payload) and torch.equal(scalar.native_scales, vector.native_scales))
    roundtrip = torch.equal(native_scales_from_canonical(gpu.canonical_scales, gpu.mp, gpu.kp), gpu.native_scales)
    codes = torch.empty((gpu.mp, gpu.kp), dtype=torch.uint8)
    codes[:,0::2], codes[:,1::2] = gpu.payload & 15, gpu.payload >> 4
    payload_padding = bool((codes[source.shape[0]:] == 0).all()) and bool((codes[:source.shape[0],source.shape[1]:] == 0).all())
    scale_padding = bool((gpu.canonical_scales[source.shape[0]:] == 0x38).all())
    passed = alpha and payload and scale and scalar_match and roundtrip and payload_padding and scale_padding and runner["passed"]
    return {"name": name, "dtype": str(source.dtype), "format": fmt, "shape": list(source.shape),
            "padded_shape": [gpu.mp,gpu.kp], "alpha": gpu.alpha, "alpha_bitwise": alpha,
            "payload_bitwise": payload, "scale_bitwise": scale, "scalar_vector_match": scalar_match,
            "native_scale_roundtrip": roundtrip, "payload_padding": payload_padding,
            "scale_padding": scale_padding, "canary": runner["payload_canary_ok"] and runner["scale_canary_ok"], "passed": passed}


def epilogue_case(a_format: str, b_format: str, root: Path) -> dict[str, Any]:
    variant = PAIRING[(a_format,b_format)]
    validate_bf16_cubin(variant)
    source = torch.randn((128,128), generator=torch.Generator().manual_seed(900 + int(variant))).to(torch.bfloat16)
    quantized, _ = run_gpu_quantizer(source, a_format, root / f"quant_{variant}")
    static = make_inputs((128,128,128), "row_column_scales", seed=700 + int(variant))
    combined = replace(static, packed_a=quantized.payload, a_scales=quantized.canonical_scales, alpha_a=quantized.alpha)
    static_path, activation_path = root / f"static_{variant}.bin", root / f"activation_{variant}.bin"
    write_runner_input(static_path, combined); activation_path.write_bytes(source_bytes(source))
    output, pipeline_report = root / f"pipeline_{variant}.bf16", root / f"pipeline_{variant}.json"
    subprocess.run([str(PIPELINE), "--activation", str(activation_path), "--static-input", str(static_path),
        "--dtype", "bf16", "--format", a_format, "--cubin", str(BF16_CUBINS[variant]),
        "--output", str(output), "--report", str(pipeline_report), "--warmup", "1", "--iterations", "1", "--rounds", "1"], check=True)
    fp32, cast, debug_report = root / f"debug_{variant}.fp32", root / f"debug_{variant}.bf16", root / f"debug_{variant}.json"
    subprocess.run([str(FP32_RUNNER), "--cubin", str(FP32_CUBINS[variant]), "--input", str(static_path),
        "--output", str(fp32), "--bf16-output", str(cast), "--report", str(debug_report),
        "--warmup", "1", "--iterations", "1", "--rounds", "1"], check=True)
    report = json.loads(pipeline_report.read_text())
    bitwise = output.read_bytes() == cast.read_bytes()
    return {"variant": variant, "pairing": f"{a_format}x{b_format}",
            "bf16_epilogue_equals_debug_fp32_cast": bitwise,
            "canary": report["output_canary_ok"], "passed": bitwise and report["passed"]}


def negatives(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    source = torch.tensor([[100.0,-40.0] + [0.0]*14], dtype=torch.bfloat16)
    q = vectorized_quantize(source, "e0m3")
    wrong = QuantizedActivation(q.m,q.k,q.mp,q.kp,"e2m1",q.alpha,q.payload,q.native_scales)
    selected = torch.cat((torch.ones((1,16)),torch.full((1,16),100.0)),dim=1).to(torch.bfloat16)
    padding_begin = source.shape[1] // 2
    bad_padding = q.payload.clone()
    bad_padding[:, padding_begin] = 1
    padding_caught = bool((q.payload[:, padding_begin:] == 0).all()) and bool(
        (bad_padding[:, padding_begin:] != 0).any())
    rejected = []
    for label,value in (("nan",float("nan")),("inf",float("inf"))):
        path=root/f"{label}.bin"; path.write_bytes(source_bytes(torch.tensor([[value]],dtype=torch.bfloat16)))
        result=subprocess.run([str(QUANTIZER),"--input",str(path),"--dtype","bf16","--m","1","--k","1","--format","e0m3",
            "--payload-output",str(root/"bad.p"),"--scale-output",str(root/"bad.s"),"--alpha-output",str(root/"bad.a"),"--report",str(root/"bad.json")],capture_output=True)
        rejected.append(result.returncode != 0)
    controls = {
        "nibble_swap": not torch.equal((q.payload >> 4) | ((q.payload & 15) << 4),q.payload),
        "e0_payload_e2_decode": not torch.equal(decode_quantized(q),decode_quantized(wrong)),
        "scale_layout_offset": not torch.equal(q.canonical_scales,torch.roll(q.canonical_scales,1,1)),
        "forgot_global_alpha": not torch.equal(decode_quantized(q),decode_quantized(q)/q.alpha),
        "selected_row_global_scale": vectorized_quantize(selected[:,:16].contiguous(),"e0m3").alpha != vectorized_quantize(selected,"e0m3").alpha,
        "nonzero_padding": padding_caught, "nan_inf_rejected": all(rejected),
        "wrong_format_routing": PAIRING[("e0m3","e0m3")] != "01"}
    controls["passed"] = all(controls.values())
    return controls


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",type=Path,default=BUILD_ROOT/"correctness"); parser.add_argument("--report",type=Path,default=BUILD_ROOT/"correctness.json"); args=parser.parse_args()
    quantizer=[compare_case(*case,args.output_root) for case in cases()]
    epilogue=[epilogue_case(*pair,args.output_root/"epilogue") for pair in (("e2m1","e2m1"),("e0m3","e2m1"),("e0m3","e0m3"))]
    negative=negatives(args.output_root/"negative")
    report={"quantizer_cases":quantizer,"bf16_epilogue_cases":epilogue,"negative_controls":negative,
            "bf16_cubins":{variant:validate_bf16_cubin(variant) for variant in BF16_CUBINS}}
    report["passed"]=all(case["passed"] for case in quantizer+epilogue) and negative["passed"]
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":report["passed"],"quantizer_cases":len(quantizer),"bf16_epilogue_cases":len(epilogue),"report":str(args.report.resolve())},indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
