#!/usr/bin/env python3
"""Materialize ignored audit/summary files from saved native-quantizer results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from .patch_bf16_gemm import KERNEL_SYMBOL, OMMA_OFFSETS
from .run_correctness import BF16_CUBINS, BUILD_ROOT, validate_bf16_cubin


def main() -> None:
    correctness=json.loads((BUILD_ROOT/"correctness.json").read_text())
    benchmark=json.loads((BUILD_ROOT/"benchmark.json").read_text())
    nvdisasm=Path("/home/chenjiaj/.conda/envs/blackwell-e0-probe/bin/nvdisasm")
    audit={}
    excerpts={}
    for variant,path in BF16_CUBINS.items():
        text=subprocess.run([str(nvdisasm),"-c",str(path)],check=True,text=True,capture_output=True).stdout
        audit[variant]=validate_bf16_cubin(variant)|{
            "kernel_symbol":KERNEL_SYMBOL,"text_size":0x6780,"target":"sm_120a",
            "decoded_omma_count":text.count("OMMA.SF.16864"),
            "ffma_count":text.count("FFMA"),"hmma_count":text.count("HMMA"),"imma_count":text.count("IMMA"),
            "operand_format_bits":[0,0] if variant=="00" else [1,0] if variant=="01" else [1,1]}
        excerpts[variant]=[line.strip() for line in text.splitlines() if "OMMA" in line][:4]
    (BUILD_ROOT/"binary_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    (BUILD_ROOT/"disassembly_excerpts.json").write_text(json.dumps(excerpts,indent=2,sort_keys=True)+"\n")
    sanitizer={path.stem:next((line.strip() for line in path.read_text().splitlines()
        if "ERROR SUMMARY" in line or "RACECHECK SUMMARY" in line),"")
        for path in sorted((BUILD_ROOT/"sanitizer").glob("*.log"))}
    sanitizer["passed"]=all("0 errors" in value or "0 hazards" in value for key,value in sanitizer.items() if key!="passed")
    (BUILD_ROOT/"sanitizer_summary.json").write_text(json.dumps(sanitizer,indent=2,sort_keys=True)+"\n")
    layout="""# Native activation layout contract

- Input: finite contiguous row-major BF16/FP16 `[M,K]`.
- Payload: CUTLASS-native row-major FP4 A `[Mp,Kp/2]`, earlier K in low nibble.
- Scale: direct `Sm1xxBlockScaledConfig<16>` SFA atom output; no canonical intermediate.
- Atom: 128 rows x 64 K, 512 UE4M3 bytes; scale offset is documented in the tracked README.
- Alpha: device FP32 scalar from the full `[M,K]` amax; a second device scalar holds `alpha_A*alpha_B`.
- Runtime launches: absmax reduction, alpha finalize, quantize/pack, then optimized GEMM.
- Padding: positive-zero payload and literal UE4M3 one (`0x38`).
"""
    (BUILD_ROOT/"layout_contract.md").write_text(layout)
    lines=["# Native FP4 low-branch feasibility summary","",f"Classification: `{benchmark['classification']}`.","",
           f"Correctness: {len(correctness['quantizer_cases'])} quantizer cases and {len(correctness['bf16_epilogue_cases'])} BF16 epilogue cases passed.","",
           "| MNK | pairing | quant ms | GEMM BF16 ms | full ms | vs BF16 | GB/s |","|---|---|---:|---:|---:|---:|---:|"]
    for shape in benchmark["shapes"]:
        for variant in ("00","01","11"):
            item=shape["pipelines"][variant]
            lines.append(f"| {'x'.join(map(str,shape['shape']))} | {item['pairing']} | {item['full_activation_quantization']['median_ms']:.9f} | {item['gemm_bf16']['median_ms']:.9f} | {item['full_low_branch']['median_ms']:.9f} | {item['full_pipeline_vs_bf16_ratio']:.6f} | {item['quantizer_effective_gbps']:.2f} |")
    lines += ["","CUDA-event results use normal sequential launches, 100 warmups, 500 iterations/round, and 5 rounds.",
              "This is not a complete DiRotQ layer or model speedup."]
    (BUILD_ROOT/"summary_report.md").write_text("\n".join(lines)+"\n")
    commands="""cmake -S kernels/blackwell_e0_probe/native_quantizer -B kernels/blackwell_e0_probe/build/native_quantizer/cmake-build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc -DCUTLASS_ROOT=$PWD/kernels/blackwell_e0_probe/build/cutlass-v4.0.0
cmake --build kernels/blackwell_e0_probe/build/native_quantizer/cmake-build -j4
python -m kernels.blackwell_e0_probe.native_quantizer.patch_bf16_gemm BUILD/bf16_baseline_00.cubin BUILD/bf16_variant_01.cubin --variant 01
python -m kernels.blackwell_e0_probe.native_quantizer.patch_bf16_gemm BUILD/bf16_baseline_00.cubin BUILD/bf16_variant_11.cubin --variant 11
CUDA_VISIBLE_DEVICES=0 python -m kernels.blackwell_e0_probe.native_quantizer.run_correctness
CUDA_VISIBLE_DEVICES=0 python -m kernels.blackwell_e0_probe.native_quantizer.benchmark_pipeline --warmup 100 --iterations 500 --rounds 5
CUDA_VISIBLE_DEVICES=0 pytest -q kernels/blackwell_e0_probe/native_quantizer/test_native_quantizer.py
"""
    (BUILD_ROOT/"commands.txt").write_text(commands)
    print(json.dumps({"classification":benchmark["classification"],"binary_variants":len(audit),"sanitizer_passed":sanitizer["passed"]},indent=2))


if __name__=="__main__": main()
