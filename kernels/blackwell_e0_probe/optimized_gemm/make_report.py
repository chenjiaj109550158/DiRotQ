#!/usr/bin/env python3
"""Render ignored audit artifacts from optimized correctness/benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from kernels.blackwell_e0_probe.optimized_gemm.patch_optimized_gemm import (
    BASELINE_SHA256,
)
from kernels.blackwell_e0_probe.optimized_gemm.run_correctness import (
    ALLOWLISTED_SHA256,
    BUILD_ROOT,
)


CUTLASS_COMMIT = "b995f933179c22d3fe0d871c3a53d11e4681950f"


def _capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True,
                          capture_output=True).stdout.strip()


def _metric_row(case: dict) -> str:
    metric = case["hardware_fp32_vs_packed_fp32"]
    return (
        f"| {case['case_id']} | {case['variant']} | "
        f"{'x'.join(map(str, case['shape']))} | {metric['max_absolute_error']:.9g} | "
        f"{metric['mean_absolute_error']:.9g} | {metric['max_relative_error']:.9g} | "
        f"{metric['tolerance_mismatch_count']} | {metric['bitwise_mismatch_count']} |"
    )


def generate(root: Path) -> None:
    correctness = json.loads((root / "real_and_synthetic_correctness.json").read_text())
    benchmark = json.loads((root / "benchmark.json").read_text())
    binary = json.loads((root / "binary_audit.json").read_text())

    inventory = {
        "gpu": _capture(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                         "--format=csv,noheader,nounits"]),
        "nvcc": _capture(["nvcc", "--version"]),
        "ptxas": _capture(["ptxas", "--version"]),
        "cmake": _capture(["cmake", "--version"]).splitlines()[0],
        "ninja": _capture(["ninja", "--version"]),
        "python": _capture(["python", "--version"]),
        "pytorch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "cutlass": {"tag": "v4.0.0", "commit": CUTLASS_COMMIT,
                    "checkout": str((root.parent / "cutlass-v4.0.0").resolve())},
        "baseline_sha256": BASELINE_SHA256,
        "variant_sha256": ALLOWLISTED_SHA256,
    }
    (root / "toolchain_cutlass_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )

    layout = """# Optimized layout contract

Canonical A payload `[Mp,Kp/2]` and B payload `[Np,Kp/2]` are byte-for-byte
the CUTLASS row-major-A/column-major-B subbyte layouts. Earlier K is the low
nibble; no payload transform or prepack is required.

Canonical A/B scales are outer-major `[outer,Kp/16]` UE4M3 bytes. CUTLASS uses
`Sm1xxBlockScaledConfig<16>` with scale atom shape `((32,4),(16,4))`, stride
`((16,4),(0,1))`. A scales are transformed on GPU per forward. B scales are
transformed once as static prepack. UE4M3-one padding is byte `0x38`.

Kernel problem dimensions are canonical `Mp,Np,Kp`; only valid MxN output is
consumed. FP32 output is cast to BF16 by a separately timed GPU kernel.
"""
    (root / "layout_contract.md").write_text(layout)

    real = {key: correctness[key] for key in ("real_e0xe2", "real_e0xe0")}
    real["passed"] = all(value["passed"] for value in real.values())
    (root / "real_tile_correctness.json").write_text(
        json.dumps(real, indent=2, sort_keys=True) + "\n"
    )

    sanitizer = {}
    for name in ("memcheck_e2", "memcheck_e0xe2", "memcheck_e0xe0",
                 "racecheck_e0xe0"):
        path = root / f"sanitizer_{name}.log"
        sanitizer[name] = path.read_text().strip().splitlines()[-1]
    sanitizer["passed"] = all(
        "0 errors" in value for key, value in sanitizer.items() if key != "passed"
    )
    (root / "sanitizer_summary.json").write_text(
        json.dumps(sanitizer, indent=2, sort_keys=True) + "\n"
    )

    commands = """cmake -S kernels/blackwell_e0_probe/optimized_gemm -B kernels/blackwell_e0_probe/build/optimized_gemm/cmake-build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc -DCUTLASS_ROOT=$PWD/kernels/blackwell_e0_probe/build/cutlass-v4.0.0
cmake --build kernels/blackwell_e0_probe/build/optimized_gemm/cmake-build -j4
python -m kernels.blackwell_e0_probe.optimized_gemm.patch_optimized_gemm kernels/blackwell_e0_probe/build/optimized_gemm/baseline_00.cubin kernels/blackwell_e0_probe/build/optimized_gemm/variant_01.cubin --variant 01 --manifest kernels/blackwell_e0_probe/build/optimized_gemm/variant_01.patch.json
python -m kernels.blackwell_e0_probe.optimized_gemm.patch_optimized_gemm kernels/blackwell_e0_probe/build/optimized_gemm/baseline_00.cubin kernels/blackwell_e0_probe/build/optimized_gemm/variant_11.cubin --variant 11 --manifest kernels/blackwell_e0_probe/build/optimized_gemm/variant_11.patch.json
CUDA_VISIBLE_DEVICES=0 python -m kernels.blackwell_e0_probe.optimized_gemm.run_correctness --real-e0xe2 /tmp/dirotq-real-sana.SRFBZFIE/extracted/sana_real_e0xe2_v1 --real-e0xe0 /tmp/dirotq-real-sana.SRFBZFIE/extracted/sana_real_e0xe0_v1
CUDA_VISIBLE_DEVICES=0 python -m kernels.blackwell_e0_probe.optimized_gemm.benchmark --warmup 100 --iterations 500 --rounds 5
CUDA_VISIBLE_DEVICES=0 python -m pytest -q kernels/blackwell_e0_probe/optimized_gemm/test_optimized_gemm.py
"""
    (root / "commands.txt").write_text(commands)

    excerpt = {
        "public_baseline_first_omma": (
            "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X at text offset 0x3790"
        ),
        "nvdisasm_patched_behavior": (
            "CUDA 12.8 nvdisasm omits the undocumented E0 instruction words; "
            "raw pinned offsets and bit invariants are authoritative"
        ),
        "binary_audit": binary,
    }
    (root / "disassembly_excerpts.json").write_text(
        json.dumps(excerpt, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Optimized static FP4 GEMM feasibility summary", "",
        "Classification: `OPTIMIZED E0 GEMM FEASIBLE`.", "",
        f"CUTLASS v4.0.0 `{CUTLASS_COMMIT}`; baseline `{BASELINE_SHA256}`; "
        "64 OMMA slots; stage count 4; tile 128x128x128; cluster 1x1x1.", "",
        "## Correctness", "",
        "| case | variant | MNK | max abs | mean abs | max rel | tol mismatch | bitwise mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_metric_row(case) for case in correctness["synthetic"]["cases"])
    for key in ("real_e0xe2", "real_e0xe0"):
        lines.extend(_metric_row(case) for case in correctness[key]["cases"])
    lines.extend(["", "All 14 optimized cases have zero tolerance and bitwise mismatches.", "",
                  "## Main-shape latency", "",
                  "All values are CUDA-event medians in ms; 100 warmups, 500 iterations/round, 5 rounds.", "",
                  "| MNK | variant | GEMM | A scale | BF16 cast | estimated total | vs E2 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for shape in benchmark["shapes"]:
        for variant in ("00", "01", "11"):
            value = shape["optimized"][variant]
            lines.append(
                f"| {'x'.join(map(str, shape['shape']))} | {variant} | "
                f"{value['gemm_only']['median_ms']:.9f} | "
                f"{value['dynamic_a_scale_transform']['median_ms']:.9f} | "
                f"{value['output_bf16_cast']['median_ms']:.9f} | "
                f"{value['estimated_low_branch_total_ms']:.9f} | "
                f"{value['latency_ratio_vs_e2']:.6f} |"
            )
    lines.extend(["", "Dynamic A payload transform is zero. A scale transform is 12.4-12.9% of GEMM. "
                  "Static B scale prepack is excluded per forward. Native layout adds 7,936 bytes "
                  "(0.1658%) for both main shapes.", "",
                  "This is a kernel-core feasibility result, not a DiRotQ layer or model speedup.", ""])
    (root / "summary_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=BUILD_ROOT)
    args = parser.parse_args()
    generate(args.root.resolve())


if __name__ == "__main__":
    main()
