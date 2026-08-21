#!/usr/bin/env python3
"""Generate matched rank-64 FLUX NVFP4 W4A16 adaptive-norm Pilot32."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from evaluate_shared_pca_audit import load_samples, validate_images
from run_flux_rank64_nvfp4_pilot32 import (
    MODEL_REVISION, ROTATION_REL, SCHEMES, run_logged, sha256_file, stat_record,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--main-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--allow-foreign-processes", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    formal = args.formal_root.resolve()
    snapshot = args.model_snapshot.resolve()
    output = args.output_root.resolve()
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError(f"expected exact FLUX revision {MODEL_REVISION}")
    dataset = ROOT / "datasets/mjhq_5000_samples.json"
    calibration = formal / "calibration_dataset/caches"
    rotation = formal / ROTATION_REL
    sidecar = output / "packed_cache/flux-modulators-nvfp4-e2-g16.pt"
    output.mkdir(parents=True, exist_ok=True)

    if not sidecar.exists():
        from utils.flux_nvfp4_w4a16 import (
            build_cache_from_safetensors, provenance, save_cache,
        )
        cache, report = build_cache_from_safetensors(snapshot / "transformer")
        info = save_cache(
            cache, report, sidecar, cache_provenance=provenance(str(snapshot))
        )
        print("Built NVFP4 W4A16 sidecar:", json.dumps(info, sort_keys=True))
    sidecar_record = stat_record(sidecar)
    if args.build_only:
        return

    env = os.environ.copy()
    env.update({
        "NVIDIA_TF32_OVERRIDE": "0",
        "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    records = []
    immutable = {str(sidecar): sidecar_record}
    for scheme, basis_rel in SCHEMES.items():
        basis = formal / basis_rel
        main_cache = args.main_cache_root / f"{scheme}-r64-nvfp4-g16-gptq.pt"
        for path in (dataset, rotation, basis, main_cache, sidecar):
            if not path.exists():
                raise FileNotFoundError(path)
            immutable.setdefault(str(path), stat_record(path))
        image_dir = output / "pilot32" / scheme
        existing = list(image_dir.rglob("*.png")) if image_dir.exists() else []
        if len(existing) not in (0, 32):
            raise RuntimeError(f"refusing partial output for {scheme}: {len(existing)}")
        if len(existing) == 0:
            command = [
                sys.executable, str(ROOT / "apply_dirotq.py"),
                "--model", "flux-dev", "--model-id", str(snapshot),
                "--dataset", str(dataset), "--rotation-path", str(rotation),
                "--basis-path", str(basis), "--calib-dir", str(calibration),
                "--gptq", "--nvfp4", "--activation-format", "nvfp4",
                "--gptq-calib-files", "3200", "--gptq-batch-size", "8",
                "--gptq-rtn-layers", ".net.2", "proj_out.linears.1",
                "--quantized-cache", str(main_cache),
                "--nvfp4-w4a16-modulators", "--nvfp4-w4a16-cache", str(sidecar),
                "--batch-size", "1", "--max-images", "32",
                "--output-dir", str(image_dir),
            ]
            record = run_logged(
                command, output / f"logs/{scheme}-generate.log", env,
                args.physical_gpu, allow_foreign=args.allow_foreign_processes,
            )
            record.update({"scheme": scheme, "stage": "generate"})
            records.append(record)

    samples = load_samples(dataset, 32)
    integrity = validate_images(
        [(scheme, output / "pilot32" / scheme) for scheme in SCHEMES], samples
    )
    after = {path: stat_record(Path(path)) for path in immutable}
    if immutable != after:
        raise RuntimeError("immutable input/cache changed during W4A16 generation")
    manifest = {
        "schema": "dirotq.flux_rank64_nvfp4_w4a16_pilot32",
        "model_revision": MODEL_REVISION,
        "contract": {
            "adaptive_norm": "BF16 activation x E2M1 NVFP4 RTN weight",
            "adaptive_norm_weight_scale": "FP32 global + E4M3 K16",
            "ffn_down": "legacy NVFP4 W4A4 configured RTN, no PCA high branch",
            "other_wrapped_linears": "legacy NVFP4 W4A4 GPTQ",
            "steps": 25, "guidance_scale": 3.5, "batch_size": 1,
        },
        "sidecar": sidecar_record,
        "immutable_before": immutable,
        "immutable_after": after,
        "integrity": integrity,
        "records": records,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"integrity": integrity, "records": records}, indent=2))


if __name__ == "__main__":
    main()
