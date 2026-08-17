#!/usr/bin/env python3
"""CLI for fit-only high-weight GPTQ and conditional W8A8 Dev32."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dirotq_fused_unrotation_fast import patch_forward_fast, preconvert_rotations_to_device
from utils.e0joint_gptq import sha256_file
from utils.fp8_high_e0_low_experiment import (
    MODEL_REVISION,
    evaluate_teacher_cache,
    load_pipeline,
    setup_wrapped_transformer,
)
from utils.fp8_high_gptq_experiment import (
    build_high_weight_sidecars,
    collect_high_activation_hessians,
    high_hessian_metadata,
    high_target_hashes,
    load_high_hessian_cache,
    materialize_high_sidecar_into_state,
    summarize_dev_weight_gate,
    validate_high_sidecar,
    write_high_hessian_cache,
)


OLD_RUN = ROOT / "models/sana-1.6b/fp8_high3x_e0_low_ada"
DEFAULT_RUN = ROOT / "models/sana-1.6b/fp8_high3x_gptq_e0_low_ada"
BASIS = ROOT / "models/sana-1.6b/basis/U-sana-1.6b.pt"
ROTATION = OLD_RUN / "matched_rotation_rank-3r.pt"
FIT_DIR = OLD_RUN / "teacher_cache/fit"
DEV_DIR = OLD_RUN / "teacher_cache/dev"
OLD_BUILD = OLD_RUN / "caches/rank-3r/build_summary.json"
OLD_BUILD_R = OLD_RUN / "caches/rank-r/build_summary.json"
B1_CACHE = OLD_RUN / "caches/rank-3r/rank-3r_high-bf16_low-e0.pt"
LOW_SIDECAR = OLD_RUN / "caches/rank-3r/low_e0_packing.pt"
LOW_HESSIAN = OLD_RUN / "caches/rank-3r/e0_hessians.pt"
SMOKE_LAYERS = (
    "transformer_blocks.10.attn1.to_q",
    "transformer_blocks.10.attn1.to_out.0",
)
WEIGHT_ARMS = {
    "E4-PC-RTN-W": ("e4-pc-rtn", "bf16"),
    "E4-PC-GPTQ-W": ("e4-pc-gptq", "bf16"),
    "MX-BEST-RTN-W": ("mx-best-rtn", "bf16"),
    "MX-NEIGHBOR-GPTQ-W": ("mx-neighbor-gptq", "bf16"),
}
A8_ARMS = {
    "E4-PTPC-GPTQ-AW": ("e4-pc-gptq", "e4m3-token"),
    "MX-K32-GPTQ-AW": ("mx-neighbor-gptq", "mxfp8-neighbor"),
}
A8_DECOMPOSITION_ARMS = {
    "E4-PT-BF16W-A": (None, "e4m3-token"),
    "MX-K32-BF16W-A": (None, "mxfp8-neighbor"),
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path), "sha256": sha256_file(path), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _fixed_artifacts() -> dict:
    paths = {
        "basis": BASIS, "rotation": ROTATION, "fit_manifest": FIT_DIR / "manifest.json",
        "dev_manifest": DEV_DIR / "manifest.json", "rank3_low_e0_hessian": LOW_HESSIAN,
        "rank3_low_e0_sidecar": LOW_SIDECAR, "rank3_bf16_high_cache": B1_CACHE,
    }
    records = {key: _file_record(path) for key, path in paths.items()}
    old = json.loads(OLD_BUILD.read_text())
    expected = {
        "basis": old["basis_sha256"], "rotation": old["rotation_sha256"],
        "fit_manifest": old["fit_manifest_sha256"],
        "rank3_low_e0_hessian": old["hessian"]["sha256"],
        "rank3_low_e0_sidecar": old["low_sidecar_sha256"],
        "rank3_bf16_high_cache": old["cache_records"]["bf16"]["sha256"],
    }
    for key, digest in expected.items():
        if records[key]["sha256"] != digest:
            raise RuntimeError(f"immutable prerequisite SHA mismatch: {key}")
    fit = json.loads((FIT_DIR / "manifest.json").read_text())
    dev = json.loads((DEV_DIR / "manifest.json").read_text())
    if fit["cache_count"] != 640 or dev["cache_count"] != 1280:
        raise RuntimeError("frozen fit64/dev32 teacher cache count mismatch")
    return records


def _load_basis_rotation():
    return (
        torch.load(BASIS, map_location="cpu", weights_only=False),
        torch.load(ROTATION, map_location="cpu", weights_only=False),
    )


def _setup_pipeline(high_activation: str = "bf16"):
    basis, rotation = _load_basis_rotation()
    pipe = load_pipeline()
    setup = setup_wrapped_transformer(
        pipe.transformer, repo=ROOT, basis=basis, rotation=rotation,
        high_activation_format=high_activation, collect_high_stats=True,
        collect_low_stats=True,
    )
    return pipe, setup


def _common_metadata(transformer, *, damping: float) -> dict:
    fit_manifest = FIT_DIR / "manifest.json"
    targets = high_target_hashes(transformer)
    return high_hessian_metadata(
        source_commit=_source_commit(), fit_manifest=fit_manifest,
        basis_path=BASIS, rotation_path=ROTATION, target_hashes=targets,
        damping=damping,
    )


def command_preflight(args):
    records = _fixed_artifacts()
    report = {
        "source_commit": _source_commit(), "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "model_revision": MODEL_REVISION, "artifacts": records,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "final_or_pilot_read": False,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.run_dir / "preflight.json", report)
    print(json.dumps(report, indent=2))


def command_smoke(args):
    _fixed_artifacts()
    pipe, _ = _setup_pipeline("bf16")
    pipe.transformer.to("cuda").eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    metadata = _common_metadata(pipe.transformer, damping=args.damp_pct)
    hessians, collection = collect_high_activation_hessians(
        pipe.transformer, FIT_DIR, batch_size=args.batch_size,
        layer_names=SMOKE_LAYERS, device="cuda",
    )
    output = args.run_dir / "smoke-two-layer"
    report = build_high_weight_sidecars(
        pipe.transformer, hessians, output, common_metadata=metadata,
        layer_names=SMOKE_LAYERS, damp_pct=args.damp_pct, require_cuda=True,
    )
    report["hessian_collection"] = collection
    report["layers"] = list(SMOKE_LAYERS)
    _write_json(output / "smoke_report.json", report)
    print(json.dumps(report, indent=2))


def command_hessian(args):
    _fixed_artifacts()
    path = args.run_dir / "high_hessians.pt"
    pipe, _ = _setup_pipeline("bf16")
    pipe.transformer.to("cuda").eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    metadata = _common_metadata(pipe.transformer, damping=args.damp_pct)
    hessians, collection = collect_high_activation_hessians(
        pipe.transformer, FIT_DIR, batch_size=args.batch_size, device="cuda"
    )
    record = write_high_hessian_cache(path, hessians, metadata)
    report = {"cache": record, "metadata": metadata, "collection": collection}
    _write_json(args.run_dir / "high_hessian_report.json", report)
    print(json.dumps(report, indent=2))


def command_build(args):
    _fixed_artifacts()
    pipe, _ = _setup_pipeline("bf16")
    pipe.transformer.to("cuda").eval().requires_grad_(False)
    metadata = _common_metadata(pipe.transformer, damping=args.damp_pct)
    hessians = load_high_hessian_cache(
        args.run_dir / "high_hessians.pt", metadata
    )
    torch.cuda.reset_peak_memory_stats()
    report = build_high_weight_sidecars(
        pipe.transformer, hessians, args.run_dir / "high_weights",
        common_metadata=metadata, damp_pct=args.damp_pct, require_cuda=True,
    )
    print(json.dumps(report, indent=2))


def _persistent_bytes(build: dict) -> tuple[int, dict[str, int], dict]:
    rank3 = json.loads(OLD_BUILD.read_text())
    rank_r = json.loads(OLD_BUILD_R.read_text())
    b0 = int(rank_r["serialized_active_weight_bytes"]["bf16"]["total"])
    e4_low = rank3["serialized_active_weight_bytes"]["e4m3"]
    mx_low = rank3["serialized_active_weight_bytes"]["mxfp8"]
    low_base_e4 = int(e4_low["low_payload"] + e4_low["low_scales"] + e4_low["low_global"])
    low_base_mx = int(mx_low["low_payload"] + mx_low["low_scales"] + mx_low["low_global"])
    values = {}
    for arm, (recipe, _) in {**WEIGHT_ARMS, **A8_ARMS}.items():
        high = int(build["sidecars"][recipe]["serialized_tensor_bytes"]["total"])
        values[arm] = (low_base_e4 if recipe.startswith("e4") else low_base_mx) + high
    detail = {"B0": b0, "rank3_low_base_e4": low_base_e4,
              "rank3_low_base_mx": low_base_mx}
    return b0, values, detail


def _activation_bytes_per_row() -> dict[str, int]:
    # K16 low payload + one UE4M3 scale per block + high payload/scales.
    b0 = 1952 // 2 + 1952 // 16 + 288 * 2
    e4 = 1376 // 2 + 1376 // 16 + 864 + 4
    mx = 1376 // 2 + 1376 // 16 + 864 + 864 // 32
    return {"B0": b0, "E4-PTPC-GPTQ-AW": e4, "MX-K32-GPTQ-AW": mx}


def command_freeze(args):
    artifacts = _fixed_artifacts()
    build_path = args.run_dir / "high_weights/build_summary.json"
    build = json.loads(build_path.read_text())
    if build["layer_count"] != 120 or build["gptq_coverage"] != 120:
        raise RuntimeError("high-weight build lacks 120/120 coverage")
    if build["rtn_fallbacks"] or build["cpu_fallbacks"]:
        raise RuntimeError("high-weight build contains a forbidden fallback")
    for recipe, record in build["sidecars"].items():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"sidecar hash mismatch: {recipe}")
    b0_bytes, arm_bytes, byte_detail = _persistent_bytes(build)
    freeze = {
        "schema": "dirotq.fp8_high_gptq_method_freeze.v1",
        "source_commit": _source_commit(), "model_revision": MODEL_REVISION,
        "immutable_artifacts": artifacts,
        "high_hessian": _file_record(args.run_dir / "high_hessians.pt"),
        "high_weight_build": _file_record(build_path),
        "sidecars": build["sidecars"],
        "recipes": {
            "e4_scale_candidates": [0.875, 1.0, 1.125],
            "e4_rtn": "fit-H-rowwise-neighbor",
            "e4_gptq": "same-frozen-per-channel-scales-sequential-GPTQ",
            "mx_rtn_candidates": ["current", "nosat", "neighbor"],
            "mx_gptq": "neighbor-raw-SSE-frozen-UE8M0-then-sequential-GPTQ",
            "dev_used_for_recipe": False,
        },
        "b0_persistent_weight_bytes": b0_bytes,
        "arm_persistent_weight_bytes": arm_bytes,
        "byte_detail": byte_detail,
        "activation_serialized_bytes_per_row": _activation_bytes_per_row(),
        "final_or_pilot_read": False,
    }
    _write_json(args.run_dir / "method_freeze.json", freeze)
    print(json.dumps(freeze, indent=2))


def _load_freeze(run_dir: Path) -> dict:
    freeze = json.loads((run_dir / "method_freeze.json").read_text())
    if freeze["source_commit"] != _source_commit():
        raise RuntimeError("formal evaluation must run at the frozen source commit")
    _fixed_artifacts()
    for recipe, record in freeze["sidecars"].items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"frozen sidecar changed: {recipe}")
    return freeze


def command_evaluate(args):
    freeze = _load_freeze(args.run_dir)
    all_arms = {**WEIGHT_ARMS, **A8_ARMS, **A8_DECOMPOSITION_ARMS}
    recipe, activation = all_arms[args.arm]
    if args.arm in {**A8_ARMS, **A8_DECOMPOSITION_ARMS}:
        gate_path = args.run_dir / "evaluation/dev/weight_gate.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text())["continue_to_A8"]:
            raise RuntimeError("W8A8 evaluation is forbidden until a weight-only arm passes")
        family = "E4" if args.arm.startswith("E4") else "MX"
        required = f"{family}-PC-GPTQ-W" if family == "E4" else "MX-NEIGHBOR-GPTQ-W"
        if not json.loads(gate_path.read_text())["arms"][required]["passed"]:
            raise RuntimeError(f"{args.arm} weight recipe did not pass the frozen gate")
    pipe, setup = _setup_pipeline(activation)
    base = torch.load(B1_CACHE, map_location="cpu", weights_only=False)
    if recipe is None:
        sidecar_path = None
        state = base
    else:
        sidecar_path = Path(freeze["sidecars"][recipe]["path"])
        sidecar = validate_high_sidecar(sidecar_path, {}, layer_count=120)
        state = materialize_high_sidecar_into_state(pipe.transformer, base, sidecar)
        del base
    pipe.transformer.load_state_dict(state, strict=False)
    del state; gc.collect()
    for layer in setup["active_layers"].values():
        layer._unrot_fused = True
    patch_forward_fast()
    pipe.transformer.to("cuda")
    preconvert_rotations_to_device(pipe.transformer, device="cuda", dtype=torch.bfloat16)
    output = args.run_dir / "evaluation/dev" / f"{args.arm}.csv"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Dev result: {output}")
    report = evaluate_teacher_cache(
        pipe.transformer, DEV_DIR, output, arm=args.arm, batch_size=args.batch_size
    )
    report.update({
        "source_commit": freeze["source_commit"],
        "sidecar": _file_record(sidecar_path) if sidecar_path is not None else None,
        "base_low_cache_sha256": freeze["immutable_artifacts"]["rank3_bf16_high_cache"]["sha256"],
        "low_sidecar_sha256": freeze["immutable_artifacts"]["rank3_low_e0_sidecar"]["sha256"],
        "high_activation_format": activation,
        "high_activation_stats": {
            name: stats.snapshot() for name, stats in setup["high_stats"].items()
        },
        "low_activation_stats": {
            name: stats.snapshot() for name, stats in setup["low_stats"].items()
        },
        "final_or_pilot_read": False,
    })
    _write_json(output.with_suffix(".provenance.json"), report)
    print(json.dumps(report, indent=2))


def _old_summary(arm: str) -> dict:
    return json.loads((OLD_RUN / "evaluation/dev" / f"{arm}.summary.json").read_text())


def _comparison_vs_b0(b0: dict, candidate: dict) -> dict:
    keys = sorted(b0["per_prompt"])
    fractions = [
        (b0["per_prompt"][key] - candidate["per_prompt"][key])
        / b0["per_prompt"][key]
        for key in keys
    ]
    groups = {"early": range(0, 7), "mid": range(7, 14), "late": range(14, 20)}
    group_changes = {}
    for group, steps in groups.items():
        base = sum(b0["per_timestep"].get(str(step), b0["per_timestep"].get(step, 0.0))
                   for step in steps)
        value = sum(candidate["per_timestep"].get(
            str(step), candidate["per_timestep"].get(step, 0.0)
        ) for step in steps)
        group_changes[group] = (value - base) / base
    return {
        "raw_aggregate_gain_fraction": (
            b0["raw_sse"] - candidate["raw_sse"]
        ) / b0["raw_sse"],
        "equal_prompt_mean_gain_fraction": statistics.fmean(fractions),
        "equal_prompt_median_gain_fraction": statistics.median(fractions),
        "prompt_wins": sum(value > 0 for value in fractions),
        "timestep_group_relative_changes": group_changes,
    }


def command_weight_gate(args):
    freeze = _load_freeze(args.run_dir)
    summaries = {
        "B0": _old_summary("B0"), "B1": _old_summary("B1"),
        "old-E4-W": _old_summary("E4-W"), "old-MX-W": _old_summary("MX-W"),
    }
    for arm in WEIGHT_ARMS:
        summaries[arm] = json.loads(
            (args.run_dir / "evaluation/dev" / f"{arm}.summary.json").read_text()
        )
    report = summarize_dev_weight_gate(
        summaries, b0_persistent_bytes=freeze["b0_persistent_weight_bytes"],
        arm_persistent_bytes=freeze["arm_persistent_weight_bytes"],
    )
    report["historical_controls"] = {
        key: {"raw_sse": value["raw_sse"], "relative_mse": value["relative_mse"]}
        for key, value in summaries.items() if key in {"B0", "B1", "old-E4-W", "old-MX-W"}
    }
    report["all_comparisons_vs_B0"] = {
        arm: _comparison_vs_b0(summaries["B0"], summary)
        for arm, summary in summaries.items() if arm != "B0"
    }
    report["classification_if_stop"] = (
        None if report["continue_to_A8"]
        else "HIGH WEIGHT GPTQ DOES NOT RECOVER RANK HEADROOM"
    )
    _write_json(args.run_dir / "evaluation/dev/weight_gate.json", report)
    print(json.dumps(report, indent=2))


def command_a8_gate(args):
    freeze = _load_freeze(args.run_dir)
    weight_gate = json.loads((args.run_dir / "evaluation/dev/weight_gate.json").read_text())
    if not weight_gate["continue_to_A8"]:
        raise RuntimeError("A8 gate cannot run because weight-only gate stopped the experiment")
    b0 = _old_summary("B0")
    results = {}
    groups = {"early": range(0, 7), "mid": range(7, 14), "late": range(14, 20)}
    for arm in A8_ARMS:
        family_weight = "E4-PC-GPTQ-W" if arm.startswith("E4") else "MX-NEIGHBOR-GPTQ-W"
        if not weight_gate["arms"][family_weight]["passed"]:
            continue
        candidate = json.loads(
            (args.run_dir / "evaluation/dev" / f"{arm}.summary.json").read_text()
        )
        keys = sorted(b0["per_prompt"])
        wins = sum(candidate["per_prompt"][key] < b0["per_prompt"][key] for key in keys)
        group_changes = {}
        for group, steps in groups.items():
            base = sum(b0["per_timestep"].get(str(step), 0.0) for step in steps)
            value = sum(candidate["per_timestep"].get(str(step), 0.0) for step in steps)
            group_changes[group] = (value - base) / base
        weight_ratio = freeze["arm_persistent_weight_bytes"][arm] / freeze[
            "b0_persistent_weight_bytes"
        ]
        activation_ratio = freeze["activation_serialized_bytes_per_row"][arm] / freeze[
            "activation_serialized_bytes_per_row"
        ]["B0"]
        raw_gain = (b0["raw_sse"] - candidate["raw_sse"]) / b0["raw_sse"]
        passed = (raw_gain >= .03 and wins >= 24 and max(group_changes.values()) <= .02
                  and weight_ratio <= 1.01 and activation_ratio <= 1.01)
        results[arm] = {
            "raw_gain_fraction": raw_gain, "prompt_wins": wins,
            "timestep_group_relative_changes": group_changes,
            "persistent_weight_ratio_vs_B0": weight_ratio,
            "batch4_activation_ratio_vs_B0": activation_ratio,
            "passed": passed,
            "decomposition": {
                "activation_only_arm": (
                    "E4-PT-BF16W-A" if arm.startswith("E4") else "MX-K32-BF16W-A"
                ),
                "activation_only_raw_sse": json.loads((
                    args.run_dir / "evaluation/dev" /
                    ("E4-PT-BF16W-A.summary.json" if arm.startswith("E4")
                     else "MX-K32-BF16W-A.summary.json")
                ).read_text())["raw_sse"],
                "weight_only_arm": family_weight,
                "weight_only_raw_sse": json.loads((
                    args.run_dir / "evaluation/dev" / f"{family_weight}.summary.json"
                ).read_text())["raw_sse"],
                "joint_raw_sse": candidate["raw_sse"],
            },
        }
    passing = [arm for arm, value in results.items() if value["passed"]]
    if len(passing) == 2:
        classification = "FP8 HIGH-3R E0-LOW FEASIBLE"
    elif passing == ["E4-PTPC-GPTQ-AW"]:
        classification = "E4 W8A8 HIGH-3R PASSES"
    elif passing == ["MX-K32-GPTQ-AW"]:
        classification = "MXFP8 W8A8 HIGH-3R PASSES"
    else:
        e4_weight = weight_gate["arms"]["E4-PC-GPTQ-W"]["passed"]
        mx_weight = weight_gate["arms"]["MX-NEIGHBOR-GPTQ-W"]["passed"]
        classification = (
            "E4 HIGH WEIGHT GPTQ PASSES, ACTIVATION FP8 BLOCKED" if e4_weight
            else "MX HIGH WEIGHT GPTQ PASSES, ACTIVATION FP8 BLOCKED" if mx_weight
            else "HIGH WEIGHT GPTQ DOES NOT RECOVER RANK HEADROOM"
        )
    report = {"arms": results, "classification": classification,
              "final_or_pilot_read": False}
    _write_json(args.run_dir / "evaluation/dev/a8_gate.json", report)
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--damp-pct", type=float, default=.01)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("smoke")
    sub.add_parser("hessian")
    sub.add_parser("build")
    sub.add_parser("freeze")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--arm", choices=tuple({**WEIGHT_ARMS, **A8_ARMS, **A8_DECOMPOSITION_ARMS}),
        required=True,
    )
    sub.add_parser("weight-gate")
    sub.add_parser("a8-gate")
    return parser


def main():
    args = build_parser().parse_args()
    commands = {
        "preflight": command_preflight, "smoke": command_smoke,
        "hessian": command_hessian, "build": command_build,
        "freeze": command_freeze, "evaluate": command_evaluate,
        "weight-gate": command_weight_gate, "a8-gate": command_a8_gate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
