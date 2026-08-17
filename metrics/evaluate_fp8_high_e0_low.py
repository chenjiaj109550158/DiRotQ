#!/usr/bin/env python3
"""CLI for the frozen FP8-high / E0-low SANA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch
import yaml

from apply_dirotq import generate_images
from dirotq_fused_unrotation_fast import patch_forward_fast, preconvert_rotations_to_device
from utils.e0joint_gptq import sha256_file
from utils.fp8_high_e0_low import (
    derive_rank_contract,
    generate_matched_residual_rotations,
    serialized_weight_bytes,
    tensor_sha256,
    validate_residual_rotation,
)
from utils.fp8_high_e0_low_experiment import (
    ARMS,
    DATASET_SHA256,
    FIT_STEP_INDICES,
    MODEL_REVISION,
    build_rank_caches,
    collect_teacher_cache,
    dev_gate,
    evaluate_teacher_cache,
    freeze_splits,
    load_pipeline,
    setup_wrapped_transformer,
)
from utils.quant_utils import ActQuantWrapper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "models/sana-1.6b/fp8_high3x_e0_low_ada"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _contracts():
    cfg = yaml.safe_load((ROOT / "models/sana-1.6b/config.yaml").read_text())
    dims = cfg["dims"]
    kwargs = dict(
        hidden_dim=dims["hidden"], head_dim=dims["head"],
        num_heads=dims["num_heads"], high_fraction=cfg["rotation"]["high_fraction"],
    )
    return (
        derive_rank_contract(**kwargs, multiplier=1),
        derive_rank_contract(**kwargs, multiplier=3),
    )


def command_prepare(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    split = freeze_splits(ROOT, args.run_dir / "split_manifest.json")
    baseline, triple = _contracts()
    rank_manifest = {
        "baseline": baseline.__dict__, "triple": triple.__dict__,
        "source_commit": _source_commit(), "model_revision": MODEL_REVISION,
        "dataset_sha256": DATASET_SHA256,
    }
    _write_json(args.run_dir / "rank_manifest.json", rank_manifest)
    print(json.dumps({"split_sha256": split["content_sha256"], **rank_manifest}, indent=2))


def command_rotations(args):
    baseline, triple = _contracts()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for label, contract in (("r", baseline), ("3r", triple)):
        rotation = generate_matched_residual_rotations(
            contract, seed=42, device=args.device
        )
        report = validate_residual_rotation(rotation, contract, atol=2e-8)
        rotation.update({
            "high_len_hidden": contract.high_hidden,
            "high_len_head": contract.high_per_head,
            "high_len_down": 0,
            "high_fraction": contract.high_hidden / contract.hidden_dim,
        })
        path = args.run_dir / f"matched_rotation_rank-{label}.pt"
        torch.save(rotation, path)
        output[label] = {
            "path": str(path), "sha256": sha256_file(path),
            "size": path.stat().st_size, "validation": report,
        }
    production = ROOT / "models/sana-1.6b/basis/R-sana-1.6b.pt"
    output["production_secondary"] = {
        "path": str(production), "sha256": sha256_file(production),
        "note": "secondary only: production metadata says 280/1960, active routing is 288/1952",
    }
    _write_json(args.run_dir / "rotation_provenance.json", output)
    print(json.dumps(output, indent=2))


def command_collect(args):
    manifest = json.loads((args.run_dir / "split_manifest.json").read_text())
    rows = manifest["splits"][args.split]
    steps = FIT_STEP_INDICES if args.split == "fit" else tuple(range(20))
    output = args.run_dir / "teacher_cache" / args.split
    pipe = load_pipeline()
    pipe = pipe.to("cuda")
    pipe.transformer.eval().requires_grad_(False)
    report = collect_teacher_cache(pipe, rows, output, selected_steps=steps)
    print(json.dumps(report, indent=2))


def _load_basis_rotation(run_dir: Path, rank: str):
    basis_path = ROOT / "models/sana-1.6b/basis/U-sana-1.6b.pt"
    rotation_path = run_dir / f"matched_rotation_rank-{rank}.pt"
    basis = torch.load(basis_path, map_location="cpu", weights_only=False)
    rotation = torch.load(rotation_path, map_location="cpu", weights_only=False)
    return basis_path, rotation_path, basis, rotation


def command_build(args):
    basis_path, rotation_path, basis, rotation = _load_basis_rotation(args.run_dir, args.rank)
    pipe = load_pipeline()
    setup_wrapped_transformer(
        pipe.transformer, repo=ROOT, basis=basis, rotation=rotation,
        high_activation_format="bf16",
    )
    fit_manifest = args.run_dir / "teacher_cache/fit/manifest.json"
    split_manifest = args.run_dir / "split_manifest.json"
    formats = ("bf16",) if args.rank == "r" else ("bf16", "e4m3", "mxfp8")
    provenance = {
        "source_commit": _source_commit(), "model_revision": MODEL_REVISION,
        "basis_path": str(basis_path), "basis_sha256": sha256_file(basis_path),
        "rotation_path": str(rotation_path), "rotation_sha256": sha256_file(rotation_path),
        "fit_manifest_sha256": sha256_file(fit_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "activation_format": "hardware-e0m3-gscale2688-k16-ue4m3",
        "weight_format": "hardware-e0m3-gscale2688-k16-ue4m3-gptq",
    }
    report = build_rank_caches(
        pipe.transformer, fit_dir=args.run_dir / "teacher_cache/fit",
        output_dir=args.run_dir / "caches" / f"rank-{args.rank}",
        rank_label=args.rank, formats=formats, provenance=provenance,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


def _cache_for_arm(run_dir: Path, arm: str) -> tuple[str, str, Path]:
    contract = ARMS[arm]
    rank = contract["rank"]
    weight = contract["weight"]
    path = run_dir / "caches" / f"rank-{rank}" / f"rank-{rank}_high-{weight}_low-e0.pt"
    return rank, contract["activation"], path


def command_evaluate(args):
    rank, high_activation, cache_path = _cache_for_arm(args.run_dir, args.arm)
    basis_path, rotation_path, basis, rotation = _load_basis_rotation(args.run_dir, rank)
    build_summary = json.loads(
        (args.run_dir / "caches" / f"rank-{rank}" / "build_summary.json").read_text()
    )
    expected = build_summary["cache_records"][ARMS[args.arm]["weight"]]["sha256"]
    if sha256_file(cache_path) != expected:
        raise RuntimeError(f"{args.arm}: reconstructed cache SHA-256 mismatch")
    pipe = load_pipeline()
    setup = setup_wrapped_transformer(
        pipe.transformer, repo=ROOT, basis=basis, rotation=rotation,
        high_activation_format=high_activation, collect_high_stats=True,
        collect_low_stats=True,
    )
    state = torch.load(cache_path, map_location="cpu", weights_only=False)
    pipe.transformer.load_state_dict(state, strict=False)
    for layer in setup["active_layers"].values():
        layer._unrot_fused = True
    patch_forward_fast()
    pipe.transformer.to("cuda")
    preconvert_rotations_to_device(pipe.transformer, device="cuda", dtype=torch.bfloat16)
    report = evaluate_teacher_cache(
        pipe.transformer, args.run_dir / "teacher_cache" / args.split,
        args.run_dir / "evaluation" / args.split / f"{args.arm}.csv",
        arm=args.arm, batch_size=args.batch_size,
    )
    report.update({
        "cache_path": str(cache_path), "cache_sha256": expected,
        "basis_sha256": sha256_file(basis_path),
        "rotation_sha256": sha256_file(rotation_path),
        "high_activation_stats": {
            name: stats.snapshot() for name, stats in setup["high_stats"].items()
        },
        "low_activation_stats": {
            name: stats.snapshot() for name, stats in setup["low_stats"].items()
        },
    })
    _write_json(
        args.run_dir / "evaluation" / args.split / f"{args.arm}.provenance.json",
        report,
    )
    print(json.dumps(report, indent=2))


def command_dev_gate(args):
    summaries = {}
    for arm in ARMS:
        path = args.run_dir / "evaluation/dev" / f"{arm}.summary.json"
        summaries[arm] = json.loads(path.read_text())
    gate = dev_gate(summaries)
    _write_json(args.run_dir / "evaluation/dev/gate.json", gate)
    print(json.dumps(gate, indent=2))


def command_final_gate(args):
    dev = json.loads((args.run_dir / "evaluation/dev/gate.json").read_text())
    passing = [arm for arm, result in dev["arms"].items() if result["passed"]]
    summaries = {
        "B0": json.loads((args.run_dir / "evaluation/final/B0.summary.json").read_text())
    }
    for arm in passing:
        summaries[arm] = json.loads(
            (args.run_dir / "evaluation/final" / f"{arm}.summary.json").read_text()
        )
    results = {}
    baseline = summaries["B0"]
    groups = {"early": range(0, 7), "mid": range(7, 14), "late": range(14, 20)}
    for arm in passing:
        candidate = summaries[arm]
        gain = (baseline["raw_sse"] - candidate["raw_sse"]) / baseline["raw_sse"]
        wins = sum(candidate["per_prompt"][key] < value
                   for key, value in baseline["per_prompt"].items())
        group_changes = {}
        for name, steps in groups.items():
            base = sum(baseline["per_timestep"].get(str(step), 0) for step in steps)
            cand = sum(candidate["per_timestep"].get(str(step), 0) for step in steps)
            group_changes[name] = (cand - base) / base
        results[arm] = {
            "aggregate_gain": gain, "prompt_wins": wins,
            "timestep_group_relative_changes": group_changes,
            "teacher_gate_passed": gain >= .01 and wins >= 40 and max(group_changes.values()) <= .02,
        }
    report = {"dev_passing_arms": passing, "arms": results,
              "pilot_allowed": any(row["teacher_gate_passed"] for row in results.values())}
    _write_json(args.run_dir / "evaluation/final/gate.json", report)
    print(json.dumps(report, indent=2))


def _pilot_dataset(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "split_manifest.json").read_text())
    data = {
        row["image_id"]: {"prompt": row["prompt"], "category": row["category"]}
        for row in manifest["splits"]["pilot"]
    }
    path = run_dir / "pilot_dataset.json"
    _write_json(path, data)
    return path


def command_generate(args):
    dataset = _pilot_dataset(args.run_dir)
    output = args.run_dir / "pilot64" / args.arm
    if args.arm == "S":
        pipe = load_pipeline()
    else:
        rank, high_activation, cache_path = _cache_for_arm(args.run_dir, args.arm)
        _, _, basis, rotation = _load_basis_rotation(args.run_dir, rank)
        pipe = load_pipeline()
        setup = setup_wrapped_transformer(
            pipe.transformer,
            repo=ROOT, basis=basis, rotation=rotation,
            high_activation_format=high_activation,
        )
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        pipe.transformer.load_state_dict(state, strict=False)
        for layer in setup["active_layers"].values():
            layer._unrot_fused = True
        patch_forward_fast()
    pipe.enable_model_cpu_offload()
    if args.arm != "S":
        preconvert_rotations_to_device(pipe.transformer, device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    generate_images(
        pipe, output, dataset,
        {"num_inference_steps": 20, "guidance_scale": 4.5,
         "height": 1024, "width": 1024},
        max_images=64, batch_size=4,
    )
    report = {
        "arm": args.arm, "output": str(output), "dataset_sha256": sha256_file(dataset),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }
    _write_json(output / "generation_provenance.json", report)
    print(json.dumps(report, indent=2))


def command_image_metrics(args):
    import csv
    import numpy as np
    from torchmetrics.image import (
        LearnedPerceptualImagePatchSimilarity,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.multimodal import CLIPScore
    from evaluate_pilot128 import _evaluate_config, _load_samples, _validate_configs, _write_csv

    dataset = _pilot_dataset(args.run_dir)
    samples = _load_samples(dataset, 64)
    final_gate = json.loads((args.run_dir / "evaluation/final/gate.json").read_text())
    passing = [arm for arm, result in final_gate["arms"].items()
               if result["teacher_gate_passed"]]
    arms = ["S", "B0", "B1", "E4-W", "MX-W", *passing]
    arms = list(dict.fromkeys(arms))
    configs = [(arm, args.run_dir / "pilot64" / arm) for arm in arms]
    _validate_configs(configs, samples)
    device = torch.device("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction="none").to(device)
    clip = CLIPScore("openai/clip-vit-large-patch14").to(device)
    clip.model.eval()
    rows = []
    reference = args.run_dir / "pilot64/S"
    for arm, path in configs:
        rows.extend(_evaluate_config(arm, path, reference, samples, 4, device, lpips, ssim, clip))
    output = args.run_dir / "pilot64/metrics"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_prompt.csv", rows)
    by_arm = {arm: [row for row in rows if row["config"] == arm] for arm in arms}
    metrics = ("psnr", "lpips", "ssim", "clip_score")
    summary = []
    rng = np.random.default_rng(20260817)
    for arm in arms:
        for metric in metrics:
            values = np.array([float(row[metric]) for row in by_arm[arm]])
            summary.append({"row_type": "arm", "arm": arm, "reference": "",
                            "metric": metric, "mean": values.mean(),
                            "median": np.median(values), "delta": "", "win_rate": "",
                            "ci95_low": "", "ci95_high": ""})
    for arm in passing:
        for metric in metrics:
            candidate = np.array([float(row[metric]) for row in by_arm[arm]])
            baseline = np.array([float(row[metric]) for row in by_arm["B0"]])
            delta = candidate - baseline
            indices = rng.integers(0, 64, size=(5000, 64))
            low, high = np.quantile(delta[indices].mean(1), (.025, .975))
            wins = candidate < baseline if metric == "lpips" else candidate > baseline
            summary.append({"row_type": "paired", "arm": arm, "reference": "B0",
                            "metric": metric, "mean": "", "median": "",
                            "delta": delta.mean(), "win_rate": wins.mean(),
                            "ci95_low": low, "ci95_high": high})
    _write_csv(output / "summary.csv", summary)
    print(json.dumps({"arms": arms, "rows": len(rows), "output": str(output)}, indent=2))


def command_memory(args):
    baseline, triple = _contracts()
    # Representative square hidden projection.  Full-model totals are derived
    # from actual layer rows in build reports below when available.
    table = {}
    specs = {
        "B0": (baseline, "bf16"), "B1": (triple, "bf16"),
        "E4-W": (triple, "e4m3"), "E4-AW": (triple, "e4m3"),
        "MX-W": (triple, "mxfp8"), "MX-AW": (triple, "mxfp8"),
    }
    for arm, (contract, fmt) in specs.items():
        table[arm] = serialized_weight_bytes(
            out_features=2240, low_rank=contract.low_hidden,
            high_rank=contract.high_hidden, high_format=fmt,
        )
    _write_json(args.run_dir / "serialized_memory_representative.json", table)
    print(json.dumps(table, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    rotations = sub.add_parser("rotations"); rotations.add_argument("--device", default="cuda")
    collect = sub.add_parser("collect"); collect.add_argument("--split", choices=("fit", "dev", "final"), required=True)
    build = sub.add_parser("build"); build.add_argument("--rank", choices=("r", "3r"), required=True); build.add_argument("--batch-size", type=int, default=4)
    evaluate = sub.add_parser("evaluate"); evaluate.add_argument("--arm", choices=tuple(ARMS), required=True); evaluate.add_argument("--split", choices=("dev", "final"), required=True); evaluate.add_argument("--batch-size", type=int, default=4)
    sub.add_parser("dev-gate")
    sub.add_parser("final-gate")
    generate = sub.add_parser("generate"); generate.add_argument("--arm", choices=("S", *tuple(ARMS)), required=True)
    sub.add_parser("image-metrics")
    sub.add_parser("memory")
    return parser


def main():
    args = build_parser().parse_args()
    {
        "prepare": command_prepare,
        "rotations": command_rotations,
        "collect": command_collect,
        "build": command_build,
        "evaluate": command_evaluate,
        "dev-gate": command_dev_gate,
        "final-gate": command_final_gate,
        "generate": command_generate,
        "image-metrics": command_image_metrics,
        "memory": command_memory,
    }[args.command](args)


if __name__ == "__main__":
    main()
