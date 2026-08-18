"""
apply_dirotq.py

DiRotQ: PCA-based rotation + GPTQ/RTN mixed-precision W4A4 quantization.

Model-specific logic (rotation assignment, quantizer config, generation params)
is loaded from models/<model>/model_utils.py.

Supports:
  --model NAME: model name (subdirectory under models/)
  --gptq: GPTQ weight quantization (default: RTN)
  --max-images N: generate only N images (for testing)
"""

import os
import sys
import json
import subprocess
import yaml
import torch
import torch.nn as nn
import diffusers.training_utils
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from utils.quant_utils import (
    ActQuantWrapper, add_actquant, find_qlayers,
    rtn_quantize_weights, nvfp4_rtn_quantize_weights,
    capture_transformed_w16_weights, install_transformed_w16_weights,
)
from utils.gptq_utils import collect_hessians, gptq_quantize_weights
from utils.tilemixfp4_utils import FormatSelectionStats
from utils.fouroversix_utils import FOUR_OVER_SIX_FORMATS, FourOverSixStats
from utils.output_tilemixfp4_utils import OutputOracleFormatStats
from utils.e0joint_gptq import (
    OBJECTIVE_VERSION as E0JOINT_OBJECTIVE_VERSION,
    build_e0joint_gptq,
    sha256_file,
    validate_e0joint_cache_path,
    validate_e0joint_metadata,
    write_e0joint_metadata,
)
from utils.weight_mixfp4 import (
    PRODUCTION_MODES as WEIGHT_MIX_MODES,
    build_weight_mix_caches,
    expected_metadata as expected_weight_mix_metadata,
    validate_cache_metadata as validate_weight_mix_metadata,
)
from utils.hardware_weight_fp4 import (
    FORMATS as HARDWARE_WEIGHT_FORMATS,
    build_hardware_fixed_caches,
    expected_metadata as expected_hardware_weight_metadata,
    validate_metadata as validate_hardware_weight_metadata,
    validate_runtime_state as validate_hardware_weight_runtime,
)

_ROOT = Path(__file__).parent


def residual_rotation_cache_tag(mode: str) -> str:
    """Cache/output tag; random keeps the historical filename regression."""
    if mode == "random":
        return ""
    if mode == "identity":
        return "_rr-identity"
    raise ValueError(f"unsupported residual rotation: {mode}")


def gptq_hessian_cache_name(num_files: int, num_layers: int, mode: str) -> str:
    return f"hessians_n{num_files}_l{num_layers}{residual_rotation_cache_tag(mode)}.pt"


def quantized_weight_cache_name(prefix: str, mode: str) -> str:
    """Insert the residual-mode tag before the historical ``_model.pt`` suffix."""
    return f"{prefix}{residual_rotation_cache_tag(mode)}_model.pt"


def identity_rotation_metadata(cfg: dict) -> dict[str, int | float]:
    """Derive the exact PCA high-tail split without loading random R tensors."""
    high_fraction = cfg["rotation"]["high_fraction"]
    dims = cfg["dims"]
    return {
        "high_len_hidden": round(high_fraction * dims["hidden"]),
        "high_len_head": round(high_fraction * dims["head"]),
        "high_len_down": round(high_fraction * dims["intermediate"]),
        "high_fraction": high_fraction,
    }


def load_model_config(model_name: str):
    """Load config from models/<model_name>/config.yaml and return derived constants."""
    cfg_path = _ROOT / "models" / model_name / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg



def hash_str_to_int(s: str) -> int:
    """Deterministic seeding matching deepcompressor's hash_str_to_int."""
    modulus = 10**9 + 7
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def apply_dirotq_to_model(transformer, basis_dict, rotation_dict, cfg, assign_online_rotations,
                           skip_layers=None, hadamard_layers=None, sign_flips_dict=None,
                           pca_only_layers=None, residual_rotation="random"):
    """Wrap linear layers with ActQuantWrapper and assign online PCA rotations."""
    if skip_layers is None:
        skip_layers = cfg["quantization"]["skip_layers"]
    transformer.eval()
    print("Wrapping linear layers with ActQuantWrapper...")
    add_actquant(transformer, skip_names=skip_layers)
    qlayers = find_qlayers(transformer, layers=[ActQuantWrapper])
    print(f"Wrapped {len(qlayers)} layers with ActQuantWrapper.")
    pca_only_layers = pca_only_layers or []
    if pca_only_layers:
        print(f"PCA-only layers (permutation instead of rotation): {pca_only_layers!r}")
    print("Assigning online PCA rotations to quantizers...")
    assign_online_rotations(transformer, basis_dict, rotation_dict, cfg,
                             hadamard_layers=hadamard_layers or [],
                             sign_flips_dict=sign_flips_dict or {},
                             pca_only_layers=pca_only_layers,
                             residual_rotation=residual_rotation)
    return transformer


def generate_images(pipeline, output_dir, dataset_json, generation_params, max_images=None,
                    batch_size=1, save_images=True, audit_controller=None):
    """Generate images with deterministic seeding, skipping existing ones."""
    with open(dataset_json) as f:
        samples = json.load(f)

    output_dir = Path(output_dir)
    target_samples = list(samples.items())
    if max_images is not None:
        target_samples = target_samples[:max_images]
        print(f"Limiting to the first {max_images} images.")

    if save_images:
        existing = sum(1 for img_id, info in target_samples
                       if (output_dir / info["category"] / f"{img_id}.png").exists())
        print(f"Found {existing}/{len(target_samples)} target images already generated.")
        to_generate = [(img_id, info) for img_id, info in target_samples
                       if not (output_dir / info["category"] / f"{img_id}.png").exists()]
    else:
        print("Stats-only generation: images will not be decoded or saved.")
        to_generate = target_samples

    if not to_generate:
        print("Nothing to generate.")
        return

    print(f"Generating {len(to_generate)} images (batch_size={batch_size})...")
    pipeline.set_progress_bar_config(disable=True)

    num_batches = (len(to_generate) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches)):
        batch = to_generate[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        prompts = [info["prompt"] for _, info in batch]
        seeds = [hash_str_to_int(img_id) for img_id, _ in batch]
        diffusers.training_utils.set_seed(seeds[0])
        generators = [torch.Generator().manual_seed(seed) for seed in seeds]

        if save_images:
            for _, info in batch:
                (output_dir / info["category"]).mkdir(parents=True, exist_ok=True)

        if audit_controller is not None:
            audit_controller.start_batch(batch)
        try:
            with torch.no_grad():
                result = pipeline(
                    prompts,
                    generator=generators,
                    **({} if save_images else {"output_type": "latent"}),
                    **generation_params,
                )
        finally:
            if audit_controller is not None:
                audit_controller.end_batch()

        if save_images:
            for (img_id, info), image in zip(batch, result.images):
                out_path = output_dir / info["category"] / f"{img_id}.png"
                image.save(out_path)

    if save_images:
        print(f"Done. Images saved to {output_dir}")
    else:
        print("Done. Stats-only generation completed without image output.")


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pixart-sigma",
                        help="Model name (subdirectory under models/, default: pixart-sigma)")
    parser.add_argument(
        "--model-id", default=None,
        help=("Override the configured Hugging Face ID with an immutable local "
              "snapshot or alternate model ID."),
    )
    parser.add_argument("--dataset", default=str(_ROOT / "datasets" / "mjhq_5000_samples.json"),
                        help="Path to dataset JSON (default: datasets/mjhq_5000_samples.json)")
    parser.add_argument(
        "--basis-path", default=None,
        help=("Override the model PCA artifact. Shared-basis experiments must "
              "use a derived artifact carrying __shared_basis_map__; the file "
              "hash is included in cache/output names."),
    )
    parser.add_argument(
        "--calib-dir", default=None,
        help=("Override the read-only calibration activation-cache directory. "
              "Useful for large FLUX experiments stored outside the repository."),
    )
    parser.add_argument(
        "--rotation-path", default=None,
        help=("Override the residual-rotation artifact. The default remains the "
              "model config path; experiments should pass an immutable file."),
    )
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: models/pixart-sigma/generated_images_gptq or models/pixart-sigma/generated_images_rtn)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Generate only N images (for testing)")
    parser.add_argument("--generate", action="store_true", default=True)
    parser.add_argument("--no-generate", action="store_false", dest="generate")
    parser.add_argument(
        "--fp16-reference", action="store_true",
        help=("Generate from the untouched fp16 pipeline: no ActQuantWrapper, "
              "rotation, activation quantization, or weight quantization"),
    )
    parser.add_argument("--gptq", action="store_true", default=False,
                        help="Use GPTQ weight quantization instead of RTN")
    parser.add_argument("--nvfp4", action="store_true", default=False,
                        help="Use NF4 (FP4 E2M1) quantization instead of INT4")
    parser.add_argument(
        "--activation-format",
        choices=("nvfp4", "nvfp4-hw", "e0m3", "block-mix-oracle",
                 "tile-mix-oracle", "tile-mix-output-oracle",
                 "a16w4-residual", "e0a-w16-residual", "nvfp4-4over6",
                 "e0m3-gscale1536", "tile-mix-e0-e2-4over6"),
        default="nvfp4",
        help=("Activation fake quant used with --nvfp4 weights: nvfp4 is the "
              "legacy DiRotQ E2M1 baseline; nvfp4-hw is fixed E2M1 with one "
              "FP32 global scale and E4M3 group scales; e0m3 and all oracle "
              "modes share exactly the nvfp4-hw scale hierarchy; "
              "tile-mix-output-oracle is a local partial-output accuracy oracle "
              "and a16w4-residual keeps residual activations in native FP16/BF16 "
              "while retaining PCA/random-R and cached GPTQ W4 weights "
              "; e0a-w16-residual is a guarded ceiling with fixed hardware "
              "E0M3 residual activations and original transformed native-dtype "
              "low weights "
              "; nvfp4-4over6 is paper-faithful per-block M=4/M=6 E2M1 "
              "selection with global denominator 1536, e0m3-gscale1536 is its "
              "fair fixed-E0 comparator, and tile-mix-e0-e2-4over6 selects "
              "between that E0 candidate and block-adaptive Four Over Six E2 "
              "(default: nvfp4)"),
    )
    parser.add_argument("--a-bits", type=int, default=None,
                        help="Override activation bits (default: from config)")
    parser.add_argument("--a-groupsize", type=int, default=None,
                        help="Override activation groupsize (default: 64 for INT4, 16 for NF4)")
    parser.add_argument("--hadamard-layers", nargs="*", default=None,
                        help="Layer patterns to use Hadamard rotation instead of PCA "
                             "(e.g. --hadamard-layers ff.net.2)")
    parser.add_argument("--sign-flips", default=None,
                        help="Path to optimized sign flips (.pt) for Hadamard layers. "
                             "If not set, uses random sign flips.")
    parser.add_argument("--skip-quant-layers", nargs="*", default=None,
                        help="Layer name patterns to skip activation quantization (bits=16). "
                             "E.g. --skip-quant-layers to_out")
    parser.add_argument("--pca-only-layers", nargs="*", default=None,
                        help="Layer name patterns to use PCA-only channel permutation (O(D) gather) "
                             "instead of full rotation matmul (O(D²)). "
                             "E.g. --pca-only-layers to_q to_k to_v")
    parser.add_argument(
        "--residual-rotation", choices=("random", "identity"), default="random",
        help=("Residual-basis transform after PCA: random preserves the legacy "
              "DiRotQ U@R behavior; identity uses PCA U only (default: random)"),
    )
    parser.add_argument("--gptq-calib-files", type=int, default=5120) # 128 samples x 20 steps
    parser.add_argument("--gptq-batch-size", type=int, default=8)
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--gptq-damp-pct", type=float, default=0.01)
    parser.add_argument(
        "--e0joint-gptq", action="store_true",
        help=("SANA-only feasibility objective min ||AW-Q_E0(A) Q_E2(W)||²; "
              "requires fixed e0m3 activation, random-R, NVFP4 and GPTQ"),
    )
    parser.add_argument(
        "--e0joint-standard-cache", default=None,
        help="Standard SANA NVFP4 E2M1 GPTQ cache used only for objective comparison",
    )
    parser.add_argument(
        "--e0joint-report-dir", default=None,
        help="Output directory for per-layer/aggregate E0-joint objective reports",
    )
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for image generation (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42)")
    parser.add_argument("--quantized-cache", default=None,
                        help="Path to save/load quantized transformer weights. "
                             "If file exists, skips GPTQ/RTN and loads from cache.")
    parser.add_argument("--slow-unrotation", action="store_true",
                        help="Use fp32 fused-unrotation path (debug/fallback). "
                             "Default is the fast bf16/fp16 path.")
    parser.add_argument(
        "--collect-format-stats", action="store_true",
        help="Count global E2M1/E0M3 oracle selections using device-side counters",
    )
    parser.add_argument(
        "--format-stats-output", default=None,
        help="JSON output path for --collect-format-stats",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Run denoising for aggregate activation stats without decoding/saving images",
    )
    parser.add_argument(
        "--real-tile-export-config", default=None,
        help=("SANA-only read-only capture config for receiver-schema-v1 real "
              "FP4 tile packages; requires stats-only fixed e0m3 and an "
              "existing hardware-fixed-e2 cache"),
    )
    parser.add_argument(
        "--distribution-audit-output", default=None,
        help=("Directory for streaming block/tile/timestep diagnostics. Requires "
              "--stats-only and --activation-format tile-mix-oracle."),
    )
    parser.add_argument(
        "--audit-quality-csv", action="append", default=None,
        help="Existing per-prompt metric CSV used only for exploratory correlations",
    )
    parser.add_argument(
        "--weight-mix-build", action="store_true",
        help=("SANA-only calibration command: build common fixed-E0 activation "
              "Hessian plus reoptimized fixed-E2, fixed-E0 and 64x8 Weight "
              "TileMix GPTQ caches, then exit"),
    )
    parser.add_argument(
        "--weight-mix-cache-kind", choices=WEIGHT_MIX_MODES, default=None,
        help="Validate and execute one previously built E0-Hessian weight cache",
    )
    parser.add_argument("--weight-mix-report-dir", default=None)
    parser.add_argument("--weight-mix-hessian-cache", default=None)
    parser.add_argument("--weight-mix-standard-cache", default=None)
    parser.add_argument(
        "--hardware-weight-build", action="store_true",
        help=("SANA-only calibration command: reuse the existing fixed-E0 "
              "activation Hessian and build packing-valid hardware-scaled "
              "fixed E2M1/E0M3 group-16 GPTQ weight caches, then exit"),
    )
    parser.add_argument(
        "--hardware-weight-cache-kind", choices=HARDWARE_WEIGHT_FORMATS, default=None,
        help="Validate and execute one packing-valid hardware fixed-weight cache",
    )
    parser.add_argument("--hardware-weight-report-dir", default=None)
    parser.add_argument("--hardware-weight-hessian-cache", default=None)
    parser.add_argument(
        "--hardware-weight-hessian-sha256", default=None,
        help=("Read-only runtime provenance alternative when the calibration "
              "Hessian is intentionally absent; cache building still requires "
              "--hardware-weight-hessian-cache"),
    )
    parser.add_argument("--hardware-weight-legacy-e2-cache", default=None)
    parser.add_argument("--hardware-weight-legacy-e0-cache", default=None)
    args = parser.parse_args()

    if args.activation_format != "nvfp4" and not args.nvfp4:
        parser.error("non-default --activation-format requires --nvfp4 weights")
    hardware_formats = {
        "nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle",
        "tile-mix-output-oracle",
    } | FOUR_OVER_SIX_FORMATS
    if args.collect_format_stats and args.activation_format not in hardware_formats:
        parser.error("--collect-format-stats requires a hardware-faithful activation format")
    if args.collect_format_stats != bool(args.format_stats_output):
        parser.error("--collect-format-stats and --format-stats-output must be used together")
    if args.fp16_reference and (args.gptq or args.nvfp4 or args.collect_format_stats):
        parser.error("--fp16-reference cannot be combined with quantization or format stats")
    if args.stats_only and not (
        args.collect_format_stats or args.distribution_audit_output
        or args.real_tile_export_config
    ):
        parser.error(
            "--stats-only requires format stats, --distribution-audit-output, "
            "or --real-tile-export-config"
        )
    if args.distribution_audit_output:
        if not args.stats_only:
            parser.error("--distribution-audit-output requires --stats-only")
        if args.activation_format != "tile-mix-oracle":
            parser.error("distribution audit requires the existing SSE tile-mix-oracle trajectory")
        if not (args.gptq and args.nvfp4):
            parser.error("distribution audit requires NVFP4 E2M1 GPTQ weights")
        if not args.audit_quality_csv:
            parser.error("distribution audit requires at least one --audit-quality-csv")
    if args.activation_format == "tile-mix-output-oracle" and not args.gptq:
        parser.error("tile-mix-output-oracle requires GPTQ weights")
    if args.real_tile_export_config:
        if args.model != "sana-1.6b":
            parser.error("real-tile export is restricted to SANA-1.6B")
        if not (args.gptq and args.nvfp4 and args.stats_only):
            parser.error("real-tile export requires --gptq --nvfp4 --stats-only")
        if args.activation_format != "e0m3" or args.residual_rotation != "random":
            parser.error("real-tile export requires fixed e0m3 and random residual rotation")
        if args.hardware_weight_cache_kind != "hardware-fixed-e2":
            parser.error("real-tile export runtime must load hardware-fixed-e2")
        if args.max_images != 1 or args.batch_size != 1:
            parser.error("real-tile export requires --max-images 1 --batch-size 1")
        if args.collect_format_stats or args.distribution_audit_output:
            parser.error("real-tile export cannot be combined with other stats collectors")
        if args.hadamard_layers or args.pca_only_layers or args.skip_quant_layers:
            parser.error("real-tile export requires the unmodified official SANA routing")
    if args.activation_format == "a16w4-residual":
        if not args.gptq:
            parser.error("a16w4-residual requires GPTQ weights")
        if args.residual_rotation != "random":
            parser.error("a16w4-residual requires the random residual rotation basis")
    if args.activation_format == "e0a-w16-residual":
        if not args.gptq:
            parser.error("e0a-w16-residual requires standard GPTQ cache provenance")
        if args.residual_rotation != "random":
            parser.error("e0a-w16-residual requires the random residual rotation basis")
    if args.e0joint_gptq:
        if args.model != "sana-1.6b":
            parser.error("--e0joint-gptq is restricted to SANA-1.6B")
        if not (args.gptq and args.nvfp4):
            parser.error("--e0joint-gptq requires --gptq and --nvfp4")
        if args.activation_format != "e0m3":
            parser.error("--e0joint-gptq requires --activation-format e0m3")
        if args.residual_rotation != "random":
            parser.error("--e0joint-gptq requires random residual rotation")
        if args.hadamard_layers or args.pca_only_layers or args.skip_quant_layers:
            parser.error("--e0joint-gptq requires the unmodified official SANA routing")
    elif args.e0joint_standard_cache or args.e0joint_report_dir:
        parser.error("E0-joint cache/report options require --e0joint-gptq")
    if args.weight_mix_build or args.weight_mix_cache_kind:
        if args.model != "sana-1.6b":
            parser.error("weight Mix feasibility is initially restricted to SANA-1.6B")
        if not (args.gptq and args.nvfp4):
            parser.error("weight Mix feasibility requires --gptq and --nvfp4")
        if args.activation_format != "e0m3":
            parser.error("weight Mix freezes activation at hardware-faithful fixed e0m3")
        if args.residual_rotation != "random":
            parser.error("weight Mix feasibility requires random residual rotation")
        if args.hadamard_layers or args.pca_only_layers or args.skip_quant_layers:
            parser.error("weight Mix feasibility requires unmodified official SANA routing")
        if args.e0joint_gptq:
            parser.error("weight Mix cannot be combined with E0-joint GPTQ")
    elif (args.weight_mix_report_dir or args.weight_mix_hessian_cache or
          args.weight_mix_standard_cache):
        parser.error("weight Mix paths require --weight-mix-build or --weight-mix-cache-kind")
    if args.hardware_weight_build or args.hardware_weight_cache_kind:
        if args.model != "sana-1.6b":
            parser.error("hardware fixed-weight feasibility is restricted to SANA-1.6B")
        if not (args.gptq and args.nvfp4):
            parser.error("hardware fixed weights require --gptq and --nvfp4")
        if args.hardware_weight_build and args.activation_format != "e0m3":
            parser.error("hardware fixed-weight cache build freezes activation at e0m3")
        if args.hardware_weight_cache_kind:
            from utils.asymmetric_tilemix_stats import (
                validate_fixed_e0_weight_activation_format,
            )
            try:
                validate_fixed_e0_weight_activation_format(args.activation_format)
            except ValueError as error:
                parser.error(str(error))
        if args.residual_rotation != "random":
            parser.error("hardware fixed weights require random residual rotation")
        if args.hadamard_layers or args.pca_only_layers or args.skip_quant_layers:
            parser.error("hardware fixed weights require unmodified official SANA routing")
        if args.e0joint_gptq or args.weight_mix_build or args.weight_mix_cache_kind:
            parser.error("hardware fixed weights cannot be combined with other weight experiments")
    elif (args.hardware_weight_report_dir or args.hardware_weight_hessian_cache or
          args.hardware_weight_hessian_sha256 or
          args.hardware_weight_legacy_e2_cache or args.hardware_weight_legacy_e0_cache):
        parser.error("hardware weight paths require a hardware weight build/runtime mode")
    identity_models = {"pixart-sigma", "sana-1.6b"}
    if args.residual_rotation != "random" and args.model not in identity_models:
        parser.error(
            "identity residual rotation is currently supported only for "
            f"{sorted(identity_models)}"
        )

    # Strip any accidental whitespace from list arguments (e.g. backslash-space in shell)
    if args.pca_only_layers:
        args.pca_only_layers = [p.strip() for p in args.pca_only_layers if p.strip()]
    if args.skip_quant_layers:
        args.skip_quant_layers = [p.strip() for p in args.skip_quant_layers if p.strip()]
    if args.hadamard_layers:
        args.hadamard_layers = [p.strip() for p in args.hadamard_layers if p.strip()]

    if args.slow_unrotation:
        from dirotq_fused_unrotation import patch_forward, preconvert_rotations_to_device
    else:
        from dirotq_fused_unrotation_fast import (
            patch_forward_fast as patch_forward,
            preconvert_rotations_to_device,
        )

    cfg = load_model_config(args.model)

    _spec = importlib.util.spec_from_file_location(
        "model_utils", _ROOT / "models" / args.model / "model_utils.py")
    model_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(model_utils)
    assign_online_rotations = model_utils.assign_online_rotations
    configure_quantizers_by_name = model_utils.configure_quantizers_by_name
    generation_params = model_utils.generation_params
    preprocess_transformer = getattr(model_utils, "preprocess_transformer", None)

    model_id      = args.model_id or cfg["model_id"]
    default_basis_path = str(_ROOT / cfg["basis"]["output_path"])
    basis_path    = str(Path(args.basis_path).resolve()) if args.basis_path else default_basis_path
    rotation_path = (
        str(Path(args.rotation_path).resolve())
        if args.rotation_path else str(_ROOT / cfg["rotation"]["output_path"])
    )
    calib_dir     = (
        str(Path(args.calib_dir).resolve())
        if args.calib_dir else str(_ROOT / cfg["calib"]["cache_dir"])
    )
    w_bits        = cfg["quantization"]["w_bits"]

    # Override activation bits if specified on CLI
    if args.a_bits is not None:
        cfg["quantization"]["a_bits"] = args.a_bits

    a_bits = cfg["quantization"]["a_bits"]

    # Select skip layers and weight groupsize based on quant format
    if args.nvfp4:
        nvfp4_cfg   = cfg.get("nvfp4", {})
        w_groupsize = nvfp4_cfg.get("w_groupsize", 16)
        skip_layers = nvfp4_cfg.get("skip_layers", cfg["quantization"]["skip_layers"])
        fmt_tag     = "nvfp4"
    else:
        w_groupsize = cfg["quantization"]["w_groupsize"]
        skip_layers = cfg["quantization"]["skip_layers"]
        fmt_tag     = "int4"

    # Activation bits tag for cache/output naming (omit if default 4)
    a_tag = f"_a{a_bits}" if a_bits != 4 else ""
    had_tag = "_had" if args.hadamard_layers else ""
    if args.sign_flips:
        had_tag += "_optflips"
    # Skip tag encodes --skip-quant-layers in both the cache and output names.
    # The cache must include it because weight quant only rotates+splits when
    # bits<16 — skipped layers have unfused weights, so a cache built with a
    # different skip config can't be reused.
    import hashlib
    if args.basis_path:
        basis_file = Path(basis_path)
        if not basis_file.is_file():
            parser.error(f"--basis-path does not exist: {basis_file}")
        digest = hashlib.sha256()
        with basis_file.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        basis_tag = "_basis" + digest.hexdigest()[:12]
    else:
        basis_tag = ""
    if args.skip_quant_layers:
        key = ",".join(sorted(args.skip_quant_layers))
        skip_tag = "_skip" + hashlib.md5(key.encode()).hexdigest()[:8]
    else:
        skip_tag = ""

    # pca_tag encodes --pca-only-layers: weight columns are permuted (not rotated)
    # for matching layers, so caches from different pca configs are incompatible.
    if args.pca_only_layers:
        key = ",".join(sorted(args.pca_only_layers))
        pca_tag = "_pca" + hashlib.md5(key.encode()).hexdigest()[:8]
    else:
        pca_tag = ""

    # Activation format normally changes runtime fake quantization only and is
    # absent from the weight key.  E0-joint is the sole explicit exception: its
    # offline weight objective depends on E0 activations and must be isolated.
    act_tag = "" if args.activation_format == "nvfp4" else f"_act-{args.activation_format}"
    if args.e0joint_gptq:
        act_tag += "_e0joint"
    residual_tag = residual_rotation_cache_tag(args.residual_rotation)

    if args.quantized_cache is None:
        if args.hardware_weight_cache_kind:
            cache_name = f"nvfp4_g16_e0h_{args.hardware_weight_cache_kind}_gptq_model.pt"
        elif args.weight_mix_cache_kind:
            cache_name = (
                f"nvfp4_g16_e0h_weightmix_{args.weight_mix_cache_kind}_gptq_model.pt"
            )
        elif args.e0joint_gptq:
            cache_name = "nvfp4_g16_e0joint_gptq_model.pt"
        else:
            method = "gptq" if args.gptq else "rtn"
            cache_prefix = f"{fmt_tag}_g{w_groupsize}_{method}{a_tag}{had_tag}{skip_tag}{pca_tag}{basis_tag}"
            cache_name = quantized_weight_cache_name(cache_prefix, args.residual_rotation)
        args.quantized_cache = str(_ROOT / "models" / args.model / "quantized_cache" / cache_name)
    elif args.residual_rotation == "identity" and "rr-identity" not in Path(args.quantized_cache).name:
        parser.error("identity residual rotation requires an identity-tagged quantized cache")
    try:
        validate_e0joint_cache_path(Path(args.quantized_cache), args.e0joint_gptq)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.output_dir is None:
        method = "gptq" if args.gptq else "rtn"
        args.output_dir = str(_ROOT / "models" / args.model / f"generated_images_{fmt_tag}_{method}{a_tag}{had_tag}{skip_tag}{pca_tag}{basis_tag}{residual_tag}{act_tag}")

    if args.nvfp4:
        print(f"Activation fake-quant format: {args.activation_format}")

    pipeline_cls_path = cfg["pipeline_class"]
    mod_name, cls_name = pipeline_cls_path.rsplit(".", 1)
    PipelineClass = getattr(importlib.import_module(mod_name), cls_name)

    dtype_str = cfg.get("dtype", "fp16")
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]
    print(f"Loading pipeline ({pipeline_cls_path}) in {dtype_str}...")
    pipe = PipelineClass.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )

    # Exit before reading DiRotQ basis/rotation caches, add_actquant(), or any
    # weight quantizer.  This is the untouched full-precision reference path.
    if args.fp16_reference:
        wrapped = sum(isinstance(mod, ActQuantWrapper) for mod in pipe.transformer.modules())
        if wrapped != 0:
            raise RuntimeError(f"fp16 reference unexpectedly contains {wrapped} ActQuantWrapper layers")
        print("FP16 reference verified: 0 ActQuantWrapper layers; skipping rotation and quantization.")
        if preprocess_transformer is not None:
            preprocess_transformer(pipe.transformer, cfg)
        if args.generate:
            print("Enabling model CPU offload for untouched fp16 pipeline...")
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                print(f"enable_model_cpu_offload failed ({e}); falling back to pipe.to('cuda').")
                pipe = pipe.to("cuda")
            generate_images(
                pipe, args.output_dir, args.dataset, generation_params,
                max_images=args.max_images, batch_size=args.batch_size,
            )
        wrapped = sum(isinstance(mod, ActQuantWrapper) for mod in pipe.transformer.modules())
        if wrapped != 0:
            raise RuntimeError(f"fp16 reference ended with {wrapped} ActQuantWrapper layers")
        print("FP16 reference complete: 0 ActQuantWrapper layers.")
        print("All done.")
        raise SystemExit(0)

    if args.residual_rotation == "random" and not os.path.exists(rotation_path):
        print(f"Rotation file not found at {rotation_path}, generating...")
        subprocess.run([sys.executable, str(_ROOT / "gen_rotation.py"), "--model", args.model], check=True)

    if not os.path.exists(basis_path):
        print(f"PCA basis not found at {basis_path}, generating...")
        subprocess.run([sys.executable, str(_ROOT / "get_basis.py"), "--model", args.model], check=True)

    print(f"Loading PCA basis from {basis_path}...")
    basis_dict = torch.load(basis_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(basis_dict)} basis matrices.")
    shared_basis_scheme = basis_dict.get("__shared_basis_scheme__")
    if args.basis_path and shared_basis_scheme is None:
        print("WARNING: overridden basis has no shared-basis metadata; it will "
              "remain cache-isolated but is not claimed as a shared-memory arm.")
    elif shared_basis_scheme is not None:
        print(f"Shared PCA basis scheme: {shared_basis_scheme}")

    if args.residual_rotation == "random":
        print(f"Loading rotations from {rotation_path}...")
        rotation_dict = torch.load(rotation_path, map_location="cpu", weights_only=False)
        print("Rotations:", {k: v.shape if isinstance(v, torch.Tensor) else v
                             for k, v in rotation_dict.items()})
    else:
        # Identity/PCA-only mode needs only the split lengths.  Derive them
        # from the same config/rounding as gen_rotation.py so no R tensor is
        # loaded and then ignored.
        rotation_dict = identity_rotation_metadata(cfg)
        print("Identity residual basis: random R tensors were not loaded; "
              f"split metadata={rotation_dict}")

    sign_flips_dict = None
    if args.sign_flips:
        print(f"Loading optimized sign flips from {args.sign_flips}...")
        sign_flips_dict = torch.load(args.sign_flips, map_location="cpu", weights_only=False)
        print(f"Loaded optimized sign flips for {len(sign_flips_dict)} blocks.")

    if preprocess_transformer is not None:
        preprocess_transformer(pipe.transformer, cfg)

    apply_dirotq_to_model(pipe.transformer, basis_dict, rotation_dict, cfg, assign_online_rotations,
                           skip_layers=skip_layers, hadamard_layers=args.hadamard_layers,
                           sign_flips_dict=sign_flips_dict,
                           pca_only_layers=args.pca_only_layers,
                           residual_rotation=args.residual_rotation)

    high_len_hidden = rotation_dict["high_len_hidden"]
    high_len_head   = rotation_dict["high_len_head"]
    high_len_down   = rotation_dict.get("high_len_down", 0)

    format_stats = None
    if args.collect_format_stats:
        unit = ({"block-mix-oracle": "block", "tile-mix-oracle": "tile",
                 "tile-mix-output-oracle": "tile", "nvfp4-4over6": "block",
                 "tile-mix-e0-e2-4over6": "tile"}.get(
                     args.activation_format, "fixed"
                 ))
        if args.activation_format in FOUR_OVER_SIX_FORMATS:
            stats_cls = FourOverSixStats
        elif args.activation_format == "tile-mix-output-oracle":
            stats_cls = OutputOracleFormatStats
        elif args.activation_format == "tile-mix-oracle":
            from utils.asymmetric_tilemix_stats import TileMixTrajectoryStats
            stats_cls = TileMixTrajectoryStats
        else:
            stats_cls = FormatSelectionStats
        format_stats = stats_cls(selection_unit=unit)

    configure_quantizers_by_name(pipe.transformer, high_len_hidden, high_len_head, cfg,
                                 nvfp4=args.nvfp4, hadamard_layers=args.hadamard_layers or [],
                                 a_groupsize=args.a_groupsize, high_len_down=high_len_down,
                                 skip_quant_layers=args.skip_quant_layers or [],
                                 activation_format=args.activation_format,
                                 format_stats=format_stats)

    if args.hardware_weight_build:
        if args.generate:
            print("Hardware fixed-weight build is calibration-only; image generation is disabled.")
        quantized_dir = _ROOT / "models" / args.model / "quantized_cache"
        report_dir = Path(
            args.hardware_weight_report_dir or
            (_ROOT / "models" / args.model / "hardware_fixed_weight" / "calibration")
        )
        hessian_cache = Path(
            args.hardware_weight_hessian_cache or
            (quantized_dir / "hessians_e0a_n5120_l120_rr-random_g16_tile64x8.pt")
        )
        cache_paths = {
            fmt: quantized_dir / f"nvfp4_g16_e0h_{fmt}_gptq_model.pt"
            for fmt in HARDWARE_WEIGHT_FORMATS
        }
        legacy_cache_paths = {
            "hardware-fixed-e2": Path(
                args.hardware_weight_legacy_e2_cache or
                (quantized_dir / "nvfp4_g16_e0h_weightmix_fixed-e2_gptq_model.pt")
            ),
            "hardware-fixed-e0": Path(
                args.hardware_weight_legacy_e0_cache or
                (quantized_dir / "nvfp4_g16_e0h_weightmix_fixed-e0_gptq_model.pt")
            ),
        }
        print("Moving transformer to CUDA for matched hardware fixed-weight GPTQ...")
        pipe.transformer = pipe.transformer.to("cuda")
        result = build_hardware_fixed_caches(
            pipe.transformer,
            hessian_cache=hessian_cache,
            cache_paths=cache_paths,
            legacy_cache_paths=legacy_cache_paths,
            report_dir=report_dir,
            basis_path=Path(basis_path),
            rotation_path=Path(rotation_path),
            skip_layers=skip_layers,
            num_calib_files=args.gptq_calib_files,
            damp_pct=args.gptq_damp_pct,
            device="cuda",
        )
        print("Hardware fixed-weight aggregate losses:", result["aggregate_losses"])
        raise SystemExit(0)

    if args.weight_mix_build:
        if args.generate:
            print("Weight Mix build is calibration-only; image generation is disabled.")
        report_dir = Path(
            args.weight_mix_report_dir or
            (_ROOT / "models" / args.model / "weight_mix_feasibility" / "calibration")
        )
        quantized_dir = _ROOT / "models" / args.model / "quantized_cache"
        hessian_cache = Path(
            args.weight_mix_hessian_cache or
            (quantized_dir / "hessians_e0a_n5120_l120_rr-random_g16_tile64x8.pt")
        )
        standard_cache = Path(
            args.weight_mix_standard_cache or
            (quantized_dir / "nvfp4_g16_gptq_model.pt")
        )
        cache_paths = {
            mode: quantized_dir / f"nvfp4_g16_e0h_weightmix_{mode}_gptq_model.pt"
            for mode in WEIGHT_MIX_MODES
        }
        print("Moving transformer to CUDA for fixed-E0 Hessian/Weight Mix GPTQ...")
        pipe.transformer = pipe.transformer.to("cuda")
        result = build_weight_mix_caches(
            pipe.transformer,
            calib_dir=calib_dir,
            hessian_cache=hessian_cache,
            cache_paths=cache_paths,
            standard_cache=standard_cache,
            report_dir=report_dir,
            basis_path=Path(basis_path),
            rotation_path=Path(rotation_path),
            skip_layers=skip_layers,
            num_calib_files=args.gptq_calib_files,
            batch_size=args.gptq_batch_size,
            damp_pct=args.gptq_damp_pct,
            device="cuda",
        )
        print("Weight Mix calibration gate:", result["calibration_gate"])
        raise SystemExit(0)

    transformed_w16 = None
    if args.activation_format == "e0a-w16-residual":
        # Capture before loading the standard quantized state dict.  This is
        # the only point at which the original model weight and the configured
        # PCA/random-R split coexist.  The helper refuses unrotated wrappers.
        transformed_w16 = capture_transformed_w16_weights(pipe.transformer)
        print(
            f"Captured {len(transformed_w16)} original transformed native-dtype "
            "W16 weights before standard-cache validation."
        )

    cache_path = Path(args.quantized_cache)
    cache_required_formats = {
        "tile-mix-output-oracle", "a16w4-residual", "e0a-w16-residual",
        *FOUR_OVER_SIX_FORMATS,
    }
    if args.activation_format in cache_required_formats and not cache_path.exists():
        raise FileNotFoundError(
            f"{args.activation_format} requires an existing NVFP4 E2M1 GPTQ "
            f"weight cache; refusing to create a new cache at {cache_path}"
        )
    if args.weight_mix_cache_kind:
        if not cache_path.exists():
            raise FileNotFoundError(f"missing requested weight-Mix cache: {cache_path}")
        validate_weight_mix_metadata(
            cache_path,
            expected_weight_mix_metadata(
                model=args.model,
                mode=args.weight_mix_cache_kind,
                calibration_count=args.gptq_calib_files,
                damp_pct=args.gptq_damp_pct,
                basis_path=Path(basis_path),
                rotation_path=Path(rotation_path),
                skip_layers=skip_layers,
            ),
        )
        print(
            f"Validated E0-Hessian weight cache kind={args.weight_mix_cache_kind}: "
            f"{cache_path}"
        )
    if args.hardware_weight_cache_kind:
        if not cache_path.exists():
            raise FileNotFoundError(f"missing requested hardware weight cache: {cache_path}")
        quantized_dir = _ROOT / "models" / args.model / "quantized_cache"
        hessian_cache = Path(
            args.hardware_weight_hessian_cache or
            (quantized_dir / "hessians_e0a_n5120_l120_rr-random_g16_tile64x8.pt")
        )
        if hessian_cache.exists():
            expected_hessian_path = hessian_cache
            expected_hessian_sha256 = args.hardware_weight_hessian_sha256
        elif args.hardware_weight_hessian_sha256:
            expected_hessian_path = None
            expected_hessian_sha256 = args.hardware_weight_hessian_sha256
            print(
                "Validating hardware weight cache against manifest-provided "
                f"Hessian SHA-256 {expected_hessian_sha256}; Hessian file is absent."
            )
        else:
            raise FileNotFoundError(
                "hardware weight provenance requires the Hessian cache or "
                "--hardware-weight-hessian-sha256"
            )
        validate_hardware_weight_metadata(
            cache_path,
            expected_hardware_weight_metadata(
                model=args.model,
                fmt=args.hardware_weight_cache_kind,
                calibration_count=args.gptq_calib_files,
                damp_pct=args.gptq_damp_pct,
                basis_path=Path(basis_path),
                rotation_path=Path(rotation_path),
                hessian_cache=expected_hessian_path,
                hessian_cache_sha256=expected_hessian_sha256,
                skip_layers=skip_layers,
            ),
        )
        print(
            f"Validated packing-valid hardware weight cache "
            f"kind={args.hardware_weight_cache_kind}: {cache_path}"
        )
    if cache_path.exists():
        if args.e0joint_gptq:
            validate_e0joint_metadata(cache_path, {
                "objective_version": E0JOINT_OBJECTIVE_VERSION,
                "activation_format": "e0m3",
                "weight_format": "nvfp4-e2m1",
                "groupsize": 16,
                "calibration_count": args.gptq_calib_files,
                "damp_pct": args.gptq_damp_pct,
                "residual_rotation": "random",
                "basis_sha256": sha256_file(Path(basis_path)),
                "rotation_sha256": sha256_file(Path(rotation_path)),
                "active_layers": 120,
            })
        print(f"Loading quantized weights from cache: {cache_path}")
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        model_keys = set(pipe.transformer.state_dict().keys())
        cache_keys = set(state.keys())
        # Quantizer scale/zero are transient (recomputed each forward pass),
        # so they are expected to be absent from the cache.
        _transient = {".quantizer.scale", ".quantizer.zero"}
        missing = {k for k in (model_keys - cache_keys)
                   if not any(k.endswith(s) for s in _transient)}
        unexpected = cache_keys - model_keys
        if args.activation_format == "a16w4-residual":
            required_cached_weights = {
                f"{name}.module.weight"
                for name, mod in pipe.transformer.named_modules()
                if isinstance(mod, ActQuantWrapper) and mod.quantizer.bits < 16
            }
            missing_cached_weights = required_cached_weights - cache_keys
            if missing_cached_weights:
                preview = ", ".join(sorted(missing_cached_weights)[:5])
                raise RuntimeError(
                    "a16w4-residual refuses to use original W_low; the GPTQ "
                    f"cache is missing {len(missing_cached_weights)} active weights: "
                    f"{preview}"
                )
        if missing:
            print(f"WARNING: {len(missing)} keys in model but not in cache (will keep init weights):")
            for k in sorted(missing)[:10]:
                print(f"  {k}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        if unexpected:
            print(f"WARNING: {len(unexpected)} keys in cache but not in model (stale cache?):")
            for k in sorted(unexpected)[:10]:
                print(f"  {k}")
            if len(unexpected) > 10:
                print(f"  ... and {len(unexpected) - 10} more")
        if args.hardware_weight_cache_kind:
            runtime_audit = validate_hardware_weight_runtime(
                pipe.transformer, state, cache_path, args.hardware_weight_cache_kind
            )
            print(
                "Verified packed payload/E4M3 scales/global scale against "
                f"reconstructed runtime weights: {runtime_audit}"
            )
        pipe.transformer.load_state_dict(state, strict=False)
        for _, mod in pipe.transformer.named_modules():
            if isinstance(mod, ActQuantWrapper) and (
                mod.rotation is not None or
                mod.rotation_per_head is not None or
                getattr(mod, 'use_hadamard', False) or
                getattr(mod, 'perm_idx', None) is not None
            ):
                mod._unrot_fused = True
        if args.activation_format == "tile-mix-output-oracle":
            armed = 0
            for _, mod in pipe.transformer.named_modules():
                if isinstance(mod, ActQuantWrapper) and mod.quantizer.bits < 16:
                    mod.output_oracle_weight_ready = True
                    armed += 1
            print(f"Armed local partial-output oracle with {armed} executed GPTQ weights.")
        elif args.activation_format == "a16w4-residual":
            armed = 0
            for _, mod in pipe.transformer.named_modules():
                if isinstance(mod, ActQuantWrapper) and mod.quantizer.bits < 16:
                    mod.a16w4_weight_ready = True
                    armed += 1
            print(
                f"Armed A16W4 residual ceiling on {armed} layers with loaded "
                f"NVFP4 E2M1 GPTQ weights; native activation dtype={torch_dtype}."
            )
        elif args.activation_format == "e0a-w16-residual":
            armed = install_transformed_w16_weights(
                pipe.transformer, transformed_w16, state
            )
            transformed_w16 = None
            print(
                f"Armed E0A x W16 residual ceiling on {armed} layers: fixed "
                f"hardware E0M3 activation, original transformed {torch_dtype} "
                "low weight; standard E2 GPTQ cache used only for verified "
                "coverage/provenance."
            )
    else:
        e0joint_result = None
        if args.e0joint_gptq:
            print("Moving transformer to CUDA for E0-aware joint GPTQ calibration...")
            pipe.transformer = pipe.transformer.to("cuda")
            standard_cache = Path(
                args.e0joint_standard_cache or
                (_ROOT / "models" / args.model / "quantized_cache" /
                 "nvfp4_g16_gptq_model.pt")
            )
            report_dir = Path(
                args.e0joint_report_dir or
                (_ROOT / "models" / args.model / "e0joint_gptq")
            )
            e0joint_result = build_e0joint_gptq(
                pipe.transformer,
                calib_dir=calib_dir,
                standard_cache=standard_cache,
                report_dir=report_dir,
                basis_path=Path(basis_path),
                rotation_path=Path(rotation_path),
                num_calib_files=args.gptq_calib_files,
                batch_size=args.gptq_batch_size,
                damp_pct=args.gptq_damp_pct,
                block_size=args.gptq_block_size,
                groupsize=w_groupsize,
                device="cuda",
            )
            if not e0joint_result["cache_created"]:
                print(
                    "E0-joint continuous gate failed: "
                    f"R_c={e0joint_result['aggregate_r_continuous']:.6f} < "
                    f"{e0joint_result['early_stop_threshold']:.2f}; "
                    "no joint cache or images were created."
                )
                sys.exit(0)
        elif args.gptq:
            print("Moving transformer to CUDA for GPTQ calibration...")
            pipe.transformer = pipe.transformer.to("cuda")

            qlayers = find_qlayers(pipe.transformer, layers=[ActQuantWrapper])
            hessian_name = gptq_hessian_cache_name(
                args.gptq_calib_files, len(qlayers), args.residual_rotation
            )
            hessian_cache = Path(args.quantized_cache).parent / hessian_name
            if hessian_cache.exists():
                print(f"Loading cached Hessians from {hessian_cache}...")
                hessians = torch.load(hessian_cache, map_location="cpu", weights_only=False)
                print(f"Loaded Hessians for {len(hessians)} layers.")
            else:
                print(f"Collecting GPTQ Hessians from {args.gptq_calib_files} calibration files...")
                hessians = collect_hessians(
                    pipe.transformer,
                    calib_dir=calib_dir,
                    device="cuda",
                    num_calib_files=args.gptq_calib_files,
                    batch_size=args.gptq_batch_size,
                )
                hessian_cache.parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving Hessians to {hessian_cache}...")
                torch.save(hessians, hessian_cache)
                print("Hessians cached.")

            print(f"Applying GPTQ weight quantization (W4, {fmt_tag}, group_size={w_groupsize})...")
            gptq_quantize_weights(
                pipe.transformer, hessians,
                bits=w_bits, groupsize=w_groupsize, sym=True,
                skip_names=skip_layers,
                damp_pct=args.gptq_damp_pct,
                block_size=args.gptq_block_size,
                device="cuda",
                nvfp4=args.nvfp4,
            )
            del hessians
        elif args.nvfp4:
            print(f"Applying NF4 RTN weight quantization (group_size={w_groupsize})...")
            nvfp4_rtn_quantize_weights(pipe.transformer, groupsize=w_groupsize,
                                       skip_names=skip_layers)
        else:
            print(f"Applying INT4 RTN weight quantization (group_size={w_groupsize})...")
            rtn_quantize_weights(pipe.transformer, bits=w_bits, groupsize=w_groupsize,
                                 sym=True, skip_names=skip_layers)

        # Clear transient quantizer buffers before saving — they are
        # recomputed by find_params() on every forward pass.
        for _, mod in pipe.transformer.named_modules():
            if isinstance(mod, ActQuantWrapper):
                mod.quantizer.free()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving quantized weights to cache: {cache_path}")
        torch.save(pipe.transformer.state_dict(), cache_path)
        if e0joint_result is not None:
            metadata = {
                **e0joint_result,
                "cache_path": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
            }
            metadata_path = write_e0joint_metadata(cache_path, metadata)
            print(f"Saved E0-joint cache metadata: {metadata_path}")

    real_tile_export = None
    if args.real_tile_export_config:
        if not cache_path.exists():
            raise FileNotFoundError(
                "real-tile export refuses to build a missing quantized cache"
            )
        from utils.real_tile_export import RealTileExportController
        real_tile_export = RealTileExportController.from_file(
            args.real_tile_export_config, pipe.transformer, pipe, state
        )
        print(
            "Armed read-only real-tile capture for "
            f"{len(real_tile_export.specs)} allowlisted cases; both packing "
            "sidecars were validated against their reconstructed BF16 caches."
        )

    distribution_audit = None
    if args.distribution_audit_output:
        if not cache_path.exists():
            raise FileNotFoundError(
                "distribution audit refuses to create a new weight cache: "
                f"{cache_path}"
            )
        from utils.distribution_audit import DistributionAuditCollector
        distribution_audit = DistributionAuditCollector(
            model_name=args.model,
            output_dir=args.distribution_audit_output,
            dataset_path=args.dataset,
            quality_csvs=args.audit_quality_csv,
            quantized_cache=cache_path,
        )
        distribution_audit.attach(pipe.transformer, pipe)
        print(
            f"Attached streaming distribution audit to "
            f"{len(distribution_audit.layer_names)} active quantized layers."
        )

    if format_stats is not None and hasattr(format_stats, "attach_timestep_source"):
        format_stats.attach_timestep_source(pipe.transformer, pipe)
        print("Attached format-stat sidecar to true transformer timesteps.")

    patch_forward()

    if args.generate:
        print("Enabling model CPU offload (text encoders/VAE swap to CUDA on demand)...")
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"enable_model_cpu_offload failed ({e}); falling back to pipe.to('cuda').")
            pipe = pipe.to("cuda")
        preconvert_rotations_to_device(pipe.transformer, device="cuda")
        if shared_basis_scheme is not None:
            from utils.shared_pca_basis import rotation_storage_report
            print("Shared rotation storage:", json.dumps(
                rotation_storage_report(pipe.transformer), sort_keys=True
            ))
        generate_images(
            pipe, args.output_dir, args.dataset, generation_params,
            max_images=args.max_images, batch_size=args.batch_size,
            save_images=not args.stats_only,
            audit_controller=real_tile_export or distribution_audit or (
                format_stats if hasattr(format_stats, "start_batch") else None
            ),
        )
        if torch.cuda.is_available():
            print(
                "Process peak CUDA memory: "
                f"allocated={torch.cuda.max_memory_allocated()} bytes, "
                f"reserved={torch.cuda.max_memory_reserved()} bytes"
            )

    if real_tile_export is not None:
        real_tile_report = real_tile_export.finalize()
        print(
            "Real-tile packages complete: "
            f"{json.dumps(real_tile_report['packages'], sort_keys=True)}"
        )
        if torch.cuda.is_available():
            print(
                "Real-tile peak CUDA allocated bytes: "
                f"{torch.cuda.max_memory_allocated()}"
            )

    if distribution_audit is not None:
        provenance = distribution_audit.finalize()
        print(
            "Distribution audit complete: "
            f"{args.distribution_audit_output}; exclusions="
            f"{provenance['excluded_alignment_events']}"
        )

    if format_stats is not None:
        stats = {
            "activation_format": args.activation_format,
            "residual_rotation": args.residual_rotation,
            **format_stats.snapshot(),
        }
        stats_path = Path(args.format_stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
            f.write("\n")
        if isinstance(format_stats, FourOverSixStats):
            compact = {
                key: stats[key] for key in (
                    "m4_ratio", "e0_tile_ratio", "m4_sse", "m6_sse",
                    "adaptive_sse", "m6_gscale2688_sse",
                    "m6_gscale1536_sse", "tile_selected_sse",
                    "selected_saturation_rate", "reconstruction_sse", "qsnr_db",
                )
            }
            print(f"Four Over Six stats saved to {stats_path}: {compact}")
        else:
            print(f"Format selection stats saved to {stats_path}: {stats}")

    print("All done.")
