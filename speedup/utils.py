"""
speedup/utils.py

Shared helpers for the DiRotQ speed-measurement scripts:

  - setup_pipeline(model, precision)  : build a diffusers pipeline with the
        chosen precision backend ("fp16", "dirotq-fake", "dirotq-torch",
        "dirotq-triton"). Mirrors apply_dirotq.py's __main__ flow but as a
        callable, without modifying any existing DiRotQ files.
  - capture_transformer_inputs(...)   : run one inference step with a
        forward-pre-hook to capture (args, kwargs) for the transformer, so
        per-step latency / per-layer microbenchmarks can replay the real
        runtime input shapes (matches nunchaku/app/.../latency.py "step" mode).
  - run_timed(...)                    : warmup + timed loop with cuda.synchronize
        and outlier trimming (matches nunchaku's latency harness).
  - human_size, fmt_speedup           : table formatting helpers.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import yaml


_DIROTQ_ROOT = Path(__file__).resolve().parent.parent
if str(_DIROTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIROTQ_ROOT))


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

VALID_PRECISIONS = ("fp16", "dirotq-fake", "dirotq-torch", "dirotq-triton")


@dataclass
class PipelineBundle:
    pipeline: Any
    transformer: nn.Module
    config: dict
    generation_params: dict
    precision: str
    high_len_hidden: int = 0
    high_len_head: int = 0
    high_len_down: int = 0


def _load_model_module(model: str):
    cfg_path = _DIROTQ_ROOT / "models" / model / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    spec = importlib.util.spec_from_file_location(
        f"{model}_model_utils", _DIROTQ_ROOT / "models" / model / "model_utils.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return cfg, mod


def _build_pipeline(cfg: dict, device: str) -> Any:
    pipeline_cls_path = cfg["pipeline_class"]
    mod_name, cls_name = pipeline_cls_path.rsplit(".", 1)
    PipelineClass = getattr(importlib.import_module(mod_name), cls_name)

    dtype_str = cfg.get("dtype", "fp16")
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                   "fp32": torch.float32}[dtype_str]
    pipe = PipelineClass.from_pretrained(
        cfg["model_id"], torch_dtype=torch_dtype, use_safetensors=True
    )
    return pipe


def _wrap_dirotq(pipe, cfg: dict, model_utils, *, nvfp4: bool,
                 a_groupsize: int | None, cache_path: Path,
                 require_cache: bool = True) -> tuple[int, int, int]:
    """Wrap transformer with ActQuantWrapper, configure quantizers, load cache.

    Mirrors apply_dirotq.py's __main__ for the "load from cache" path. Returns
    (high_len_hidden, high_len_head, high_len_down).
    """
    from utils.quant_utils import ActQuantWrapper, add_actquant, find_qlayers

    basis_path = _DIROTQ_ROOT / cfg["basis"]["output_path"]
    rotation_path = _DIROTQ_ROOT / cfg["rotation"]["output_path"]
    if not basis_path.exists():
        raise FileNotFoundError(f"PCA basis not found: {basis_path}")
    if not rotation_path.exists():
        raise FileNotFoundError(f"Rotation file not found: {rotation_path}")

    print(f"Loading PCA basis from {basis_path}...")
    basis_dict = torch.load(basis_path, map_location="cpu", weights_only=False)
    print(f"Loading rotations from {rotation_path}...")
    rotation_dict = torch.load(rotation_path, map_location="cpu",
                               weights_only=False)

    if nvfp4:
        skip_layers = cfg.get("nvfp4", {}).get(
            "skip_layers", cfg["quantization"]["skip_layers"])
    else:
        skip_layers = cfg["quantization"]["skip_layers"]

    if hasattr(model_utils, "preprocess_transformer") and \
            model_utils.preprocess_transformer is not None:
        model_utils.preprocess_transformer(pipe.transformer, cfg)

    pipe.transformer.eval()
    add_actquant(pipe.transformer, skip_names=skip_layers)
    n_qlayers = len(find_qlayers(pipe.transformer, layers=[ActQuantWrapper]))
    print(f"Wrapped {n_qlayers} linear layers with ActQuantWrapper.")

    model_utils.assign_online_rotations(
        pipe.transformer, basis_dict, rotation_dict, cfg,
        hadamard_layers=[], sign_flips_dict={},
    )

    high_len_hidden = rotation_dict["high_len_hidden"]
    high_len_head = rotation_dict["high_len_head"]
    high_len_down = rotation_dict.get("high_len_down", 0)

    model_utils.configure_quantizers_by_name(
        pipe.transformer, high_len_hidden, high_len_head, cfg,
        nvfp4=nvfp4, hadamard_layers=[], a_groupsize=a_groupsize,
        high_len_down=high_len_down, skip_quant_layers=[],
    )

    if not cache_path.exists():
        if require_cache:
            raise FileNotFoundError(
                f"Quantized cache not found: {cache_path}. "
                "The speedup harness expects pre-fused quantized weights — "
                "run apply_dirotq.py first to populate the cache."
            )
        print(f"WARNING: cache not found at {cache_path}. Skipping load — "
              "weights remain at their initial values (OK for static "
              "shape/FLOP analysis, NOT for latency measurement).")
    else:
        print(f"Loading quantized weights from cache: {cache_path}")
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        pipe.transformer.load_state_dict(state, strict=False)

    # Mark layers as fused so the patched forward skips unrotation.
    for _, mod in pipe.transformer.named_modules():
        if isinstance(mod, ActQuantWrapper) and (
            mod.rotation is not None or
            mod.rotation_per_head is not None or
            getattr(mod, "use_hadamard", False)
        ):
            mod._unrot_fused = True

    return high_len_hidden, high_len_head, high_len_down


def _default_cache_path(model: str, *, gptq: bool, nvfp4: bool,
                        w_groupsize: int, a_bits: int) -> Path:
    method = "gptq" if gptq else "rtn"
    fmt_tag = "nvfp4" if nvfp4 else "int4"
    a_tag = f"_a{a_bits}" if a_bits != 4 else ""
    cache_name = f"{fmt_tag}_g{w_groupsize}_{method}{a_tag}_model.pt"
    return _DIROTQ_ROOT / "models" / model / "quantized_cache" / cache_name


def setup_pipeline(
    model: str,
    precision: str,
    *,
    device: str = "cuda",
    nvfp4: bool = False,
    gptq: bool = False,
    a_groupsize: int | None = None,
    cache_path: str | None = None,
    enable_cpu_offload: bool = True,
    require_cache: bool = True,
) -> PipelineBundle:
    """Build a diffusers pipeline at the requested precision.

    precision options:
      "fp16"           — vanilla baseline, no DiRotQ wrapping.
      "dirotq-fake"    — DiRotQ + the existing fake-quantization forward
                          (utils/dirotq_fused_unrotation_fast.patch_forward_fast).
                          Useful as the "simulation overhead" baseline.
      "dirotq-torch"   — DiRotQ low-region runs on torch's W4A16 kernel
                          (torch._weight_int4pack_mm). Activations are kept fp16/bf16
                          in the low region; only weights are quantized to int4.
                          Falls back to fake on layers that can't be handled
                          (e.g. per-head rotation_per_head) — those stay fake.
      "dirotq-triton"  — DiRotQ low-region runs on a Triton W4A4 kernel
                          (both activations and weights int4 in the low region).
    """
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"precision must be one of {VALID_PRECISIONS}, got {precision!r}")

    cfg, model_utils = _load_model_module(model)
    print(f"Building {precision} pipeline for {model} ({cfg['model_id']})...")
    pipe = _build_pipeline(cfg, device=device)
    bundle = PipelineBundle(
        pipeline=pipe,
        transformer=pipe.transformer,
        config=cfg,
        generation_params=model_utils.generation_params,
        precision=precision,
    )

    if precision != "fp16":
        if cache_path is None:
            cache = _default_cache_path(
                model, gptq=gptq, nvfp4=nvfp4,
                w_groupsize=cfg["quantization"]["w_groupsize"],
                a_bits=cfg["quantization"]["a_bits"],
            )
        else:
            cache = Path(cache_path)
        h_hidden, h_head, h_down = _wrap_dirotq(
            pipe, cfg, model_utils, nvfp4=nvfp4,
            a_groupsize=a_groupsize, cache_path=cache,
            require_cache=require_cache,
        )
        bundle.high_len_hidden = h_hidden
        bundle.high_len_head = h_head
        bundle.high_len_down = h_down

        if precision == "dirotq-fake":
            from dirotq_fused_unrotation_fast import (
                patch_forward_fast, preconvert_rotations_to_device)
            patch_forward_fast()
            preconvert_rotations_to_device(pipe.transformer, device=device)
        elif precision == "dirotq-torch":
            from speedup.mixed_linear import patch_forward_real
            patch_forward_real(pipe.transformer, backend="torch", device=device)
        elif precision == "dirotq-triton":
            from speedup.mixed_linear import patch_forward_real
            patch_forward_real(pipe.transformer, backend="triton", device=device)

    if enable_cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"enable_model_cpu_offload failed ({e}); using pipe.to({device})")
            pipe.to(device)
    else:
        pipe.to(device)

    return bundle


# ---------------------------------------------------------------------------
# Input capture (for per-step / per-layer microbench)
# ---------------------------------------------------------------------------

def capture_transformer_inputs(pipe, generation_params: dict,
                               dummy_prompt: str = "A cat holding a sign that says hello world"):
    """Run one inference step with a pre-hook to capture transformer inputs.

    Returns (args_tuple, kwargs_dict) ready to be replayed via
    pipe.transformer(*args, **kwargs).
    """
    captured: dict = {}

    def hook(module, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    handle = pipe.transformer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        gp = dict(generation_params)
        gp["num_inference_steps"] = 1
        with torch.no_grad():
            pipe(dummy_prompt, output_type="latent", **gp)
    finally:
        handle.remove()

    if "args" not in captured:
        raise RuntimeError("Failed to capture transformer inputs")
    return captured["args"], captured["kwargs"]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def run_timed(fn: Callable[[], Any], *, warmup: int = 2, repeats: int = 10,
              ignore_ratio: float = 0.2) -> dict:
    """Warmup + repeats with cuda.synchronize. Trim outliers, return stats."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times_sorted = sorted(times)
    drop = int(ignore_ratio * len(times_sorted) / 2)
    if drop > 0:
        trimmed = times_sorted[drop:-drop]
    else:
        trimmed = times_sorted
    mean = sum(trimmed) / len(trimmed)
    var = sum((t - mean) ** 2 for t in trimmed) / max(len(trimmed) - 1, 1)
    return {
        "mean": mean,
        "std": var ** 0.5,
        "min": min(times),
        "max": max(times),
        "raw": times,
        "trimmed": trimmed,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def fmt_speedup(t_ref: float, t: float) -> str:
    if t <= 0:
        return "n/a"
    return f"{t_ref / t:.2f}x"


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"
