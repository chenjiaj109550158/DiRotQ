"""
Model-specific utilities for FLUX.1-dev.

Same architecture as FLUX.1-schnell (19 double + 38 single blocks),
but with guidance_embeds=True and 50-step inference at guidance_scale=3.5.

All rotation/quantization logic is shared with flux-schnell.
"""

import importlib.util
import os

# Load shared logic from flux-schnell
_schnell_path = os.path.join(os.path.dirname(__file__), "..", "flux-schnell", "model_utils.py")
_spec = importlib.util.spec_from_file_location("flux_schnell_model_utils", _schnell_path)
_schnell = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schnell)

preprocess_transformer = _schnell.preprocess_transformer
assign_online_rotations = _schnell.assign_online_rotations
configure_quantizers_by_name = _schnell.configure_quantizers_by_name

# FLUX.1-dev generation parameters
generation_params = dict(
    num_inference_steps=25,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    max_sequence_length=512,
)
