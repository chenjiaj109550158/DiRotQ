"""
Quantization utilities for DiRotQ.
"""

import torch
import torch.nn as nn

from . import hadamard_utils
from .hadamard_utils import fast_hadamard_transform
from .tilemixfp4_utils import fake_quantize_activation


FP4_ACTIVATION_FORMATS = {
    "nvfp4", "nvfp4-hw", "e0m3", "block-mix-oracle", "tile-mix-oracle",
}
EXPERIMENTAL_FP4_ACTIVATION_FORMATS = FP4_ACTIVATION_FORMATS - {"nvfp4"}


# ---------------------------------------------------------------------------
# NF4 (FP4 E2M1) codebook — matches deepcompressor's sfp4_e2m1_all
# ---------------------------------------------------------------------------
NF4_CODEBOOK_VALUES = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
                        0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0]
NF4_MAX = 6.0  # max absolute codebook value

_nf4_codebook_cache = {}  # (device, dtype) -> tensor


def _get_nf4_codebook(device, dtype):
    key = (device, dtype)
    if key not in _nf4_codebook_cache:
        _nf4_codebook_cache[key] = torch.tensor(
            NF4_CODEBOOK_VALUES, dtype=dtype, device=device)
    return _nf4_codebook_cache[key]


def round_to_nf4_codebook(x):
    """Round each element to the nearest NF4 codebook value (binary search)."""
    cb = _get_nf4_codebook(x.device, x.dtype)
    midpoints = (cb[:-1] + cb[1:]) / 2  # 14 midpoints
    indices = torch.bucketize(x.contiguous(), midpoints)
    return cb[indices]


class NF4STEQuantize(torch.autograd.Function):
    """STE quantize for NF4: normalize by scale, round to codebook, rescale."""
    @staticmethod
    def forward(ctx, x, scale):
        scale = scale.to(x.device)
        x_norm = x / scale
        x_q = round_to_nf4_codebook(x_norm)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0
    return minq, maxq



class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(nn.Module):
    """
    Activation quantizer with mixed-precision support.
    Splits activations into 2 zones: quantized (main, e.g., 4-bit) 
    and unquantized passthrough (tail, e.g., bf16/fp16).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))

        self.bits = 16
        self.high_bits_length = 0
        self.groupsize = -1
        self.sym = False
        self.clip_ratio = 1.0
        self.quant_dtype = "int"  # int or one of FP4_ACTIVATION_FORMATS
        self.format_stats = None

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype

        if self.bits == 16:
            return x

        # NF4 with groupsize: fp16/bf16 split first, then group only the 4-bit region
        if self.quant_dtype == "nvfp4" and self.groupsize > 0:
            return self._forward_nvfp4_grouped(x, x_dtype)

        # Hardware-oriented FP4 modes use one global scale over exactly this
        # call's low-precision region, then 1x16 E4M3 block scales.  Candidate
        # calculation and selection are performed by the pure PyTorch fake
        # quantizer, while the high-precision tail is split off and untouched.
        if self.quant_dtype in EXPERIMENTAL_FP4_ACTIVATION_FORMATS:
            return self._forward_experimental_fp4(x, x_dtype)

        # Split: quantized region first, bf16/fp16 passthrough tail.
        # Pad quantized region to next multiple of groupsize if needed, quantize, trim.
        q_len = x.shape[-1] - self.high_bits_length
        x_l = x[..., :q_len] # quantized to low prec, e.g., 4-bit
        x_h = x[..., q_len:] # bf16/fp16 passthrough — never quantized

        if self.groupsize > 0:
            pad = (-q_len) % self.groupsize
            if pad > 0:
                x_l = torch.nn.functional.pad(x_l, (0, pad))
            x_l = x_l.reshape(x_l.shape[0], x_l.shape[1], -1, self.groupsize)

        if self.quant_dtype == "nvfp4":
            x_l = NF4STEQuantize.apply(x_l, self.scale)
        elif self.sym:
            x_l = STEQuantize.apply(x_l, self.scale, self.maxq)
        else:
            x_l = AsymSTEQuantize.apply(x_l, self.scale, self.zero, self.maxq)

        if self.groupsize > 0:
            x_l = x_l.reshape(x_l.shape[0], x_l.shape[1], -1)[..., :q_len]

        if self.high_bits_length == 0:
            return x_l.to(x_dtype)
        return torch.cat([x_l, x_h], dim=-1).to(x_dtype)

    def _forward_nvfp4_grouped(self, x, x_dtype):
        """NF4 quantization with groupsize.

        Grouping applies only to the quantized region. The fp16/bf16 passthrough
        (last high_bits_length channels) needs no grouping — there is no
        quantization there, hence no scale to share.

        The quantized region is padded to the next multiple of groupsize if
        needed, quantized group-wise, then trimmed back to original length.
        """
        init_shape = x.shape
        gs = self.groupsize
        hlen = self.high_bits_length
        q_len = init_shape[-1] - hlen  # channels to NF4-quantize

        x_q = x[..., :q_len]               # 4-bit nvfp4 region
        x_h = x[..., q_len:] if hlen > 0 else None  # bf16/ fp16 passthrough

        # Pad to next multiple of gs, quantize, trim
        pad = (-q_len) % gs
        if pad > 0:
            x_q = torch.nn.functional.pad(x_q, (0, pad))

        x_g = x_q.reshape(x_q.shape[0], x_q.shape[1], -1, gs)
        scale = torch.amax(x_g.abs(), dim=-1, keepdim=True) * self.clip_ratio
        scale = (scale / NF4_MAX).clamp(min=1e-5)
        x_g_q = round_to_nf4_codebook(x_g / scale) * scale

        x_q_out = x_g_q.reshape(x_q.shape[0], x_q.shape[1], -1)[..., :q_len]

        if x_h is not None:
            return torch.cat([x_q_out, x_h], dim=-1).to(x_dtype)
        return x_q_out.to(x_dtype)

    def _forward_experimental_fp4(self, x, x_dtype):
        """Hardware-faithful FP4 on the low region; fp16/bf16 tail passes through."""
        q_len = x.shape[-1] - self.high_bits_length
        x_q = x[..., :q_len]
        x_h = x[..., q_len:] if self.high_bits_length > 0 else None
        x_q_out = fake_quantize_activation(
            x_q,
            self.quant_dtype,
            clip_ratio=self.clip_ratio,
            format_stats=self.format_stats,
        )
        if x_h is not None:
            return torch.cat([x_q_out, x_h], dim=-1).to(x_dtype)
        return x_q_out.to(x_dtype)

    def configure(
        self,
        bits,
        groupsize=-1,
        sym=False,
        clip_ratio=1.0,
        high_bits_length=0,
        quant_dtype="int",
    ):
        self.quant_dtype = quant_dtype
        if quant_dtype in FP4_ACTIVATION_FORMATS:
            sym = True  # all FP4 codebooks are signed/symmetric
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.high_bits_length = high_bits_length

    def find_params_per_token_groupwise(self, x, maxq):
        xmax = torch.amax(x, dim=-1, keepdim=True) * self.clip_ratio
        xmin = torch.amin(x, dim=-1, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = xmax / maxq
            scale[tmp] = 1
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin).clamp(min=1e-5) / maxq
            zero = torch.round(-xmin / scale)
        return scale.to(x.dtype), zero.to(x.dtype)

    def find_params(self, x):
        is_nvfp4 = (self.quant_dtype == "nvfp4")

        # Experimental FP4 modes compute per-block candidate scales inline.
        if self.quant_dtype in EXPERIMENTAL_FP4_ACTIVATION_FORMATS:
            return

        # NF4 with groupsize: scale computation is done inline in _forward_nvfp4_grouped
        if is_nvfp4 and self.groupsize > 0:
            return  # No pre-computed params needed; done in forward

        # Only the quantized region needs scale; bf16/fp16 tail is passthrough.
        q_len = x.shape[-1] - self.high_bits_length
        x_l = x[..., :q_len]

        if self.groupsize > 0:
            pad = (-q_len) % self.groupsize
            if pad > 0:
                x_l = torch.nn.functional.pad(x_l, (0, pad))
            x_l_grouped = x_l.reshape(x_l.shape[0], x_l.shape[1], -1, self.groupsize)
            self.scale, self.zero = self.find_params_per_token_groupwise(x_l_grouped, self.maxq)
            return

        if is_nvfp4:
            self.scale, self.zero = self._find_params_nvfp4(x_l)
        else:
            self.scale, self.zero = self._find_params(x_l, self.maxq)

    def _find_params(self, x, maxq):
        dev = x.device
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev, dtype=x.dtype)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        # Store scale/zero as [..., 1] and let STEQuantize broadcast along the
        # channel dim. 
        bcast_shape = (*init_shape[:-1], 1)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = xmax / maxq
            scale[tmp] = 1
            scale = scale.reshape(bcast_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin).clamp(min=1e-5) / maxq
            zero = torch.round(-xmin / scale)
            scale = scale.reshape(bcast_shape)
            zero = zero.reshape(bcast_shape)

        return scale.to(x.dtype), zero.to(x.dtype)

    def _find_params_nvfp4(self, x):
        """Compute per-token NF4 scale: max(|x|) / 6.0."""
        dev = x.device
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))

        xmax = reshaped_x.abs().max(1)[0] * self.clip_ratio
        tmp = xmax == 0
        scale = xmax / NF4_MAX
        scale[tmp] = 1
        scale = scale.reshape(*init_shape[:-1], 1)
        zero = torch.zeros_like(scale)
        return scale.to(x.dtype), zero.to(x.dtype)


class ActQuantWrapper(nn.Module):
    """
    Wrapper that applies activation quantization before a linear layer.
    Supports optional online rotation for DiRotQ mixed-precision quantization.

    Online rotation (DiRotQ):
      - rotation: [D, D] matrix for hidden-dim input rotation
      - rotation_per_head: [H, d, d] matrix for per-head input rotation
    The rotation is applied online (x_rot = x @ U), quantization happens in the
    rotated basis, then the input is unrotated (x_unrot = x_rot_quant @ U.T) before
    the linear layer. This keeps the model weights UNCHANGED (no divergence) while
    achieving mixed-precision quantization in the PCA basis.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()

        # Online rotation for DiRotQ (set externally after wrapping)
        self.rotation = None           # [D, D] float32 tensor or None
        self.rotation_per_head = None  # [H, d, d] float32 tensor or None
        self.num_heads = None
        self.head_dim = None

        # Hadamard rotation (alternative to dense PCA rotation for specific layers)
        self.use_hadamard = False
        self.hadamard_sign_flips = None  # [D] tensor of ±1
        self.hadamard_low_dim = None     # power-of-2 dim for FWHT

        # PCA-only permutation: O(D) gather instead of O(D²) matmul.
        # perm_idx[i] = original channel that maps to position i after permutation.
        # Channels are sorted ascending by per-channel variance (low-var first),
        # so the high-variance tail naturally lands in the fp16 passthrough region.
        # Weight is pre-permuted offline (W[:, perm_idx]), no unrotation needed.
        self.perm_idx = None             # [D] int64 tensor or None

    def extra_repr(self):
        s = f"Input Quant: {self.quantizer.bits}b"
        if self.quantizer.high_bits_length > 0:
            s += f", fp16_tail={self.quantizer.high_bits_length}d"
        return s

    def forward(self, x):
        x_dtype = x.dtype

        # Online rotation: rotate input to PCA basis, quantize, unrotate.
        # Keep everything in fp32 through the full rotate→quantize→unrotate
        # cycle; the only fp16 cast is the single .to(x_dtype) at the end.
        # This avoids error accumulation from intermediate fp16 casts that
        # compounds across many layers and classifier-free guidance.
        if self.rotation is not None and self.quantizer.bits < 16:
            # Rotate → quantize in rotated basis → unrotate, all in fp32.
            # Only applied when quantization is active (bits < 16); when bits=16
            # the rotation is a mathematical no-op and we skip it entirely to
            # keep x unchanged (no fp16 precision loss at all).
            init_shape = x.shape
            U = self.rotation.to(x.device, dtype=torch.float32)
            x_flat_fp32 = x.float().reshape(-1, init_shape[-1])  # fp32
            x_rot_3d = (x_flat_fp32 @ U).reshape(init_shape)     # fp32
            self.quantizer.find_params(x_rot_3d)
            x_quant = self.quantizer(x_rot_3d)                    # fp32 in/out
            self.quantizer.free()
            # Single fp16 cast after the full rotate→quantize→unrotate cycle
            x = (x_quant.reshape(-1, init_shape[-1]) @ U.T).to(x_dtype).reshape(init_shape)

        elif self.rotation_per_head is not None and self.quantizer.bits < 16:
            # Per-head rotation: input is [B, T, H*d].
            # Rotate each head, then rearrange so the 4-bit region is contiguous
            # [H*d_q] and fp16 region is contiguous [H*hlen] before quantization.
            # This ensures grouping applies only within 4-bit channels and the
            # fp16 passthrough is never accidentally quantized.
            B_T = x.shape[:-1]
            H, d = self.num_heads, self.head_dim
            hlen = self.quantizer.high_bits_length  # fp16 channels per head (e.g. 9)
            d_q  = d - hlen                         # 4-bit channels per head (e.g. 63)
            U_ph = self.rotation_per_head.to(x.device, dtype=torch.float32)  # [H, d, d]
            x_heads = x.float().reshape(*B_T, H, d)                           # fp32
            x_rot_heads = torch.einsum('...hd,hde->...he', x_heads, U_ph)    # [*B_T, H, d]
            if hlen > 0:
                # Rearrange: [*B_T, H, d] -> [*B_T, H*d_q | H*hlen]
                x_rearranged = torch.cat([
                    x_rot_heads[..., :d_q].reshape(*B_T, H * d_q),
                    x_rot_heads[..., d_q:].reshape(*B_T, H * hlen),
                ], dim=-1)
                saved_hlen = self.quantizer.high_bits_length
                saved_gs   = self.quantizer.groupsize
                self.quantizer.high_bits_length = H * hlen
                self.quantizer.groupsize        = d_q if d_q > 0 else -1
                self.quantizer.find_params(x_rearranged)
                x_quant = self.quantizer(x_rearranged)                        # fp32
                self.quantizer.free()
                self.quantizer.high_bits_length = saved_hlen
                self.quantizer.groupsize        = saved_gs
                # Re-interleave: -> [*B_T, H, d]
                x_q_heads = x_quant[..., :H * d_q].reshape(*B_T, H, d_q)
                x_h_heads = x_quant[..., H * d_q:].reshape(*B_T, H, hlen)
                x_quant_heads = torch.cat([x_q_heads, x_h_heads], dim=-1)    # [*B_T, H, d]
            else:
                x_rot_flat = x_rot_heads.reshape(*B_T, H * d)
                self.quantizer.find_params(x_rot_flat)
                x_quant_heads = self.quantizer(x_rot_flat).reshape(*B_T, H, d)  # fp32
                self.quantizer.free()
            x_unrot = torch.einsum('...hd,hde->...he', x_quant_heads,
                                   U_ph.transpose(1, 2))                       # fp32
            x = x_unrot.to(x_dtype).reshape(*B_T, H * d)

        elif self.use_hadamard and self.quantizer.bits < 16:
            # Hadamard rotation: O(D log D) instead of O(D²).
            # Split into [low_dim (power-of-2, quantized) | high_dim (16-bit passthrough)].
            # FWHT applied only on the low_dim part. fp32 throughout.
            init_shape = x.shape
            D = init_shape[-1]
            low_dim = self.hadamard_low_dim
            x_fp32 = x.float()

            if low_dim < D:
                x_low = x_fp32[..., :low_dim]
                x_high = x_fp32[..., low_dim:]  # 16-bit passthrough
            else:
                x_low = x_fp32
                x_high = None

            # Sign flips + FWHT on low-precision part
            if self.hadamard_sign_flips is not None:
                sf = self.hadamard_sign_flips.to(x_low.device, dtype=x_low.dtype)
                x_low = x_low * sf
            x_rot_low = fast_hadamard_transform(x_low)

            # Quantize the rotated low part
            if x_high is not None:
                x_rot = torch.cat([x_rot_low, x_high], dim=-1)
            else:
                x_rot = x_rot_low
            self.quantizer.find_params(x_rot)
            x_quant = self.quantizer(x_rot)
            self.quantizer.free()

            # Inverse FWHT on low part (FWHT is its own inverse when normalized)
            x_q_low = x_quant[..., :low_dim]
            x_unrot_low = fast_hadamard_transform(x_q_low)
            if self.hadamard_sign_flips is not None:
                x_unrot_low = x_unrot_low * sf

            if x_high is not None:
                x = torch.cat([x_unrot_low, x_quant[..., low_dim:]], dim=-1).to(x_dtype)
            else:
                x = x_unrot_low.to(x_dtype)

        elif self.perm_idx is not None and self.quantizer.bits < 16:
            # PCA-only: O(D) gather to put low-var channels first, then quantize.
            # The weight is pre-permuted (W[:, perm_idx]) so no unrotation needed.
            x = x[..., self.perm_idx.to(x.device)]
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        else:
            # No rotation: standard activation quantization
            if self.quantizer.bits < 16:
                self.quantizer.find_params(x)
                x = self.quantizer(x).to(x_dtype)
                self.quantizer.free()

        return self.module(x).to(x_dtype)



def _match_skip(full_name, patterns):
    """Match a full module path against a list of skip patterns.

    Pattern conventions:
      "pat"    - substring match (default, backwards compatible)
      "^pat"   - full_name.startswith("pat")
      "pat$"   - full_name.endswith("pat")
      "^pat$"  - full_name == "pat" (exact)
    """
    for p in patterns:
        anchor_start = p.startswith("^")
        anchor_end = p.endswith("$")
        core = p[1 if anchor_start else 0 : -1 if anchor_end else None]
        if anchor_start and anchor_end:
            if full_name == core:
                return True
        elif anchor_start:
            if full_name.startswith(core):
                return True
        elif anchor_end:
            if full_name.endswith(core):
                return True
        else:
            if core in full_name:
                return True
    return False


def add_actquant(module, name="", skip_names=None):
    """
    Recursively wrap all nn.Linear layers with ActQuantWrapper.

    Args:
        module: The module to process
        name: Current path name
        skip_names: List of patterns; see _match_skip for supported anchors.
    """
    if skip_names is None:
        skip_names = []

    if isinstance(module, ActQuantWrapper):
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        if isinstance(tmp, nn.Linear):
            full_name = f"{name}.{attr}" if name else attr
            if _match_skip(full_name, skip_names):
                continue
            setattr(module, attr, ActQuantWrapper(tmp))
        elif isinstance(tmp, (nn.Sequential, nn.ModuleList)):
            replaced = []
            for i, child in enumerate(tmp.children()):
                child_name = f"{name}.{attr}.{i}" if name else f"{attr}.{i}"
                if isinstance(child, nn.Linear):
                    if _match_skip(child_name, skip_names):
                        replaced.append(child)
                    else:
                        replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            if isinstance(tmp, nn.ModuleList):
                new_container = nn.ModuleList(replaced)
            else:
                new_container = nn.Sequential(*replaced)
            setattr(module, attr, new_container)

    for name1, child in module.named_children():
        child_full = f"{name}.{name1}" if name else name1
        add_actquant(child, child_full, skip_names)


def find_qlayers(module, layers=None, name=""):
    """Find all quantized layers (ActQuantWrapper instances)."""
    if layers is None:
        layers = [nn.Linear, ActQuantWrapper]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(child, layers=layers,
                         name=f"{name}.{name1}" if name else name1)
        )
    return res


# ---------------------------------------------------------------------------
# PCA-only permutation helpers
# ---------------------------------------------------------------------------

def perm_idx_from_eigendecomp(evec: torch.Tensor, evals: torch.Tensor) -> torch.Tensor:
    """Compute channel permutation from flat PCA eigendecomposition.

    Per-channel variance in the original space:
        var[i] = sum_j  evec[i, j]^2 * evals[j]
    Returns perm_idx [D] int64 sorted ascending by variance (low-var first),
    so the last `high_bits_length` positions are the highest-variance channels.
    """
    var = (evec.float() ** 2) @ evals.float()
    return torch.argsort(var)


def perm_idx_from_eigendecomp_per_head(
    evec_ph: torch.Tensor, evals_ph: torch.Tensor
) -> torch.Tensor:
    """Compute global channel permutation from per-head PCA eigendecomposition.

    evec_ph:  [H, d, d] per-head eigenvectors
    evals_ph: [H, d]    per-head eigenvalues
    Returns perm_idx [H*d] int64 sorted ascending by per-head variance (global flat sort).
    """
    # var_ph[h, i] = sum_j evec_ph[h, i, j]^2 * evals_ph[h, j]
    var_ph   = torch.einsum('hid,hd->hi', evec_ph.float() ** 2, evals_ph.float())  # [H, d]
    var_flat = var_ph.reshape(-1)  # [H*d]
    return torch.argsort(var_flat)


# ---------------------------------------------------------------------------
# Mixed-precision weight quantization (rotate → split → quantize-low → stitch)
# ---------------------------------------------------------------------------
#
# Implements
#
#     y = Q(W_rot[:, :low]) @ Q(x_rot[:, :low])
#       + W_rot[:, tail]    @ x_rot[:, tail]         (fp16 both sides)
#
# where W_rot = W @ U (hidden rotation) or W @ BlockDiag(U_ph) (per-head).
# The tail columns are kept in fp16 and the fused weight sits in the rotated
# basis, so the runtime forward (see dirotq_fused_unrotation._fused_forward)
# skips the unrotation matmul entirely. `mod._unrot_fused = True` tells the
# forward which layers are in the rotated basis.


def _rotate_and_split_W(qlayer, W):
    """Rotate W into the PCA basis and split columns into low/tail.

    Returns (W_low, W_tail, stitch_fn, groupsize_override).

    - For qlayer.rotation (hidden): W_rot = W @ U, contiguous tail.
    - For qlayer.rotation_per_head: each head h has its own d×d rotation and
      the last `high_bits_length` channels *within each head* are tail.
      groupsize_override is forced to d_q (= head_dim - hlen) so every head
      becomes exactly one weight RTN group (no scale contamination from
      tail channels, and no mismatch with activation head layout).

    stitch_fn(W_low_q) returns the full [out, in] weight in the original
    contiguous layout — for per-head this interleaves quant and tail
    channels back within each head so the activations (which stay in the
    interleaved layout at runtime) line up column-by-column.
    """
    W_f32 = W.float()
    out_f = W_f32.shape[0]

    if qlayer.rotation is not None:
        U = qlayer.rotation.to(W_f32.device, dtype=torch.float32)  # [D, D]
        W_rot = W_f32 @ U                                           # [out, D]
        hlen = int(qlayer.quantizer.high_bits_length)
        if hlen > 0:
            n_low = W_rot.shape[1] - hlen
            W_low  = W_rot[:, :n_low].contiguous()
            W_tail = W_rot[:, n_low:].contiguous()

            def stitch(W_low_q):
                return torch.cat([W_low_q.to(W_tail.device), W_tail], dim=1)
        else:
            W_low, W_tail = W_rot.contiguous(), None

            def stitch(W_low_q):
                return W_low_q

        return W_low, W_tail, stitch, None

    if getattr(qlayer, 'use_hadamard', False):
        # Hadamard rotation: only the first `low_dim` input channels are
        # FWHT-rotated; the last D-low_dim channels are an fp16 passthrough.
        # Quantizer's high_bits_length must equal (D - low_dim); matched by
        # configure_quantizers_by_name.
        low_dim = int(qlayer.hadamard_low_dim)
        D = W_f32.shape[1]

        # Build the dense inverse Hadamard matrix; FWHT is self-inverse,
        # so inv_H = diag(sign_flips) @ FWHT.
        H_dense = torch.eye(low_dim, dtype=torch.float32, device=W_f32.device)
        H_dense = fast_hadamard_transform(H_dense)            # rows = FWHT basis
        if qlayer.hadamard_sign_flips is not None:
            sf = qlayer.hadamard_sign_flips.to(W_f32.device, dtype=torch.float32)
            H_dense = sf.unsqueeze(0) * H_dense                # diag(sf) @ FWHT

        W_low = (W_f32[:, :low_dim] @ H_dense.T).contiguous()  # [out, low_dim]
        if D > low_dim:
            W_tail = W_f32[:, low_dim:].contiguous()           # [out, D-low_dim]

            def stitch(W_low_q):
                return torch.cat([W_low_q.to(W_tail.device), W_tail], dim=1)
        else:
            W_tail = None

            def stitch(W_low_q):
                return W_low_q

        return W_low, W_tail, stitch, None

    if qlayer.rotation_per_head is not None:
        U_ph = qlayer.rotation_per_head.to(W_f32.device, dtype=torch.float32)  # [H, d, d]
        H = int(qlayer.num_heads)
        d = int(qlayer.head_dim)
        W_3d = W_f32.reshape(out_f, H, d)
        W_rot_3d = torch.einsum('ohd,hde->ohe', W_3d, U_ph)         # [out, H, d]

        hlen = int(qlayer.quantizer.high_bits_length)  # tail within each head
        d_q = d - hlen
        if hlen > 0:
            W_low_3d  = W_rot_3d[:, :, :d_q].contiguous()           # [out, H, d_q]
            W_tail_3d = W_rot_3d[:, :, d_q:].contiguous()           # [out, H, hlen]
            W_low  = W_low_3d.reshape(out_f, H * d_q)

            def stitch(W_low_q):
                W_low_q_3d = W_low_q.to(W_tail_3d.device).reshape(out_f, H, d_q)
                return torch.cat([W_low_q_3d, W_tail_3d], dim=2).reshape(out_f, H * d)

            W_tail_return = W_tail_3d.reshape(out_f, H * hlen)
            return W_low, W_tail_return, stitch, d_q

        W_low = W_rot_3d.reshape(out_f, H * d)

        def stitch(W_low_q):
            return W_low_q.reshape(out_f, H * d)

        return W_low, None, stitch, d

    if getattr(qlayer, 'perm_idx', None) is not None:
        # PCA-only permutation: reorder columns of W by perm_idx (no rotation matmul).
        # Low-variance channels come first; high-variance tail stays in fp16.
        perm = qlayer.perm_idx.to(W_f32.device)  # [D]
        W_perm = W_f32[:, perm]                   # [out, D]
        hlen = int(qlayer.quantizer.high_bits_length)
        if hlen > 0:
            n_low  = W_perm.shape[1] - hlen
            W_low  = W_perm[:, :n_low].contiguous()
            W_tail = W_perm[:, n_low:].contiguous()

            def stitch(W_low_q):
                return torch.cat([W_low_q.to(W_tail.device), W_tail], dim=1)
        else:
            W_low, W_tail = W_perm.contiguous(), None

            def stitch(W_low_q):
                return W_low_q

        return W_low, W_tail, stitch, None

    # No rotation
    def _identity(W_q):
        return W_q
    return W_f32.contiguous(), None, _identity, None


def _quant_group_int(W, bits, groupsize, sym):
    """Per-group symmetric/asymmetric INT RTN. W: [out, in] float32.

    Handles partial last group by zero-padding to the next multiple of
    groupsize, quantizing uniformly, then trimming back.
    """
    out_f, in_f = W.shape
    if groupsize > 0 and groupsize < in_f:
        pad = (-in_f) % groupsize
        if pad > 0:
            W = torch.nn.functional.pad(W, (0, pad))
        Wg = W.reshape(out_f, -1, groupsize)
        if sym:
            maxq = 2 ** (bits - 1) - 1
            scale = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / maxq
            Wq = torch.clamp(torch.round(Wg / scale), -maxq - 1, maxq) * scale
        else:
            maxq = 2 ** bits - 1
            wmin = torch.minimum(Wg.amin(dim=-1, keepdim=True), torch.zeros_like(Wg[:, :, :1]))
            wmax = torch.maximum(Wg.amax(dim=-1, keepdim=True), torch.zeros_like(Wg[:, :, :1]))
            scale = ((wmax - wmin) / maxq).clamp(min=1e-5)
            zero  = torch.round(-wmin / scale)
            Wq = scale * (torch.clamp(torch.round(Wg / scale) + zero, 0, maxq) - zero)
        return Wq.reshape(out_f, -1)[:, :in_f]

    # Per-channel (groupsize <= 0 or >= in_f)
    if sym:
        maxq = 2 ** (bits - 1) - 1
        scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / maxq
        return torch.clamp(torch.round(W / scale), -maxq - 1, maxq) * scale
    maxq = 2 ** bits - 1
    wmin = torch.minimum(W.amin(dim=1, keepdim=True), torch.zeros_like(W[:, :1]))
    wmax = torch.maximum(W.amax(dim=1, keepdim=True), torch.zeros_like(W[:, :1]))
    scale = ((wmax - wmin) / maxq).clamp(min=1e-5)
    zero  = torch.round(-wmin / scale)
    return scale * (torch.clamp(torch.round(W / scale) + zero, 0, maxq) - zero)


def _quant_group_nvfp4(W, groupsize):
    """Per-group NF4 codebook RTN. W: [out, in] float32.

    Handles partial last group by zero-padding, same as _quant_group_int.
    """
    out_f, in_f = W.shape
    if groupsize > 0 and groupsize < in_f:
        pad = (-in_f) % groupsize
        if pad > 0:
            W = torch.nn.functional.pad(W, (0, pad))
        Wg = W.reshape(out_f, -1, groupsize)
        scale = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / NF4_MAX
        Wq = round_to_nf4_codebook(Wg / scale) * scale
        return Wq.reshape(out_f, -1)[:, :in_f]
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / NF4_MAX
    return round_to_nf4_codebook(W / scale) * scale


def rtn_quantize_weights(model, bits=4, groupsize=64, sym=True, skip_names=None):
    """Apply RTN weight quantization to all ActQuantWrapper layers.

    When a layer has a rotation and active activation quantization
    (bits<16), the weight is rotated into the PCA basis, split at the
    high-bits boundary, and only the low region is quantized — the tail
    stays in fp16. The layer is then flagged `_unrot_fused=True` so the
    patched forward knows the weight already lives in the rotated basis.
    """
    if skip_names is None:
        skip_names = []

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name, qlayer in qlayers.items():
        if _match_skip(name, skip_names):
            continue

        W = qlayer.module.weight.data
        orig_dtype = W.dtype
        rotate = (qlayer.quantizer.bits < 16 and
                  (qlayer.rotation is not None or
                   qlayer.rotation_per_head is not None or
                   getattr(qlayer, 'use_hadamard', False) or
                   getattr(qlayer, 'perm_idx', None) is not None))

        if rotate:
            W_low, W_tail, stitch, gs_over = _rotate_and_split_W(qlayer, W)
            gs = gs_over if gs_over is not None else groupsize
            W_low_q = _quant_group_int(W_low, bits, gs, sym)
            qlayer.module.weight.data = stitch(W_low_q).to(orig_dtype)
            qlayer._unrot_fused = True
        else:
            W_q = _quant_group_int(W.float(), bits, groupsize, sym)
            qlayer.module.weight.data = W_q.to(orig_dtype)


def nvfp4_rtn_quantize_weights(model, groupsize=16, skip_names=None):
    """Apply NF4 (FP4 E2M1) RTN weight quantization.

    Same rotate→split→quantize-low→stitch flow as rtn_quantize_weights.
    """
    if skip_names is None:
        skip_names = []

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name, qlayer in qlayers.items():
        if _match_skip(name, skip_names):
            continue

        W = qlayer.module.weight.data
        orig_dtype = W.dtype
        rotate = (qlayer.quantizer.bits < 16 and
                  (qlayer.rotation is not None or
                   qlayer.rotation_per_head is not None or
                   getattr(qlayer, 'use_hadamard', False) or
                   getattr(qlayer, 'perm_idx', None) is not None))

        if rotate:
            W_low, W_tail, stitch, gs_over = _rotate_and_split_W(qlayer, W)
            gs = gs_over if gs_over is not None else groupsize
            W_low_q = _quant_group_nvfp4(W_low, gs)
            qlayer.module.weight.data = stitch(W_low_q).to(orig_dtype)
            qlayer._unrot_fused = True
        else:
            W_q = _quant_group_nvfp4(W.float(), groupsize)
            qlayer.module.weight.data = W_q.to(orig_dtype)
