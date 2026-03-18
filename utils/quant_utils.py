"""
Quantization utilities for DiRotQ.
"""

import math
import torch
import torch.nn as nn

from . import hadamard_utils


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


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq, stoch=False):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq, stoch=False):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class ActQuantizer(nn.Module):
    """
    Per-token activation quantizer with mixed-precision support.
    Splits activations into 3 zones: low-bit, mid-bit (main), high-bit.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))

        self.register_buffer("maxq_h", torch.tensor(0))
        self.register_buffer("scale_h", torch.zeros(1))
        self.register_buffer("zero_h", torch.zeros(1))

        self.register_buffer("maxq_l", torch.tensor(0))
        self.register_buffer("scale_l", torch.zeros(1))
        self.register_buffer("zero_l", torch.zeros(1))

        self.bits = 16
        self.high_bits = 16
        self.low_bits = 16
        self.high_bits_length = 0
        self.low_bits_length = 0
        self.quant_dtype = "int"  # "int" or "nvfp4"

    def free(self):
        self.zero = None
        self.scale = None
        self.zero_h = None
        self.scale_h = None
        self.zero_l = None
        self.scale_l = None

    def forward(self, x):
        x_dtype = x.dtype

        if self.bits == 16:
            return x

        # NF4 with groupsize: mixed-precision split is across GROUPS, not within
        if self.quant_dtype == "nvfp4" and self.groupsize > 0:
            return self._forward_nvfp4_grouped(x, x_dtype)

        if self.groupsize > 0:
            init_shape = x.shape
            x = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize
            )

        low_dim = self.low_bits_length
        high_dim = x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        if self.quant_dtype == "nvfp4":
            # NF4 without groupsize (per-token)
            x = NF4STEQuantize.apply(x_m, self.scale)
            if self.high_bits_length != 0:
                if self.high_bits < 16:
                    x_h = STEQuantize.apply(x_h, self.scale_h, self.maxq_h)
                x = torch.cat([x, x_h], dim=-1).to(x_dtype)
            if self.low_bits_length != 0:
                if self.low_bits < 16:
                    x_l = STEQuantize.apply(x_l, self.scale_l, self.maxq_l)
                x = torch.cat([x_l, x], dim=-1).to(x_dtype)
        elif self.sym:
            x = STEQuantize.apply(x_m, self.scale, self.maxq)
            if self.high_bits_length != 0:
                if self.high_bits < 16:
                    x_h = STEQuantize.apply(x_h, self.scale_h, self.maxq_h)
                x = torch.cat([x, x_h], dim=-1).to(x_dtype)
            if self.low_bits_length != 0:
                if self.low_bits < 16:
                    x_l = STEQuantize.apply(x_l, self.scale_l, self.maxq_l)
                x = torch.cat([x_l, x], dim=-1).to(x_dtype)
        else:
            x = AsymSTEQuantize.apply(x_m, self.scale, self.zero, self.maxq)
            if self.high_bits_length != 0:
                if self.high_bits < 16:
                    x_h = AsymSTEQuantize.apply(x_h, self.scale_h, self.zero_h, self.maxq_h)
                x = torch.cat([x, x_h], dim=-1).to(x_dtype)
            if self.low_bits_length != 0:
                if self.low_bits < 16:
                    x_l = AsymSTEQuantize.apply(x_l, self.scale_l, self.zero_l, self.maxq_l)
                x = torch.cat([x_l, x], dim=-1).to(x_dtype)

        if self.groupsize > 0:
            x = x.reshape(init_shape)

        return x

    def _forward_nvfp4_grouped(self, x, x_dtype):
        """NF4 quantization with groupsize: mixed-precision split across groups."""
        init_shape = x.shape
        gs = self.groupsize
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // gs, gs)
        n_groups = x.shape[-2]
        n_high = self.high_bits_length // gs if self.high_bits_length > 0 else 0

        if n_high > 0 and n_high < n_groups:
            # Split groups: first (n_groups - n_high) are NF4, last n_high are 16-bit
            x_m = x[..., :n_groups - n_high, :]  # NF4 groups
            x_h = x[..., n_groups - n_high:, :]  # high-precision groups (16-bit)

            # NF4 quantize the main groups
            scale = torch.amax(x_m.abs(), dim=-1, keepdim=True) * self.clip_ratio
            scale = (scale / NF4_MAX).clamp(min=1e-8)
            x_m_q = round_to_nf4_codebook(x_m / scale) * scale

            x = torch.cat([x_m_q, x_h], dim=-2).to(x_dtype)
        else:
            # All groups quantized with NF4
            scale = torch.amax(x.abs(), dim=-1, keepdim=True) * self.clip_ratio
            scale = (scale / NF4_MAX).clamp(min=1e-8)
            x = (round_to_nf4_codebook(x / scale) * scale).to(x_dtype)

        return x.reshape(init_shape)

    def configure(
        self,
        bits,
        groupsize=-1,
        sym=False,
        clip_ratio=1.0,
        high_bits_length=0,
        high_bits=16,
        low_bits_length=0,
        low_bits=16,
        quant_dtype="int",
    ):
        self.quant_dtype = quant_dtype
        if quant_dtype == "nvfp4":
            sym = True  # NF4 is always symmetric
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio

        self.high_bits_length = high_bits_length
        self.high_bits = high_bits
        _, self.maxq_h = get_minq_maxq(high_bits, sym)

        self.low_bits_length = low_bits_length
        self.low_bits = low_bits
        _, self.maxq_l = get_minq_maxq(low_bits, sym)

    def find_params_per_token_groupwise(self, x, maxq, use_nvfp4=False):
        if use_nvfp4:
            # NF4: scale = max(|x|) / 6.0 (symmetric, codebook range [-6, 6])
            xmax = torch.amax(x.abs(), dim=3, keepdim=True) * self.clip_ratio
            tmp = xmax == 0
            scale = xmax / NF4_MAX
            scale[tmp] = 1
            return scale, torch.zeros_like(scale)

        xmax = torch.amax(x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(x, dim=3, keepdim=True) * self.clip_ratio
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
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
        return scale, zero

    def find_params(self, x):
        is_nvfp4 = (self.quant_dtype == "nvfp4")

        # NF4 with groupsize: scale computation is done inline in _forward_nvfp4_grouped
        if is_nvfp4 and self.groupsize > 0:
            return  # No pre-computed params needed; done in forward

        if self.groupsize > 0:
            init_shape = x.shape
            x_reshaped = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize
            )
            low_dim = self.low_bits_length
            high_dim = x_reshaped.shape[-1] - self.high_bits_length
            x_l = x_reshaped[..., :low_dim]
            x_m = x_reshaped[..., low_dim:high_dim]
            x_h = x_reshaped[..., high_dim:]

            self.scale, self.zero = self.find_params_per_token_groupwise(x_m, self.maxq)
            if self.high_bits_length != 0 and self.high_bits < 16:
                self.scale_h, self.zero_h = self.find_params_per_token_groupwise(x_h, self.maxq_h)
            if self.low_bits_length != 0 and self.low_bits < 16:
                self.scale_l, self.zero_l = self.find_params_per_token_groupwise(x_l, self.maxq_l)
            return

        low_dim = self.low_bits_length
        high_dim = x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        if is_nvfp4:
            self.scale, self.zero = self._find_params_nvfp4(x_m)
        else:
            self.scale, self.zero = self._find_params(x_m, self.maxq)
        if self.high_bits_length != 0 and self.high_bits < 16:
            self.scale_h, self.zero_h = self._find_params(x_h, self.maxq_h)
        if self.low_bits_length != 0 and self.low_bits < 16:
            self.scale_l, self.zero_l = self._find_params(x_l, self.maxq_l)

    def _find_params(self, x, maxq):
        if self.bits == 16:
            return torch.zeros(1), torch.zeros(1)

        dev = x.device
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            scale[tmp] = 1
            scale = scale.reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            scale = scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero

    def _find_params_nvfp4(self, x):
        """Compute per-token NF4 scale: max(|x|) / 6.0."""
        if self.bits == 16:
            return torch.zeros(1), torch.zeros(1)

        dev = x.device
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))

        xmax = reshaped_x.abs().max(1)[0] * self.clip_ratio
        tmp = xmax == 0
        scale = (xmax / NF4_MAX).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
        scale[tmp] = 1
        scale = scale.reshape(init_shape)
        zero = torch.zeros_like(scale)
        return scale, zero


class ActQuantWrapper(nn.Module):
    """
    Wrapper that applies activation quantization before a linear layer.
    Supports optional online rotation for DiRotQ mixed-precision quantization.

    Online rotation (DiRotQ):
      - rotation: [D, D] matrix for hidden-dim input rotation (Q, K, V, FFN up)
      - rotation_per_head: [H, d, d] matrix for per-head input rotation (to_out[0])
      - rotation_down: [D_down, D_down] matrix for down-proj input rotation (ff.net[2])
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
        self.out_quantizer = ActQuantizer()

        # Online rotation for DiRotQ (set externally after wrapping)
        self.rotation = None           # [D, D] float32 tensor or None
        self.rotation_per_head = None  # [H, d, d] float32 tensor or None
        self.num_heads = None
        self.head_dim = None

    def extra_repr(self):
        s = f"Input Quant: {self.quantizer.bits}b"
        if self.quantizer.high_bits_length > 0:
            s += f", high={self.quantizer.high_bits}b/{self.quantizer.high_bits_length}d"
        return s

    def forward(self, x):
        x_dtype = x.dtype

        # Online rotation: rotate input to PCA basis, quantize, unrotate.
        # Key: keep everything in fp32 through the full rotate→quantize→unrotate
        # cycle.  The only fp16 cast is the single .to(x_dtype) at the end,
        # which avoids the error accumulation from two intermediate fp16 casts
        # that would otherwise be amplified by classifier-free guidance (×4.5).
        if self.rotation is not None and self.quantizer.bits < 16:
            # Rotate → quantize in rotated (PCA) basis → unrotate.
            # Done entirely in fp32 to avoid fp16 rounding accumulation across
            # 140 layers that gets amplified by classifier-free guidance (×4.5).
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
            # Per-head rotation for to_out[0]: input is [B, T, H*d].
            # Same strategy: all in fp32, single cast at the end.
            B_T = x.shape[:-1]
            H, d = self.num_heads, self.head_dim
            U_ph = self.rotation_per_head.to(x.device, dtype=torch.float32)  # [H, d, d]
            x_heads = x.float().reshape(*B_T, H, d)                          # fp32
            x_rot_flat = torch.einsum('...hd,hde->...he', x_heads, U_ph).reshape(*B_T, H * d)
            self.quantizer.find_params(x_rot_flat)
            x_quant_flat = self.quantizer(x_rot_flat)                         # fp32
            self.quantizer.free()
            x_quant_heads = x_quant_flat.reshape(*B_T, H, d)
            x_unrot = torch.einsum('...hd,hde->...he', x_quant_heads,
                                   U_ph.transpose(1, 2))                      # fp32
            x = x_unrot.to(x_dtype).reshape(*B_T, H * d)

        else:
            # No rotation: standard activation quantization
            if self.quantizer.bits < 16:
                self.quantizer.find_params(x)
                x = self.quantizer(x).to(x_dtype)
                self.quantizer.free()

        # Linear forward
        x = self.module(x).to(x_dtype)

        # Output quantization (unused in current DiRotQ setup)
        if self.out_quantizer.bits < 16:
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(nn.Module):
    """Per-channel or per-tensor weight quantizer using RTN."""

    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False,
                  norm=2.4, grid=100, maxshrink=0.8):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                if self.sym:
                    scale1 = xmax1 / self.maxq
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq, False).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq, False).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(module, name="", skip_names=None):
    """
    Recursively wrap all nn.Linear layers with ActQuantWrapper.

    Args:
        module: The module to process
        name: Current path name
        skip_names: List of substrings; if any matches, skip wrapping
    """
    if skip_names is None:
        skip_names = []

    if isinstance(module, ActQuantWrapper):
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        if isinstance(tmp, nn.Linear):
            full_name = f"{name}.{attr}" if name else attr
            if any(skip in full_name for skip in skip_names):
                continue
            setattr(module, attr, ActQuantWrapper(tmp))
        elif isinstance(tmp, nn.Sequential):
            replaced = []
            for i, child in enumerate(tmp.children()):
                child_name = f"{name}.{attr}.{i}" if name else f"{attr}.{i}"
                if isinstance(child, nn.Linear):
                    if any(skip in child_name for skip in skip_names):
                        replaced.append(child)
                    else:
                        replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.Sequential(*replaced))

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


def rtn_quantize_weights(model, bits=4, groupsize=64, sym=True, skip_names=None):
    """
    Apply round-to-nearest (RTN) weight quantization to all ActQuantWrapper layers.
    """
    if skip_names is None:
        skip_names = []

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name, qlayer in qlayers.items():
        if any(skip in name for skip in skip_names):
            continue

        W = qlayer.module.weight.data.clone()
        if groupsize > 0 and W.shape[1] % groupsize == 0:
            # Per-group quantization
            W_groups = W.reshape(W.shape[0], -1, groupsize)
            if sym:
                maxq = 2 ** (bits - 1) - 1
                scale = W_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / maxq
                W_q = torch.clamp(torch.round(W_groups / scale), -maxq - 1, maxq)
                W_dq = (W_q * scale).reshape(W.shape)
            else:
                maxq = 2**bits - 1
                wmin = W_groups.amin(dim=-1, keepdim=True)
                wmax = W_groups.amax(dim=-1, keepdim=True)
                wmin = torch.minimum(wmin, torch.zeros_like(wmin))
                wmax = torch.maximum(wmax, torch.zeros_like(wmax))
                scale = ((wmax - wmin) / maxq).clamp(min=1e-5)
                zero = torch.round(-wmin / scale)
                W_q = torch.clamp(torch.round(W_groups / scale) + zero, 0, maxq)
                W_dq = (scale * (W_q - zero)).reshape(W.shape)
            qlayer.module.weight.data = W_dq
        else:
            # Per-channel quantization fallback
            quantizer = WeightQuantizer()
            quantizer.configure(bits, perchannel=True, sym=sym)
            quantizer.find_params(W)
            qlayer.module.weight.data = quantizer.quantize(W)


def nvfp4_rtn_quantize_weights(model, groupsize=16, skip_names=None):
    """
    Apply NF4 (FP4 E2M1) RTN weight quantization to all ActQuantWrapper layers.

    For each group of `groupsize` weights:
      scale = max(|W_group|) / 6.0
      W_norm = W / scale
      W_q = round_to_nf4_codebook(W_norm)
      W_dq = W_q * scale
    """
    if skip_names is None:
        skip_names = []

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name, qlayer in qlayers.items():
        if any(skip in name for skip in skip_names):
            continue

        W = qlayer.module.weight.data.clone().float()
        if groupsize > 0 and W.shape[1] % groupsize == 0:
            W_groups = W.reshape(W.shape[0], -1, groupsize)
            scale = W_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / NF4_MAX
            W_norm = W_groups / scale
            W_q = round_to_nf4_codebook(W_norm)
            W_dq = (W_q * scale).reshape(W.shape)
            qlayer.module.weight.data = W_dq.to(qlayer.module.weight.dtype)
        else:
            # Per-channel fallback
            scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / NF4_MAX
            W_norm = W / scale
            W_q = round_to_nf4_codebook(W_norm)
            W_dq = W_q * scale
            qlayer.module.weight.data = W_dq.to(qlayer.module.weight.dtype)
