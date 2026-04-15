"""
FLUX.1-schnell specific PCA basis collection.

Registers hooks on FluxTransformerBlock (double) and FluxSingleTransformerBlock
(single) layers to accumulate activation covariances, then runs per-layer
eigendecomposition.

Basis keys (match flux-schnell/model_utils.py:assign_online_rotations):
  Double blocks (transformer_blocks.{i}):
    layer.{i}.img_attn          [hidden, hidden]            img Q/K/V input
    layer.{i}.txt_attn          [hidden, hidden]            txt Q/K/V input
    layer.{i}.img_attn.value    [num_heads, head, head]     img V output per-head
    layer.{i}.txt_attn.value    [num_heads, head, head]     txt V output per-head
    layer.{i}.img_ffn           [hidden, hidden]            img FFN up input
    layer.{i}.img_ffn.down      [inter, inter]              img FFN down input
    layer.{i}.txt_ffn           [hidden, hidden]            txt FFN up input
    layer.{i}.txt_ffn.down      [inter, inter]              txt FFN down input

  Single blocks (single_transformer_blocks.{i}):
    single.{i}.attn             [hidden, hidden]            Q/K/V input
    single.{i}.mlp              [hidden, hidden]            MLP up input
    single.{i}.attn_out.value   [num_heads, head, head]     attn-out half of
                                                            split proj_out
                                                            (per-head)
    single.{i}.mlp.down         [inter, inter]              mlp half of split
                                                            proj_out
"""

import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from split_proj_out import split_flux_single_proj_out


def collect_basis(transformer, cache_files: list, cfg: dict) -> dict:
    dims              = cfg["dims"]
    hidden            = dims["hidden"]
    head_dim          = dims["head"]
    num_heads         = dims["num_heads"]
    inter             = dims["intermediate"]
    num_double_layers = dims["num_double_layers"]
    num_single_layers = dims["num_single_layers"]

    # Match SVDQuant's handling of the single-block proj_out: split it into
    # two Linear branches so we can collect per-half covariances independently.
    n_split = split_flux_single_proj_out(transformer)
    print(f"Split {n_split} single-block proj_out layers into attn/mlp halves.")

    # Double-block covariance accumulators
    H_img_attn = [torch.zeros(hidden, hidden, dtype=torch.float64)                 for _ in range(num_double_layers)]
    H_txt_attn = [torch.zeros(hidden, hidden, dtype=torch.float64)                 for _ in range(num_double_layers)]
    H_img_val  = [torch.zeros(num_heads, head_dim, head_dim, dtype=torch.float64)  for _ in range(num_double_layers)]
    H_txt_val  = [torch.zeros(num_heads, head_dim, head_dim, dtype=torch.float64)  for _ in range(num_double_layers)]
    H_img_ffn  = [torch.zeros(hidden, hidden, dtype=torch.float64)                 for _ in range(num_double_layers)]
    H_txt_ffn  = [torch.zeros(hidden, hidden, dtype=torch.float64)                 for _ in range(num_double_layers)]
    H_img_down = [torch.zeros(inter, inter, dtype=torch.float64)                   for _ in range(num_double_layers)]
    H_txt_down = [torch.zeros(inter, inter, dtype=torch.float64)                   for _ in range(num_double_layers)]

    cnt_img_attn = [0] * num_double_layers
    cnt_txt_attn = [0] * num_double_layers
    cnt_img_val  = [0] * num_double_layers
    cnt_txt_val  = [0] * num_double_layers
    cnt_img_ffn  = [0] * num_double_layers
    cnt_txt_ffn  = [0] * num_double_layers
    cnt_img_down = [0] * num_double_layers
    cnt_txt_down = [0] * num_double_layers

    # Single-block covariance accumulators
    H_sattn    = [torch.zeros(hidden, hidden, dtype=torch.float64) for _ in range(num_single_layers)]
    H_smlp     = [torch.zeros(hidden, hidden, dtype=torch.float64) for _ in range(num_single_layers)]
    H_sattnout = [torch.zeros(num_heads, head_dim, head_dim, dtype=torch.float64) for _ in range(num_single_layers)]
    H_smlpdown = [torch.zeros(inter, inter, dtype=torch.float64) for _ in range(num_single_layers)]
    cnt_sattn    = [0] * num_single_layers
    cnt_smlp     = [0] * num_single_layers
    cnt_sattnout = [0] * num_single_layers
    cnt_smlpdown = [0] * num_single_layers

    # Accumulators live on CPU. The partial covariance is computed on GPU
    # (fast), then moved to CPU before accumulation. FLUX's 12288-wide
    # intermediate dim makes the 19 img/txt_down accumulators alone ~35GB —
    # keeping them on GPU alongside a 30GB bf16 transformer OOMs instantly.
    def _input_hook(i, H_list, cnt_list):
        def hook(module, args, output):
            x = args[0] if isinstance(args, tuple) else args
            if x.dim() == 2:
                x = x.unsqueeze(0)
            B, T, D = x.shape
            x_flat = x.reshape(-1, D).detach().double()
            partial = (x_flat.T @ x_flat).cpu()
            H_list[i] += partial
            cnt_list[i] += B * T
        return hook

    def _value_hook(i, H_list, cnt_list):
        def hook(module, args, output):
            if output.dim() == 2:
                output = output.unsqueeze(0)
            B, T, _ = output.shape
            v = output.reshape(B * T, num_heads, head_dim).detach().double()
            partial = torch.einsum('nhd,nhe->hde', v, v).cpu()
            H_list[i] += partial
            cnt_list[i] += B * T
        return hook

    def _attn_out_input_hook(i, H_list, cnt_list):
        """Per-head input cov for proj_out.linears[0].

        Input x is the attention output sliced from the concat, shape
        [B, T, hidden] where hidden = num_heads * head_dim. We reinterpret
        as [B*T, num_heads, head_dim] and accumulate a per-head Gram matrix —
        matching how double-block attn.to_out uses _value_hook on attn output.
        """
        def hook(module, args, output):
            x = args[0] if isinstance(args, tuple) else args
            if x.dim() == 2:
                x = x.unsqueeze(0)
            B, T, D = x.shape
            v = x.reshape(B * T, num_heads, head_dim).detach().double()
            partial = torch.einsum('nhd,nhe->hde', v, v).cpu()
            H_list[i] += partial
            cnt_list[i] += B * T
        return hook

    hooks = []

    for i, block in enumerate(transformer.transformer_blocks):
        # Image stream
        hooks.append(block.attn.to_q.register_forward_hook(_input_hook(i, H_img_attn, cnt_img_attn)))
        hooks.append(block.attn.to_v.register_forward_hook(_value_hook(i, H_img_val,  cnt_img_val)))
        hooks.append(block.ff.net[0].proj.register_forward_hook(_input_hook(i, H_img_ffn,  cnt_img_ffn)))
        hooks.append(block.ff.net[2].register_forward_hook(      _input_hook(i, H_img_down, cnt_img_down)))
        # Text stream
        hooks.append(block.attn.add_q_proj.register_forward_hook(_input_hook(i, H_txt_attn, cnt_txt_attn)))
        hooks.append(block.attn.add_v_proj.register_forward_hook(_value_hook(i, H_txt_val,  cnt_txt_val)))
        hooks.append(block.ff_context.net[0].proj.register_forward_hook(_input_hook(i, H_txt_ffn,  cnt_txt_ffn)))
        hooks.append(block.ff_context.net[2].register_forward_hook(      _input_hook(i, H_txt_down, cnt_txt_down)))

    for i, block in enumerate(transformer.single_transformer_blocks):
        hooks.append(block.attn.to_q.register_forward_hook(_input_hook(i, H_sattn, cnt_sattn)))
        hooks.append(block.proj_mlp.register_forward_hook(  _input_hook(i, H_smlp,  cnt_smlp)))
        # Split proj_out halves
        hooks.append(block.proj_out.linears[0].register_forward_hook(
            _attn_out_input_hook(i, H_sattnout, cnt_sattnout)))
        hooks.append(block.proj_out.linears[1].register_forward_hook(
            _input_hook(i, H_smlpdown, cnt_smlpdown)))

    print(f"Registered {len(hooks)} hooks across "
          f"{num_double_layers} double + {num_single_layers} single blocks.")

    with torch.no_grad():
        for f in tqdm(cache_files, desc="Replaying calibration cache"):
            data = torch.load(f, map_location="cuda", weights_only=False)
            input_args = [a.to("cuda").float() for a in data["input_args"]]
            input_kwargs = {}
            for k, v in data["input_kwargs"].items():
                if isinstance(v, torch.Tensor):
                    input_kwargs[k] = v.to("cuda").float() if v.dtype.is_floating_point else v.to("cuda")
                elif isinstance(v, dict):
                    input_kwargs[k] = {kk: (vv.to("cuda") if isinstance(vv, torch.Tensor) else vv)
                                       for kk, vv in v.items() if vv is not None}
                else:
                    input_kwargs[k] = v
            try:
                transformer(*input_args, **input_kwargs)
            except Exception as e:
                print(f"Warning: {f} failed: {e}")

    for h in hooks:
        h.remove()

    torch.cuda.empty_cache()

    print("Computing PCA eigendecompositions (CPU, float64)...")
    basis_dict = {}

    for i in tqdm(range(num_double_layers), desc="double blocks"):
        if cnt_img_attn[i] > 0: H_img_attn[i] /= cnt_img_attn[i]
        if cnt_txt_attn[i] > 0: H_txt_attn[i] /= cnt_txt_attn[i]
        if cnt_img_val[i]  > 0: H_img_val[i]  /= cnt_img_val[i]
        if cnt_txt_val[i]  > 0: H_txt_val[i]  /= cnt_txt_val[i]
        if cnt_img_ffn[i]  > 0: H_img_ffn[i]  /= cnt_img_ffn[i]
        if cnt_txt_ffn[i]  > 0: H_txt_ffn[i]  /= cnt_txt_ffn[i]
        if cnt_img_down[i] > 0: H_img_down[i] /= cnt_img_down[i]
        if cnt_txt_down[i] > 0: H_txt_down[i] /= cnt_txt_down[i]

        basis_dict[f"layer.{i}.img_attn"]       = _eigh(H_img_attn[i])
        basis_dict[f"layer.{i}.txt_attn"]       = _eigh(H_txt_attn[i])
        basis_dict[f"layer.{i}.img_attn.value"] = _eigh_per_head(H_img_val[i], num_heads)
        basis_dict[f"layer.{i}.txt_attn.value"] = _eigh_per_head(H_txt_val[i], num_heads)
        basis_dict[f"layer.{i}.img_ffn"]        = _eigh(H_img_ffn[i])
        basis_dict[f"layer.{i}.txt_ffn"]        = _eigh(H_txt_ffn[i])
        basis_dict[f"layer.{i}.img_ffn.down"]   = _eigh(H_img_down[i])
        basis_dict[f"layer.{i}.txt_ffn.down"]   = _eigh(H_txt_down[i])

    for i in tqdm(range(num_single_layers), desc="single blocks"):
        if cnt_sattn[i]    > 0: H_sattn[i]    /= cnt_sattn[i]
        if cnt_smlp[i]     > 0: H_smlp[i]     /= cnt_smlp[i]
        if cnt_sattnout[i] > 0: H_sattnout[i] /= cnt_sattnout[i]
        if cnt_smlpdown[i] > 0: H_smlpdown[i] /= cnt_smlpdown[i]

        basis_dict[f"single.{i}.attn"]           = _eigh(H_sattn[i])
        basis_dict[f"single.{i}.mlp"]            = _eigh(H_smlp[i])
        basis_dict[f"single.{i}.attn_out.value"] = _eigh_per_head(H_sattnout[i], num_heads)
        basis_dict[f"single.{i}.mlp.down"]       = _eigh(H_smlpdown[i])

    return basis_dict


def _eigh(H: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    """Eigendecomposition of a covariance matrix. Returns eigenvectors (ascending).

    H is accumulated in fp64 for numerical stability of the eigendecomposition,
    but the returned evec is cast to fp32.
    """
    H = H + damping * H.diagonal().mean() * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    _, evec = torch.linalg.eigh(H)
    return evec.float().cpu()


def _eigh_per_head(H: torch.Tensor, num_heads: int, damping: float = 0.01) -> torch.Tensor:
    """Per-head eigendecomposition. H: [num_heads, d, d]. Returns [num_heads, d, d] fp32."""
    evec_all = torch.zeros(H.shape, dtype=torch.float32)
    for h in range(num_heads):
        evec_all[h] = _eigh(H[h], damping)
    return evec_all
