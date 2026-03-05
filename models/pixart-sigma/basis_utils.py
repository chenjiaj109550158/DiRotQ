"""
PixArt-Sigma specific PCA basis collection.
Registers hooks on PixArt transformer blocks to accumulate activation covariances.
"""

import torch
from tqdm import tqdm


def collect_basis(transformer, cache_files: list, cfg: dict) -> dict:
    """
    Replay calibration cache through the PixArt transformer and compute PCA basis.

    Hooks are registered on the following layers per block:
      - attn1.to_q    (self-attn input, shared with K/V)
      - attn1.to_v    (self-attn value output, per-head)
      - attn2.to_q    (cross-attn Q input)
      - attn2.to_v    (cross-attn value output, per-head)
      - ff.net[0].proj (FFN up-proj input)
      - ff.net[2]     (FFN down-proj input)

    Args:
        transformer:  PixArtTransformer2DModel on CUDA, in eval mode.
        cache_files:  List of .pt calibration file paths to replay.
        cfg:          Model config dict (from config.yaml).

    Returns:
        basis_dict: {
            "layer.{i}.self_attn":          [hidden, hidden] float64 eigenvectors,
            "layer.{i}.self_attn.value":    [num_heads, head_dim, head_dim] float64,
            "layer.{i}.cross_attn_q":       [hidden, hidden] float64,
            "layer.{i}.cross_attn_q.value": [num_heads, head_dim, head_dim] float64,
            "layer.{i}.ffn":                [hidden, hidden] float64,
            "layer.{i}.ffn.down_proj":      [intermediate, intermediate] float64,
        }
    """
    dims       = cfg["dims"]
    hidden     = dims["hidden"]
    head_dim   = dims["head"]
    num_heads  = dims["num_heads"]
    inter      = dims["intermediate"]
    num_layers = dims["num_layers"]

    # Covariance accumulators
    H_sa     = [torch.zeros(hidden, hidden, dtype=torch.float64)           for _ in range(num_layers)]
    H_sa_val = [torch.zeros(num_heads, head_dim, head_dim, dtype=torch.float64) for _ in range(num_layers)]
    H_ca_q   = [torch.zeros(hidden, hidden, dtype=torch.float64)           for _ in range(num_layers)]
    H_ca_val = [torch.zeros(num_heads, head_dim, head_dim, dtype=torch.float64) for _ in range(num_layers)]
    H_ffn    = [torch.zeros(hidden, hidden, dtype=torch.float64)           for _ in range(num_layers)]
    H_down   = [torch.zeros(inter, inter, dtype=torch.float64)             for _ in range(num_layers)]

    cnt_sa     = [0] * num_layers
    cnt_sa_val = [0] * num_layers
    cnt_ca_q   = [0] * num_layers
    cnt_ca_val = [0] * num_layers
    cnt_ffn    = [0] * num_layers
    cnt_down   = [0] * num_layers

    def _input_hook(i, H_list, cnt_list):
        def hook(module, args, output):
            x = args[0] if isinstance(args, tuple) else args
            if x.dim() == 2:
                x = x.unsqueeze(0)
            B, T, D = x.shape
            x_flat = x.reshape(-1, D).detach().double()
            if H_list[i].device != x_flat.device:
                H_list[i] = H_list[i].to(x_flat.device)
            H_list[i] += x_flat.T @ x_flat
            cnt_list[i] += B * T
        return hook

    def _value_hook(i, H_list, cnt_list):
        def hook(module, args, output):
            if output.dim() == 2:
                output = output.unsqueeze(0)
            B, T, _ = output.shape
            v = output.reshape(B * T, num_heads, head_dim).detach().double()
            cov = torch.einsum('nhd,nhe->hde', v, v)
            if H_list[i].device != cov.device:
                H_list[i] = H_list[i].to(cov.device)
            H_list[i] += cov
            cnt_list[i] += B * T
        return hook

    hooks = []
    for i, block in enumerate(transformer.transformer_blocks):
        hooks.append(block.attn1.to_q.register_forward_hook(_input_hook(i, H_sa, cnt_sa)))
        hooks.append(block.attn1.to_v.register_forward_hook(_value_hook(i, H_sa_val, cnt_sa_val)))
        hooks.append(block.attn2.to_q.register_forward_hook(_input_hook(i, H_ca_q, cnt_ca_q)))
        hooks.append(block.attn2.to_v.register_forward_hook(_value_hook(i, H_ca_val, cnt_ca_val)))
        hooks.append(block.ff.net[0].proj.register_forward_hook(_input_hook(i, H_ffn, cnt_ffn)))
        hooks.append(block.ff.net[2].register_forward_hook(_input_hook(i, H_down, cnt_down)))

    print(f"Registered {len(hooks)} hooks across {num_layers} blocks.")

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

    print("Computing PCA eigendecompositions...")
    basis_dict = {}

    for i in tqdm(range(num_layers)):
        if cnt_sa[i] > 0:     H_sa[i]     /= cnt_sa[i]
        if cnt_sa_val[i] > 0: H_sa_val[i] /= cnt_sa_val[i]
        if cnt_ca_q[i] > 0:   H_ca_q[i]   /= cnt_ca_q[i]
        if cnt_ca_val[i] > 0: H_ca_val[i] /= cnt_ca_val[i]
        if cnt_ffn[i] > 0:    H_ffn[i]    /= cnt_ffn[i]
        if cnt_down[i] > 0:   H_down[i]   /= cnt_down[i]

        basis_dict[f"layer.{i}.self_attn"]          = _eigh(H_sa[i])
        basis_dict[f"layer.{i}.self_attn.value"]    = _eigh_per_head(H_sa_val[i], num_heads)
        basis_dict[f"layer.{i}.cross_attn_q"]       = _eigh(H_ca_q[i])
        basis_dict[f"layer.{i}.cross_attn_q.value"] = _eigh_per_head(H_ca_val[i], num_heads)
        basis_dict[f"layer.{i}.ffn"]                = _eigh(H_ffn[i])
        basis_dict[f"layer.{i}.ffn.down_proj"]      = _eigh(H_down[i])

    return basis_dict


def _eigh(H: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    """Eigendecomposition of a covariance matrix. Returns eigenvectors (ascending)."""
    H = H + damping * H.diagonal().mean() * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    _, evec = torch.linalg.eigh(H)
    return evec.cpu()


def _eigh_per_head(H: torch.Tensor, num_heads: int, damping: float = 0.01) -> torch.Tensor:
    """Per-head eigendecomposition. H: [num_heads, d, d]. Returns [num_heads, d, d]."""
    head_dim = H.shape[-1]
    evec_all = torch.zeros_like(H)
    for h in range(num_heads):
        evec_all[h] = _eigh(H[h], damping)
    return evec_all.cpu()
