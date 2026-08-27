"""Validate the DiRotQ-absorb-basis packed checkpoint against the real
nunchaku fp4 kernel, layer by layer.

For each validation layer saved by build_checkpoint (--out ...valref.pt), load
the packed tensors into SVDQW4A4Linear(precision='nvfp4') and compare the
kernel output on random bf16 input with the reference
    y_ref = fp4sim(x) @ W_q^T + x @ U @ lora_up^T
computed from the *unpacked* float tensors. The residual mismatch comes only
from the kernel's exact activation-quantization rounding vs our simulation, so
relative error should be a few percent; a packing/layout bug shows up as ~100%.

Run in the svdquant env (needs nunchaku):
  python absorb_basis/validate_kernel.py --ckpt <path.safetensors>
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

E2M1_CODEBOOK = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def simulate_act_fp4(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    """Reference NVFP4 dynamic activation quantization (per-group-16 e4m3 scale)."""
    shp = x.shape
    xg = x.float().reshape(-1, shp[-1] // group, group)
    s = (xg.abs().amax(dim=-1, keepdim=True) / 6.0).clamp(min=1e-8)
    s = s.to(torch.float8_e4m3fn).float().clamp(min=2.0 ** -9)
    cb = E2M1_CODEBOOK.to(x.device)
    mag = (xg / s).abs().unsqueeze(-1).sub(cb).abs().argmin(dim=-1)
    xq = cb[mag] * xg.sign() * s
    return xq.reshape(shp).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    args = ap.parse_args()

    from safetensors import safe_open
    from nunchaku.models.linear import SVDQW4A4Linear

    valrefs = torch.load(args.ckpt + ".valref.pt", map_location="cpu", weights_only=False)

    with safe_open(args.ckpt, framework="pt") as f:
        keys = set(f.keys())
        for prefix, ref in valrefs.items():
            names = [k for k in keys if k.startswith(prefix + ".")]
            sd = {k[len(prefix) + 1:]: f.get_tensor(k) for k in names}

            oc = sd["lora_up"].shape[0]
            ic = sd["lora_down"].shape[0]
            rank = sd["lora_down"].shape[1]

            m = SVDQW4A4Linear(
                in_features=ic, out_features=oc, rank=rank, bias=("bias" in sd),
                precision="nvfp4", torch_dtype=torch.bfloat16,
            )
            # checkpoint-key -> module-key mapping (mirrors nunchaku's loader)
            mapped = {}
            rename = {"smooth": "smooth_factor", "smooth_orig": "smooth_factor_orig",
                      "lora_down": "proj_down", "lora_up": "proj_up"}
            for k, v in sd.items():
                if k == "wtscale":
                    continue
                mapped[rename.get(k, k)] = v
            if "wcscales" not in mapped and hasattr(m, "wcscales"):
                mapped["wcscales"] = torch.ones_like(m.wcscales)
            info = m.load_state_dict(mapped, strict=False)
            if m.wtscale is not None and "wtscale" in sd:
                m.wtscale = sd["wtscale"].float().item()
            m = m.to("cuda")

            torch.manual_seed(0)
            x = (torch.randn(1, args.tokens, ic, device="cuda") * 0.5).to(torch.bfloat16)
            with torch.no_grad():
                y = m(x)

            W_q = ref["W_q"].to("cuda", torch.float32)
            U = ref["U"].to("cuda", torch.float32)
            lora_up = ref["lora_up"].to("cuda", torch.float32)
            s = ref.get("s")
            s = torch.ones(ic, device="cuda") if s is None else s.to("cuda", torch.float32)
            xs = (x.float()[0] / s)  # smoothed input (kernel divides by smooth factor)
            y_ref = simulate_act_fp4(xs.to(torch.bfloat16)).float() @ W_q.t() + xs @ U @ lora_up.t()
            if "bias" in sd:
                # official bias tensors are stored in packed (permuted) order;
                # skip bias in the reference and subtract the kernel's bias
                # contribution by running the module a second time on zeros.
                with torch.no_grad():
                    y_bias = m(torch.zeros_like(x))
                y = y - y_bias

            yf = y.float()[0]
            rel = (yf - y_ref).norm() / y_ref.norm()
            cos = torch.nn.functional.cosine_similarity(
                yf.flatten(), y_ref.flatten(), dim=0
            )
            print(f"{prefix}: rel_err={rel.item():.4f} cos={cos.item():.6f} "
                  f"finite={torch.isfinite(y).all().item()} "
                  f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})")


if __name__ == "__main__":
    main()
