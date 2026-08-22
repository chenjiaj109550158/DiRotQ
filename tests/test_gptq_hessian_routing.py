import torch.nn as nn

from utils.gptq_utils import select_hessian_qlayers
from utils.quant_utils import ActQuantWrapper


def test_configured_rtn_layers_do_not_allocate_unused_hessians():
    model = nn.Module()
    model.attn = ActQuantWrapper(nn.Linear(8, 8, bias=False))
    model.ff = nn.Module()
    model.ff.net = nn.ModuleList(
        [nn.Identity(), nn.Identity(), ActQuantWrapper(nn.Linear(16, 8, bias=False))]
    )
    selected = select_hessian_qlayers(model, rtn_names=[".net.2"])
    assert set(selected) == {"attn"}


def test_empty_rtn_routing_preserves_all_wrappers():
    model = nn.Sequential(
        ActQuantWrapper(nn.Linear(4, 4, bias=False)),
        ActQuantWrapper(nn.Linear(4, 4, bias=False)),
    )
    assert len(select_hessian_qlayers(model)) == 2
