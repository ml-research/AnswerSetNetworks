import torch

from asn.utils import eval_dict


def constr_eval(
    vals: torch.Tensor,
    lops: torch.Tensor,
    lbounds: torch.Tensor,
    rops: torch.Tensor,
    rbounds: torch.Tensor,
) -> torch.Tensor:
    res = torch.ones_like(vals, dtype=torch.bool)

    for lop in torch.unique(lops):
        if lop == -1:
            # no left guard specified
            continue

        op_mask = (lops == lop).squeeze(-1)
        res[op_mask] = eval_dict[lop.item()](lbounds[op_mask], vals[op_mask])

    for rop in torch.unique(rops):
        if rop == -1:
            # no left guard specified
            continue

        op_mask = (rops == rop).squeeze(-1)
        res[op_mask] &= eval_dict[rop.item()](vals[op_mask], rbounds[op_mask])

    return res
