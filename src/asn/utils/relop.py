import torch
from ground_slash.program.operators import RelOp

relop_dict = {
    op: i
    for i, op in enumerate(
        (
            RelOp.EQUAL,
            RelOp.UNEQUAL,
            RelOp.LESS,
            RelOp.GREATER,
            RelOp.LESS_OR_EQ,
            RelOp.GREATER_OR_EQ,
        )
    )
}


eval_dict = {
    0: torch.eq,
    1: torch.ne,
    2: torch.lt,
    3: torch.gt,
    4: torch.le,
    5: torch.ge,
}
