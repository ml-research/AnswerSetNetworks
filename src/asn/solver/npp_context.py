from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NPPContext:
    """TODO"""

    p: Optional[torch.Tensor] = None  # batch_size x n_out

    def to(self, *args, **kwargs) -> "NPPContext":
        """TODO"""
        if self.p is not None:
            self.p.to(*args, **kwargs)

        return self
