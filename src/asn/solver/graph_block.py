from functools import cached_property
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from torch_geometric.data.storage import EdgeStorage, NodeStorage


class GraphBlock:
    """TODO"""

    def __init__(
        self,
        node_dict: "NodeStorage",
        edge_dict: "EdgeStorage",
        npp_choices: torch.Tensor,
        certain_atom_ids: Optional[torch.Tensor] = None,
        sink_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """TODO"""
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.npp_choices = npp_choices
        self.sink_ids = (
            sink_ids if sink_ids is not None else torch.tensor([0], device=self.device)
        )
        # get unique sink ids and the inverse indicices
        self.unique_sink_ids, self.inverse_unique_sink_ids = self.sink_ids.unique(
            sorted=False, return_inverse=True
        )
        self.certain_atom_ids = certain_atom_ids
        self.device = self.node_dict["conj"]["x"].device

    @cached_property
    def block_size(self) -> int:
        # infer number of combinations in graph block
        return self.node_dict["conj"]["x"].shape[1]

    @property
    def atoms(self) -> torch.Tensor:
        """TODO"""
        return self.node_dict["atom"]["x"].T

    @property
    def is_model(self) -> torch.Tensor:
        return ~self.node_dict["disj"].x[self.unique_sink_ids].unsqueeze(-1).bool()
