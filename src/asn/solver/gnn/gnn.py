import itertools
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .message_passing import (
    AggrMaxUpdater,
    AggrMinUpdater,
    AggrSumUpdater,
    ConjUpdater,
    DisjUpdater,
)

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.nn import MessagePassing


_node_types = ("atom", "disj", "conj", "count", "sum", "min", "max")


class ASNGNN(nn.Module):
    def __init__(
        self,
        updaters: Optional[Dict[str, "MessagePassing"]] = None,
    ) -> None:
        super().__init__()

        if updaters is None:
            updaters = {
                **dict.fromkeys(["atom", "disj"], DisjUpdater()),
                "conj": ConjUpdater(),
                **dict.fromkeys(["count", "sum"], AggrSumUpdater()),
                "min": AggrMinUpdater(),
                "max": AggrMaxUpdater(),
            }

        # message passing node updaters
        self.updaters = updaters

    def forward(
        self,
        node_dict: Dict[str, Dict[str, "Tensor"]],
        edge_dict: Dict[str, Dict[str, "Tensor"]],
        certain_atom_ids: Optional["Tensor"] = None,
    ) -> Tuple["Tensor", bool]:
        # flag signifying whether or not any node values changed
        converged = True

        # for all different edge types
        for edge_type in edge_dict.keys():
            # parse source and destination node types
            src_types = edge_type[0].split("/") if edge_type[0] != "_" else _node_types
            dst_types = edge_type[2].split("/")

            # update nodes
            for dst_type, x_prime in zip(
                dst_types,
                torch.split(
                    # assumes that all target nodes share same update function
                    self.updaters[dst_types[0]](
                        x=(
                            torch.cat(
                                [node_dict[src_type].x for src_type in src_types],
                                dim=0,
                            ),
                            torch.cat(
                                [node_dict[dst_type].x for dst_type in dst_types],
                                dim=0,
                            ),
                        ),
                        edge_index=edge_dict[edge_type].edge_index,
                        edge_weight=edge_dict[edge_type].edge_weight,
                        guards=(
                            torch.cat(
                                [node_dict[dst_type].guards for dst_type in dst_types],
                                dim=0,
                            )
                            if dst_types[0] in _node_types[3:]
                            else None
                        ),
                    ),
                    tuple(node_dict[dst_type].num_nodes for dst_type in dst_types),
                    dim=0,
                ),
            ):
                if converged and not torch.allclose(node_dict[dst_type].x, x_prime):
                    converged = False

                node_dict[dst_type].x = x_prime

        # manually set value for certain atoms to 1 (True)
        if certain_atom_ids is not None:
            node_dict["atom"].x[certain_atom_ids] = 1.0

        return node_dict, converged
