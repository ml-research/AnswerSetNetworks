from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_softmax

from .constr import constr_eval


def scatter_boltzmann(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    temp: float = 1.0,
    out: torch.Tensor = None,
    dim_size: int = None,
) -> torch.Tensor:
    softmax = scatter_softmax(temp * src, index, dim=dim)
    return scatter_add(src * softmax, index, dim=dim, out=out, dim_size=dim_size)


class ASNUpdater(MessagePassing, ABC):
    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_weight: Tensor,
        guards: Optional[Tensor] = None,
    ) -> Tensor:
        return self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            num_nodes=x[1].shape[0],
            guards=guards,
        )

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return (edge_weight < 0) + x_j * edge_weight

    @abstractmethod
    def aggregate(self, *args, **kwargs) -> Tensor:
        pass


class ConjUpdater(ASNUpdater):
    def aggregate(self, inputs: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        out = torch.ones(
            num_nodes, inputs.shape[-1], dtype=inputs.dtype, device=inputs.device
        )
        # compute logical AND for boolean values
        return scatter_min(inputs, index, dim=0, out=out)[0]


class SoftConjUpdater(ASNUpdater):
    def aggregate(self, inputs: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        out = torch.ones(
            num_nodes, inputs.shape[-1], dtype=inputs.dtype, device=inputs.device
        )
        # compute soft (i.e., differentiable) logical AND
        return scatter_boltzmann(inputs, index, temp=8.0, dim=0, out=out)


class DisjUpdater(ASNUpdater):
    def aggregate(self, inputs: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        out = torch.zeros(
            num_nodes, inputs.shape[-1], dtype=inputs.dtype, device=inputs.device
        )
        # compute logical OR for boolean values
        return scatter_max(inputs, index, dim=0, out=out)[0]


class SoftDisjUpdater(ASNUpdater):
    def aggregate(self, inputs: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        out = torch.zeros(
            num_nodes, inputs.shape[-1], dtype=inputs.dtype, device=inputs.device
        )
        # compute soft (i.e., differentiable) logical OR
        return scatter_boltzmann(inputs, index, temp=-8.0, dim=0, out=out)


class AggrUpdater(ASNUpdater, ABC):
    @abstractmethod
    def message(self, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def aggregate(self, *args, **kwargs) -> Tensor:
        pass

    def update(self, aggr_out: Tensor, guards: Tensor, x_i: Tensor) -> Tensor:
        # check bounds for aggregated value
        return constr_eval(
            aggr_out,
            torch.index_select(guards, -1, torch.tensor(0, device=aggr_out.device)),
            torch.index_select(guards, -1, torch.tensor(1, device=aggr_out.device)),
            torch.index_select(guards, -1, torch.tensor(2, device=aggr_out.device)),
            torch.index_select(guards, -1, torch.tensor(3, device=aggr_out.device)),
        ).type(x_i.dtype)


class AggrSumUpdater(AggrUpdater):
    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight * x_j

    def aggregate(self, inputs: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        # sum up active incoming edge weights
        return scatter_add(inputs, index, dim=0, dim_size=num_nodes)


class AggrMinUpdater(AggrUpdater):
    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return torch.where(
            torch.isclose(x_j.type(torch.get_default_dtype()), torch.tensor(1.0)),
            edge_weight,
            float("inf"),
        )

    def aggregate(self, inputs: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        # compute minimum accross all values
        return scatter_min(inputs, index, dim=0, dim_size=num_nodes)[0]


class AggrMaxUpdater(AggrUpdater):
    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return torch.where(
            torch.isclose(x_j.type(torch.get_default_dtype()), torch.tensor(1.0)),
            edge_weight,
            -float("inf"),
        )

    def aggregate(self, inputs: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        # compute maximum accross all values
        return scatter_max(inputs, index, dim=0, dim_size=num_nodes)[0]
