from typing import Optional

from .gnn import ASNGNN
from .graph_block import GraphBlock

_node_types = ("atom", "disj", "conj", "count", "sum", "min", "max")


# TODO: repackage in GNN class ???
class Solver:
    """TODO"""

    def __init__(self):
        """TODO"""
        # initialize GNN
        self.gnn = ASNGNN()

    def solve(self, graph_block: GraphBlock, max_iter: int = -1) -> GraphBlock:
        """TODO"""
        if max_iter == -1:
            max_iter = float("inf")

        converged = False

        while not converged and max_iter > 0:
            # forward step
            _, converged = self.gnn(
                graph_block.node_dict,
                graph_block.edge_dict,
                graph_block.certain_atom_ids,
            )

            # decrease iteration counter
            max_iter -= 1

            # check for convergence
            if converged:
                break

        return graph_block
