import itertools
from typing import Optional

import torch
from torch_geometric.data import HeteroData

__node_types = ("atom", "disj", "conj", "count", "sum", "min", "max")


def condense_edges_pyg(
    data: HeteroData, device: Optional[torch.device] = None
) -> HeteroData:
    """TODO"""

    # initialize new heterogeneous PyG graph
    data_condensed = HeteroData()
    data_condensed.hard = data.hard
    data_condensed.copies = data.copies

    # number of nodes per type
    num_nodes_dict = {
        node_type: data[node_type].num_nodes for node_type in __node_types
    }
    # accumulate numbers of nodes
    num_nodes_dict["_"] = sum([n for n in num_nodes_dict.values()])
    num_nodes_dict["atom/disj"] = num_nodes_dict["atom"] + num_nodes_dict["disj"]
    num_nodes_dict["count/sum"] = num_nodes_dict["count"] + num_nodes_dict["sum"]
    num_nodes_dict["atom/disj/conj"] = (
        num_nodes_dict["atom/disj"] + num_nodes_dict["conj"]
    )

    if device is None:
        # infer device of 'data'
        # NOTE: assumes all tensors reside on the same device!
        for node_type, num_nodes in num_nodes_dict.items():
            if num_nodes > 0:
                device = data[node_type].x.device
                break
        else:
            device = None

    # infer number of copies in graph
    copies = data["disj"].x.shape[-1]

    # offsets of node ids for each node type
    offset_dict = {
        node_type: cum_nodes
        for node_type, cum_nodes in zip(
            __node_types,
            [0]
            + list(
                itertools.accumulate(
                    [num_nodes_dict[node_type] for node_type in __node_types]
                )
            ),
        )
    }

    # ----- edge indices -----

    # -> atom/disj.
    data_condensed[("_", "to", "atom/disj")].edge_index = torch.cat(
        [
            torch.cat(
                [
                    torch.tensor([[offset_dict[src_type]], [0]], device=device)
                    + data[(src_type, "to", dst_type)].edge_index
                    if (src_type, "to", dst_type) in data.edge_types
                    else torch.empty(2, 0, device=device)
                    for src_type in __node_types
                ],
                dim=1,
            )
            + torch.tensor([[0], [offset_dict[dst_type]]], device=device)
            for dst_type in ("atom", "disj")
        ],
        dim=1,
    )
    # -> conj.
    data_condensed[("_", "to", "conj")].edge_index = torch.cat(
        [
            torch.tensor([[offset_dict[src_type]], [0]], device=device)
            + data[(src_type, "to", "conj")].edge_index
            if (src_type, "to", "conj") in data.edge_types
            else torch.empty(2, 0, device=device)
            for src_type in __node_types
        ],
        dim=1,
    )
    # -> count/sum
    data_condensed[("atom/disj/conj", "to", "count/sum")].edge_index = torch.cat(
        [
            torch.cat(
                [
                    torch.tensor([[offset_dict[src_type]], [0]], device=device)
                    + data[(src_type, "to", dst_type)].edge_index
                    if (src_type, "to", dst_type) in data.edge_types
                    else torch.empty(2, 0, device=device)
                    for src_type in ("atom", "disj", "conj")
                ],
                dim=1,
            )
            + torch.tensor(
                [[0], [offset_dict[dst_type] - offset_dict["count"]]],
                device=device,
            )
            for dst_type in ("count", "sum")
        ],
        dim=1,
    )
    # -> min
    data_condensed[("atom/disj/conj", "to", "min")].edge_index = torch.cat(
        [
            torch.tensor([[offset_dict[src_type]], [0]], device=device)
            + data[(src_type, "to", "min")].edge_index
            if (src_type, "to", "min") in data.edge_types
            else torch.empty(2, 0, device=device)
            for src_type in ("atom", "disj", "conj")
        ],
        dim=1,
    )
    # -> max
    data_condensed[("atom/disj/conj", "to", "max")].edge_index = torch.cat(
        [
            torch.tensor([[offset_dict[src_type]], [0]], device=device)
            + data[(src_type, "to", "max")].edge_index
            if (src_type, "to", "max") in data.edge_types
            else torch.empty(2, 0, device=device)
            for src_type in ("atom", "disj", "conj")
        ],
        dim=1,
    )

    # ----- node features -----

    for node_type, num_nodes in num_nodes_dict.items():
        data_condensed[node_type].num_nodes = num_nodes_dict[node_type]

    for node_type in __node_types:
        data_condensed[node_type].x = data[node_type].x

        if node_type in __node_types[3:]:
            data_condensed[node_type].guards = data[node_type].guards

    # ----- edge features -----

    # -> atom/disj.
    if ("_", "to", "atom/disj") in data_condensed.edge_types:
        data_condensed[("_", "to", "atom/disj")].edge_weight = torch.cat(
            [
                data[(src_type, "to", dst_type)].edge_weight
                if (src_type, "to", dst_type) in data.edge_types
                else torch.empty(
                    0,
                    data.copies,
                    dtype=torch.int8 if data.hard else torch.get_default_dtype(),
                    device=device,
                )
                for dst_type in ("atom", "disj")
                for src_type in __node_types
            ],
            dim=0,
        )
    # -> conj.
    if ("_", "to", "conj") in data_condensed.edge_types:
        data_condensed[("_", "to", "conj")].edge_weight = torch.cat(
            [
                data[(src_type, "to", "conj")].edge_weight
                if (src_type, "to", "conj") in data.edge_types
                else torch.empty(
                    0,
                    data.copies,
                    dtype=torch.int8 if data.hard else torch.get_default_dtype(),
                    device=device,
                )
                for src_type in __node_types
            ],
            dim=0,
        )
    # -> count/sum
    if ("atom/disj/conj", "to", "count/sum") in data_condensed.edge_types:
        data_condensed[("atom/disj/conj", "to", "count/sum")].edge_weight = torch.cat(
            [
                data[(src_type, "to", dst_type)].edge_weight
                if (src_type, "to", dst_type) in data.edge_types
                else torch.empty(0, data.copies, device=device)
                for dst_type in ("count", "sum")
                for src_type in __node_types[:3]
            ],
            dim=0,
        )
    # -> min
    if ("atom/disj/conj", "to", "min") in data_condensed.edge_types:
        data_condensed[("atom/disj/conj", "to", "min")].edge_weight = torch.cat(
            [
                data[(src_type, "to", "min")].edge_weight
                if (src_type, "to", "min") in data.edge_types
                else torch.empty(0, data.copies, device=device)
                for src_type in __node_types[:3]
            ],
            dim=0,
        )
    # -> max
    if ("atom/disj/conj", "to", "max") in data_condensed.edge_types:
        data_condensed[("atom/disj/conj", "to", "max")].edge_weight = torch.cat(
            [
                data[(src_type, "to", "max")].edge_weight
                if (src_type, "to", "max") in data.edge_types
                else torch.empty(0, data.copies, device=device)
                for src_type in __node_types[:3]
            ],
            dim=0,
        )

    return data_condensed
