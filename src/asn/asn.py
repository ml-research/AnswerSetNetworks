# TODO: asn object that handles things like input/output, training etc.?
import math
from copy import deepcopy
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
from ground_slash.grounding import Grounder
from ground_slash.program import Constraint, NPPRule, PredLiteral, Program

from asn.data.reasoning_graph import ReasoningGraph
from asn.data.utils import condense_edges_pyg

from .solver import GraphBlock, NPPContext, Solver, SolvingContext

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


class ASN:
    def __init__(
        self,
        prog: Program,
        always_ground: bool = True,
        grounder: Optional[Grounder] = None,
        num_phases: int = 1,
        rank: int = 0,
        world_size: int = 1,
        group: Optional["ProcessGroup"] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """TODO"""
        # initialize program (optionally ground)
        self.grounder = Grounder(prog) if grounder is None else grounder
        self.prog = self.grounder.ground() if not prog.ground or always_ground else prog
        self.certain_literals = self.grounder.certain_literals

        # initialize reasoning graph
        self.rg = ReasoningGraph(self.prog, self.certain_literals)

        # initialize solver
        self.solver = Solver()
        self.npp_cfg_dict = dict()

        self.num_phases = num_phases

        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError(f"Invalid rank '{rank}' or world size '{world_size}'.")
        self.rank = rank
        self.world_size = world_size

        # TODO: verify group
        if world_size > 1 and group is None:
            # create group with all processes
            group = dist.new_group(list(range(self.world_size)))
        self.group = group

        self.device = device

    @classmethod
    def from_string(
        cls,
        prog_str: str,
        ground: bool = True,
        num_phases: int = 1,
        rank: int = 0,
        world_size: int = 1,
        group: Optional["ProcessGroup"] = None,
        device: Optional[torch.device] = None,
    ) -> "ASN":
        """TODO"""
        # initialize program from string
        return ASN(
            Program.from_string(prog_str),
            ground,
            num_phases=num_phases,
            rank=rank,
            world_size=world_size,
            group=group,
            device=device,
        )

    def configure_NPPs(self, npp_cfg_dict: Dict[NPPRule, NPPContext]) -> None:
        """TODO"""
        self.npp_cfg_dict.update(npp_cfg_dict)

    def npp_forward(
        self, npp_data: Dict[str, Tuple[Any]]
    ) -> Dict["NPPRule", NPPContext]:
        """TODO"""
        return {
            rule: NPPContext(npp_cfg["model"](*npp_data[rule]))
            for rule, npp_cfg in self.npp_cfg_dict.items()
        }

    def encode_queries(self, queries: Optional[List[Constraint]]) -> ReasoningGraph:
        if queries is None:
            queries = []

        # copy original reasoning graph to encode queries
        rg = deepcopy(self.rg) if queries else self.rg

        # encode all queries
        for query in queries:
            rg.encode_query(query)

        return rg

    def prepare_block(
        self,
        queries: Optional[List[Constraint]] = None,
        rg: Optional[ReasoningGraph] = None,
        device: Optional[torch.device] = None,
        phase: int = 0,
    ) -> GraphBlock:
        """TODO"""
        if queries is None:
            queries = []

        if rg is None:
            # copy original reasoning graph to encode queries
            rg = deepcopy(self.rg) if queries else self.rg

            # encode all queries
            for query in queries:
                rg.encode_query(query)

        if device is None:
            device = self.device

        # get node ids for all query sinks (indicating SAT)
        # NOTE: uses global sink in case no queries are specified
        query_sinks = (
            torch.tensor(rg.query_sinks, device=device)
            if queries
            else torch.tensor([0], device=device)
        )

        # get powerset of choices for all non-deterministic rules
        powerset_dict = {
            rule: rule.powerset()
            for rule in chain(
                rg.npp_edges.keys(),
                rg.choice_edges.keys(),
            )
        }

        # total number of non-deterministic choice/outcome combinations
        total_combinations = math.prod(
            [len(powerset) for powerset in powerset_dict.values()]
        )

        # divide total number of combinations by world size
        n_per_block, remainder = divmod(
            total_combinations, self.world_size * self.num_phases
        )

        # block id
        block_id = self.rank * self.num_phases + phase

        # start index for combinations
        start_index = n_per_block * block_id

        # spread out remainder across chunks
        if block_id < remainder:
            start_index += self.rank
        else:
            start_index += remainder

        # end index for combinations (excluding)
        end_index = start_index + n_per_block

        # spread out remainder across chunks
        if block_id < remainder:
            end_index += 1

        combination_bounds = (start_index, end_index)

        # number of combinations in chunk
        n_combinations = combination_bounds[1] - combination_bounds[0]

        if total_combinations == 0:
            # deterministic program
            total_combinations = 1

        # create pyg "batch" from abstract reasoning graph
        batch = rg.to_pyg(device=device, hard=True, copies=n_combinations)

        # set all choice edges to zero initially
        # TODO: do not do this (no way to recover original sign for activation)
        for edges in chain(rg.npp_edges.values(), rg.choice_edges.values()):
            for _, edge_type, edge_id in edges:
                # NOTE: zeros entries for all copies of the same edge!
                # TODO: initialize directly to zero in RG to avoid doing it here
                batch[edge_type].edge_weight[edge_id] = 0

        # initialize NPP contexts
        npp_choices_dict = {rule: [] for rule in rg.npp_edges}

        # set choices
        for i, powerset_choices in enumerate(
            torch.cartesian_prod(
                *[torch.arange(len(powerset)) for powerset in powerset_dict.values()]
            )[combination_bounds[0] : combination_bounds[1]]
        ):
            # set choices
            for (rule, edges), powerset_choice in zip(
                chain(rg.npp_edges.items(), rg.choice_edges.items()),
                powerset_choices,
            ):
                # get selected powerset
                choices = powerset_dict[rule][powerset_choice]

                # enable edges
                for c in choices:
                    _, edge_type, edge_id = edges[c]
                    batch[edge_type].edge_weight[edge_id][i] = 1

                if isinstance(rule, NPPRule):
                    npp_choices_dict[rule].append(choices)

        # initialize batch
        batch = condense_edges_pyg(batch, device=device)

        # broadcast ids for certain atoms across each graph
        certain_atom_ids = torch.tensor(rg.certain_atom_ids, device=device)
        query_sinks_batch = query_sinks

        graph_block = GraphBlock(
            dict(batch.node_items()),
            dict(batch.edge_items()),
            torch.cat(
                [
                    torch.tensor(choices, device=device)
                    for choices in npp_choices_dict.values()
                ],
                dim=-1,
            )
            if npp_choices_dict
            else torch.empty(n_combinations, 0, device=device),
            certain_atom_ids=certain_atom_ids,
            sink_ids=query_sinks_batch,
        )

        return graph_block

    def solve(self, graph_block: GraphBlock, max_iter: int = -1) -> GraphBlock:
        return self.solver.solve(graph_block, max_iter)

    def zero_grad(self) -> None:
        """TODO"""
        for npp_cfg in self.npp_cfg_dict.values():
            if "optimizer" in npp_cfg and npp_cfg["optimizer"] is not None:
                npp_cfg["optimizer"].zero_grad()

    def step(self) -> None:
        """TODO"""
        for npp_cfg in self.npp_cfg_dict.values():
            if "optimizer" in npp_cfg and npp_cfg["optimizer"] is not None:
                npp_cfg["optimizer"].step()

    def get_answer_sets(
        self,
        queries: Optional[List[Constraint]] = None,
        npp_data: Optional[Dict[str, Tuple[Any]]] = None,
        device: Optional[torch.device] = None,
        num_phases: int = 1,
    ) -> Union[Dict[Constraint, List[Set[PredLiteral]]], List[Set[PredLiteral]]]:
        if device is None:
            device = self.device

        # NPP forward
        npp_ctx_dict = self.npp_forward(npp_data)

        # initialize solving context
        solving_ctx = SolvingContext(
            len(queries) if queries else 1,
            npp_ctx_dict,
        )

        for phase in range(num_phases):
            # prepare batch (includes NPP forward)
            graph_block = self.prepare_block(
                queries=queries,
                device=device,
                phase=phase,
            )

            # solve graph block
            graph_block = self.solve(graph_block)

            # update stable models
            solving_ctx.update_SMs(graph_block)

        # synchronize SMs across processes
        solving_ctx.synchronize_SMs()

        SM_dict = {}

        if queries is None:
            queries = [None]

        for query, query_is_SM in zip(queries, solving_ctx.sm_ctx.is_SM):
            # get labels for SMs
            SM_dict[query] = [
                {
                    label
                    for label, atom in zip(self.rg.node_dict["atom"]["label"], atoms)
                    if torch.isclose(atom, torch.ones_like(atom))
                }
                for atoms in solving_ctx.sm_ctx.atoms[query_is_SM.squeeze(-1)]
            ]

        # gather all SMs in main process
        if self.world_size > 1:
            # main process
            if self.rank == 0:
                SM_dict_list = [None] * self.world_size

                # receive local SMs from secondary processes
                dist.gather_object(
                    SM_dict,
                    SM_dict_list,
                    dst=0,
                    group=self.group,
                )
                # merge local SM dictionaries into single global one
                for query in SM_dict.keys():
                    SM_dict[query] = sum([d[query] for d in SM_dict_list], [])

            # secondary process
            else:
                # send local SMs to main process
                dist.gather_object(
                    SM_dict,
                    None,
                    dst=0,
                    group=self.group,
                )

        if queries == [None]:
            return SM_dict[None]

        return SM_dict


# TODO: warning / error if not all NPPs configured
