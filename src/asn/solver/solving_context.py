from functools import cached_property
from typing import TYPE_CHECKING, Dict, Iterable, Optional

import torch
import torch.distributed as dist

from .stable_model_context import StableModelContext

if TYPE_CHECKING:
    from ground_slash.program import NPPRule
    from torch.distributed.distributed_c10d import ProcessGroup

    from asn.solver.graph_block import GraphBlock


class SolvingContext:
    """TODO"""

    def __init__(
        self,
        batch_size: int = 1,
        npp_ctx_dict: Optional[Dict["NPPRule", torch.Tensor]] = None,
        sm_ctx: Optional[StableModelContext] = None,
        minimize: bool = True,
        rank: int = 0,
        world_size: int = 1,
        group: Optional["ProcessGroup"] = None,
    ) -> None:
        """TODO"""
        self.batch_size = batch_size
        self.npp_ctx_dict = npp_ctx_dict

        if world_size < 1:
            raise ValueError(f"World size should be positive, but was {world_size}.")
        self.world_size = world_size
        if rank >= world_size:
            raise ValueError(f"Rank should be smaller than world_size, but was {rank}.")
        self.rank = rank

        # create group with all processes
        if group is None and world_size > 1:
            group = dist.new_group(list(range(self.world_size)))
        self.group = group

        # stable model context
        self.sm_ctx = sm_ctx

        # whether or not to compute minimal models
        self.minimize = minimize

        # flag indicating whether or not the solving context is synchronized across processes
        self.synchronized = False

    def clear_cache(self, attrs: Optional[Iterable[str]] = None) -> None:
        """TODO"""
        if attrs is None:
            # per default clear all
            attrs = ("p_I", "p_Q")

        for attr in attrs:
            # clear cached properties
            self.__dict__.pop(attr, None)

    def filter_SMs(self, atoms: torch.Tensor, is_SM: torch.Tensor) -> None:
        """TODO"""
        # NOTE: 'is_SM' should hold the preliminary indicators for stable models
        # at the very least it should be initialized to 'is_model'

        # filter for stable models
        for k in range(is_SM.shape[1]):
            # check if I_k is superset of I_j, j > k
            is_supseteq = torch.all(
                torch.eq(
                    # element-wise OR with all subsequent interpretations
                    # I_k LOR I_j for j > k
                    torch.logical_or(atoms[[k], :], atoms[k+1:, :]),
                    # -> n_combinations-(k+1) x n_atoms
                    atoms[[k], :]
                ),
                # -> n_combinations-(k+1) x n_atoms
                dim=1,
                keepdims=True,
            ).unsqueeze(0)
            # -> 1 x n_combinations-(k+1) x 1

            # update I_k to reflect whether it is a superset of another stable model candidate
            is_SM[:, [k], :] *= ~torch.any(is_supseteq * is_SM[:, k + 1 :, :], dim=1, keepdims=True)

            # check if I_k is subset of I_j, j > k
            is_subseteq = torch.all(
                torch.eq(
                    # element-wise OR with all subsequent interpretations
                    # I_k LAND I_j for j > k
                    torch.logical_and(atoms[[k], :], atoms[k+1:, :]),
                    # -> n_combinations-(k+1) x n_atoms
                    atoms[[k], :]
                ),
                # -> n_combinations-(k+1) x n_atoms
                dim=1,
                keepdims=True,
            ).unsqueeze(0)
            # -> 1 x n_combinations-(k+1) x 1
    
            # update I_j to reflect whether it is a subset of another stable model candidate
            is_SM[:, k+1:, :] *= ~(is_subseteq * is_SM[:, [k], :])

    def update_SMs(self, graph_block: "GraphBlock") -> None:
        atoms = graph_block.atoms
        # -> n_combinations x n_atoms

        # initialize mask to indicate whether interpretation is a model or not
        is_SM = graph_block.is_model
        # -> n_unique_queries x n_combinations x 1

        # NPP contexts
        npp_choices = graph_block.npp_choices
        # m x n_NPPs

        # filter out any interpretations which are not a model for SOME query
        is_SM_for_some = is_SM.any(dim=0).squeeze(-1)
        atoms = atoms[is_SM_for_some, :]
        is_SM = is_SM[:, is_SM_for_some, :]
        npp_choices = npp_choices[is_SM_for_some, :]

        if self.sm_ctx is not None:
            # append current stable model (candidates)
            atoms = torch.cat((atoms, self.sm_ctx.atoms), dim=0)
            is_SM = torch.cat((is_SM, self.sm_ctx.is_SM), dim=1)
            npp_choices = torch.cat((npp_choices, self.sm_ctx.npp_choices), dim=0)

        if self.minimize:
            # filter for minimality
            self.filter_SMs(atoms, is_SM)

        # mask indicating whether interpretation is a stable model for ANY query
        is_SM_for_some = is_SM.any(dim=0).squeeze(-1)

        # store all relevant information to keep track of updated stable models
        self.sm_ctx = StableModelContext(
            atoms[is_SM_for_some, :],
            is_SM[:, is_SM_for_some, :],
            npp_choices[is_SM_for_some, :],
            graph_block.inverse_unique_sink_ids,
        )

        # indicate that solving context may not be synchronized anymore
        self.synchronized = False

    def synchronize_SMs(self) -> None:
        if self.world_size > 1 and self.minimize:
            atoms = self.sm_ctx.atoms
            # -> m x n_atoms
            # (where m is the number of combinations where the interpretation is considered a SM for some query)
            is_SM = self.sm_ctx.is_SM
            # -> n_unique_queries x m x 1

            # NPP contexts
            npp_choices = self.sm_ctx.npp_choices
            # m x n_NPPs

            # main process
            if self.rank == 0:
                # receive 'filtered_atoms' from all processes
                atoms_list = [None] * self.world_size
                dist.gather_object(
                    atoms,
                    atoms_list,
                    dst=0,
                    group=self.group,
                )
                # receive 'filtered_is_model' from all processes
                is_SM_list = [None] * self.world_size
                dist.gather_object(
                    is_SM,
                    is_SM_list,
                    dst=0,
                    group=self.group,
                )
                # infer number of interpretations received from all processes (for splitting later)
                m_list = [t.shape[0] for t in atoms_list]
                device = atoms.device
                # concatenate received tensors together
                atoms_concat = torch.concat([t.to(device) for t in atoms_list], dim=0)
                is_SM_concat = torch.concat([t.to(device) for t in is_SM_list], dim=1)

                # filter for stable models
                self.filter_SMs(atoms_concat, is_SM_concat)

                # scatter final 'is_SM' entries
                scattered_list = [None]
                dist.scatter_object_list(
                    scattered_list,
                    list(torch.split(is_SM_concat, m_list, dim=1)),
                    src=0,
                    group=self.group,
                )
                is_SM = scattered_list[0].to(device)

                # mask indicating whether interpretation is a stable model for ANY query
                is_SM_for_some = is_SM.any(dim=0).squeeze(-1)

                # store all relevant information to keep track of updated stable models
                self.sm_ctx = StableModelContext(
                    atoms[is_SM_for_some, :],
                    is_SM[:, is_SM_for_some, :],
                    npp_choices[is_SM_for_some, :],
                    self.sm_ctx.inverse_sink_ids,
                )
            # secondary process
            else:
                # send 'filtered_atoms' to main process
                dist.gather_object(
                    atoms,
                    None,
                    dst=0,
                    group=self.group,
                )
                # send 'filtered_is_model' to main process
                dist.gather_object(
                    is_SM,
                    None,
                    dst=0,
                    group=self.group,
                )
                # receive updated 'is_SM' entries
                scattered_list = [None]
                dist.scatter_object_list(
                    scattered_list,
                    [None] * self.world_size,
                    src=0,
                    group=self.group,
                )
                # update SM mask
                is_SM = scattered_list[0].to(atoms.device)

                # mask indicating whether interpretation is a stable model for ANY query
                is_SM_for_some = is_SM.any(dim=0).squeeze(-1)

                # store all relevant information to keep track of updated stable models
                self.sm_ctx = StableModelContext(
                    atoms[is_SM_for_some, :],
                    is_SM[:, is_SM_for_some, :],
                    npp_choices[is_SM_for_some, :],
                    self.sm_ctx.inverse_sink_ids,
                )

        # indicate that solving context is synchronized
        self.synchronized = True

    @cached_property
    def p_I(self) -> torch.Tensor:
        """TODO"""
        is_SM = self.sm_ctx.is_SM

        if self.npp_ctx_dict:
            return (
                is_SM[self.sm_ctx.inverse_sink_ids]
                * torch.cat(
                    [
                        torch.gather(
                            npp_ctx.p,
                            # -> batch_size x n_out
                            -1,
                            npp_choices.repeat(npp_ctx.p.shape[0], 1)
                            # -> batch_size x n_combinations
                        ).unsqueeze(-1)
                        # -> batch_size x n_combinations x 1
                        for npp_ctx, npp_choices in zip(
                            self.npp_ctx_dict.values(), self.sm_ctx.npp_choices.T
                        )
                    ],
                    dim=-1,
                ).prod(dim=-1, keepdims=True)
                / torch.tensor(len(self.npp_ctx_dict))
            )
            # -> batch_size x n_combinations x 1
        else:
            return is_SM[self.sm_ctx.inverse_sink_ids]
            # -> batch_size x n_combinations x 1

    @cached_property
    def p_Q(self) -> torch.Tensor:
        """TODO"""
        p_Q = self.p_I.sum(dim=-2)

        if self.world_size > 1:
            if self.synchronized:
                # reduce incomplete probabilities across all processes
                dist.all_reduce(
                    p_Q,
                    group=self.group,
                )
            else:
                raise Exception(
                    "Computing query probabilities from unsynchronized solving context."
                )

        return p_Q

    @property
    def npp_grads(self) -> Dict["NPPRule", torch.Tensor]:
        """TODO"""
        p_I = self.p_I
        p_Q = self.p_Q

        npp_grads = {}

        for (npp, npp_ctx), npp_choices in zip(
            self.npp_ctx_dict.items(), self.sm_ctx.npp_choices.T
        ):
            with torch.no_grad():
                p_c_eq_vi = npp_ctx.p.unsqueeze(1)
                # -> batch_size x 1 x n_out

                # TODO: wir haben 'npp_choices' gespeichert
                choices_mask = torch.zeros(
                    npp_choices.shape[0],
                    npp_ctx.p.shape[1],
                    device=npp_ctx.p.device,
                    dtype=torch.bool,
                ).scatter_(index=npp_choices.unsqueeze(-1), dim=1, value=1.0)
                # -> n_combinations x n_out

                # TODO: umwandeln von 'npp_choices' zu 'choices_mask'
                c_eq_vi = p_c_eq_vi * choices_mask
                # -> batch_size x n_combinations x n_out

                p_interp_div_c_eq_vi = p_I / c_eq_vi
                p_interp_div_c_eq_vi[c_eq_vi == 0.0] = 0.0
                # -> batch_size x n_combinations x n_out
                pos_grads = (p_interp_div_c_eq_vi).sum(dim=-2)
                # -> batch_size x n_out
                neg_grads = pos_grads - pos_grads.sum(dim=-1, keepdims=True)
                # -> batch_size x n_out
                # TODO: if p(Q) is zero -> division by zero -> NaNs
                npp_grads[npp] = (pos_grads + neg_grads) / p_Q
                # -> batch_size x n_out

        return npp_grads

    @property
    def npp_loss(self) -> torch.Tensor:
        """TODO"""
        npp_grads = self.npp_grads
        loss = torch.tensor(0.0, device=self.p_I.device)

        for npp, grads in npp_grads.items():
            # multiply manual gradients by NPP outputs
            # this way the gradients during backward are exactly our gradients
            preped_grads = self.npp_ctx_dict[npp].p * grads
            # update loss
            loss += preped_grads[preped_grads.abs() != 0.0].sum()

        # return normalized loss
        return loss / self.batch_size
