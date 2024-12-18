import io
import itertools
from collections import defaultdict
from copy import deepcopy
from math import isfinite
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import pygraphviz as pgv
import torch
from ground_slash.program import (
    NPP,
    AggrCount,
    AggrElement,
    AggrLiteral,
    BuiltinLiteral,
    Choice,
    ChoiceRule,
    Constraint,
    DisjunctiveRule,
    Expr,
    FalseConstant,
    Guard,
    Infimum,
    Literal,
    LiteralCollection,
    Naf,
    Neg,
    NormalRule,
    NPPRule,
    Number,
    PredLiteral,
    Program,
    Statement,
    TermTuple,
    TrueConstant,
)
from PIL import Image
from torch_geometric.data import HeteroData

# TODO: refactor
from asn.utils import relop_dict
from asn.utils.collections import get_minimal_collections

from .expression import ComplexQuery, Conjunction, Disjunction


class ReasoningGraph:
    def __init__(
        self,
        prog: Program,
        certain_atoms: Optional[Set[PredLiteral]] = None,
    ) -> None:
        """
        Args:
            prog: SLASH `Program` instance.
            certain_atoms: optional set of atoms (`PredLiteral` instances) whose nodes
                are initialized to `True`. Can be used to reduce the number of
                iterations during solving.

        Raises:
            TODO:
        """
        # ---------- init graph ----------
        self.node_types = ("atom", "disj", "conj", "count", "sum", "min", "max")

        if certain_atoms is None:
            certain_atoms = set()

        # list of ids for query-specific SAT nodes (i.e. 'False')
        self.query_sinks = []

        # map for some specific unicode symbols
        self.__unicode_symbols = {
            "true": "\u22a5",
            "false": "\u22a4",
            "disj": "\u2228",
            "conj": "\u2227",
            "neq": "\u2260",
            "leq": "\u2264",
            "geq": "\u2265",
        }

        # node & edge dictionaries
        self.node_dict: Dict[str, Dict[str, List]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.edge_dict: Dict[Tuple[str], Dict[str, List]] = defaultdict(
            lambda: defaultdict(list)
        )

        # dictionaries mapping ASP constructs to node ids
        self.node_id_dict: Dict[Expr, Tuple[str, int]] = dict()
        self.edge_id_dict: Dict[Tuple[Expr, Expr], int] = dict()

        # dictionaries mapping conj. ids to list of edges for choices/disjs. & NPPs
        # TODO: typing
        self.choice_edges: Dict[
            Union[Choice, LiteralCollection], List[Tuple[Literal, int, int]]
        ] = dict()
        # TODO: typing
        self.npp_edges: Dict[NPP, List[Tuple[Literal, int, int]]] = dict()

        # set of `Choice` & `NPP` instances that are already incorporated in the graph
        # (to avoid duplicate encodings)
        self.choices: Set[Choice, LiteralCollection] = set()
        self.npps: Set[NPP] = set()

        # create literals for constants 'True' and 'False'
        self.true_const = TrueConstant()
        self.false_const = FalseConstant()

        # initialize constant nodes for 'True' and 'False'
        # 'True' represented as conj. node (no inputs result in True)
        self.add_node(
            self.true_const,
            "conj",
            self.__unicode_symbols["true"],
            x=1.0,
        )
        # 'False' represented as disj. node (no inputs result in False)
        self.add_node(
            self.false_const,
            "disj",
            self.__unicode_symbols["false"],
        )

        # ---------- process program ----------
        for stmt in prog.statements:
            self.encode_statement(stmt, certain_atoms)

        # map certain atoms to their node ids
        self.certain_atom_ids = [self.node_id_dict[atom][1] for atom in certain_atoms]

    def encode_statement(
        self,
        statement: Statement,
        certain_atoms: Optional[Set[PredLiteral]] = None,
    ) -> None:
        if certain_atoms is None:
            certain_atoms = set()

        # check if statement is ground
        if not statement.ground:
            raise ValueError(f"Statement {str(statement)} is not ground.")

        # --------------- process body ---------------

        if any(
            isinstance(literal, BuiltinLiteral) and not literal.eval()
            for literal in statement.body
        ):
            # false built-in literal (i.e., body never satisfied)
            # no need to process rule
            return

        body_literals = []
        body_literal_signs = []

        # pre-process body literals
        for literal in statement.body:
            # encode literal (if not exists)
            sign = self.encode_literal(literal, certain_atoms)

            # predicate or aggregate literal
            if sign != 0:
                body_literals.append(literal)
                body_literal_signs.append(sign)
            # removes built-in literals from body
            # we already know that these evaluate to 'True'

        # fact
        if not body_literals:
            body_literals.append(self.true_const)
            body_literal_signs.append(1)

        # TODO: better way?
        body_literals = Conjunction(*body_literals)

        # single body literal
        if len(body_literals) == 1:
            # use literal directly
            body_key = abs(body_literals[0])
            body_sign = body_literal_signs[0]
        # conjunction of body literals
        else:
            body_key = body_literals

            # connect body literals to a conjunction node (if not exists)
            if body_key not in self.node_id_dict:
                self.add_node(
                    body_key,
                    "conj",
                    f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                )

                for literal, sign in zip(body_literals, body_literal_signs):
                    self.add_edge(
                        abs(literal),
                        body_key,
                        edge_weight=float(sign),
                    )

            body_sign = 1

        # --------------- process head ---------------

        consequents = defaultdict(list)

        if isinstance(statement, ChoiceRule):
            # choice rule
            choice = statement.head

            for element in choice.elements:
                consequents[element.atom].append(Conjunction(*element.literals))
        elif isinstance(statement, NPPRule):
            # NPP rule
            choice = statement.npp.as_choice()

            for element in choice.elements:
                consequents[element.atom].append(Conjunction(*element.literals))
        else:
            # normal/disjunctive rules
            for atom in statement.head:
                consequents[atom].append(Conjunction())

            # normal/disjunctive rules
            if not consequents:
                consequents[self.false_const].append(Conjunction())

        # dictionary to store the edges corresponding to a choice/disjunction
        # NOTE: used later if statement actually non-deterministic
        choice_edges = list()
        # TODO
        cond_map = dict()

        # iterate over all consequent literals
        for i, (cond_literal, conditions) in enumerate(consequents.items()):
            # if 'cond_literal' is not a query sink (i.e., Constraint expression)
            if not isinstance(cond_literal, (Constraint, ComplexQuery)):
                # encode or update literal node
                # NOTE: is always positive due to language specifications
                self.encode_literal(cond_literal, certain_atoms)

            # ----- process conditions -----

            literal_conditions = []

            # pre-process conditions
            # remove builtin-literals and check their satisfiability
            for cond in conditions:
                literals = []

                for literal in cond:
                    if isinstance(literal, BuiltinLiteral):
                        if not literal.eval():
                            # condition can never be satisfied (remove)
                            break

                    literals.append(literal)
                else:
                    # condition can be satisfied (keep)
                    literal_conditions.append(Conjunction(*literals))

            # get minimal conditions
            # (supersets irrelevant if a subset already satisfies condition)
            minimal_cond_candidates = get_minimal_collections(*literal_conditions)

            # list of tuples containing the keys to different conditions and their sign
            cond_keys = []
            # list of aggregate elements for a choice aggregate
            # NOTE: used later if statement is a choice rule
            count_elements = []

            # process final conditions
            for cond in minimal_cond_candidates:
                # encode literals (if not already)
                # NOTE: do NOT need to check for existence of literals from here on out
                for literal in cond:
                    self.encode_literal(literal, certain_atoms)

                # save aggregate element for choice rule/NPP
                count_elements.append(
                    AggrElement(
                        TermTuple(Number(i)),
                        LiteralCollection(cond_literal, *cond),
                    )
                )

                # empty condition (unconditional)
                if len(cond) == 0:
                    continue
                # single condition literal (use directly)
                elif len(cond) == 1:
                    # sign depends on literal since we its node directly
                    cond_keys.append((abs(cond[0]), -1 if cond[0].naf else 1))
                # multiple condition literals (combine in conjunction)
                else:
                    if cond not in self.node_id_dict:
                        # create new conj. node
                        self.add_node(
                            cond,
                            "conj",
                            f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                        )

                        # add edges from literals to conj.
                        for literal in cond:
                            self.add_edge(
                                abs(literal), cond, edge_weight=-1 if literal.naf else 1
                            )

                    # sign is always positive
                    cond_keys.append((cond, 1))

            # no condition (unconditional)
            if len(cond_keys) == 0:
                # NOTE: even if body is just 'True', we need the edge here
                choice_edges.append(
                    (
                        cond_literal,
                        *self.add_edge(body_key, cond_literal, edge_weight=body_sign),
                    )
                )

                cond_map[cond_literal] = None

            # single condition (use directly)
            elif len(cond_keys) == 1:
                # NOTE: condition already encoded as a conjunction (no need to check)
                cond_key, sign = cond_keys[0]
                conj_key = (
                    Conjunction(*body_key, cond_key)
                    if isinstance(body_key, LiteralCollection)
                    else Conjunction(body_key, cond_key)
                )

                if conj_key not in self.node_id_dict:
                    self.add_node(
                        conj_key,
                        "conj",
                        f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                    )

                    if body_key != self.true_const:
                        self.add_edge(
                            body_key,
                            conj_key,
                            edge_weight=body_sign,
                        )

                    self.add_edge(cond_key, conj_key, edge_weight=float(sign))

                choice_edges.append(
                    (
                        cond_literal,
                        *self.add_edge(
                            conj_key,
                            cond_literal,
                        ),
                    )
                )

                cond_map[cond_literal] = conj_key

            # multiple conditions (combine in disjunction)
            else:
                disj_key = Disjunction(cond for cond, _ in cond_keys)
                conj_key = (
                    Conjunction(*body_key, disj_key)
                    if isinstance(body_key, LiteralCollection)
                    else Conjunction(body_key, disj_key)
                )

                if conj_key not in self.node_id_dict:
                    self.add_node(
                        conj_key,
                        "conj",
                        f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                    )

                    if disj_key not in self.node_id_dict:
                        self.add_node(
                            disj_key,
                            "disj",
                            f"{self.__unicode_symbols['disj']}_{{{len(self.node_dict['disj']['x'])-1}}}",
                        )

                        for cond, sign in cond_keys:
                            self.add_edge(
                                cond,
                                disj_key,
                                edge_weight=sign,
                            )

                    self.add_edge(
                        disj_key,
                        conj_key,
                    )

                    if body_key != self.true_const:
                        self.add_edge(
                            body_key,
                            conj_key,
                            edge_weight=body_sign,
                        )

                choice_edges.append(
                    (
                        cond_literal,
                        *self.add_edge(
                            conj_key,
                            cond_literal,
                        ),
                    )
                )

                cond_map[cond_literal] = conj_key

        if isinstance(statement, (NormalRule, Constraint)):
            # done
            return

        if isinstance(statement, DisjunctiveRule):
            head_literals = statement.head

            # constraint that is active ONLY if the rules body is satisfied
            # AND NONE of the head literals is satisfied
            constr_key = Conjunction(
                *[Naf(deepcopy(atom), True) for atom in head_literals], *body_literals
            )

            if constr_key not in self.node_id_dict:
                self.add_node(
                    constr_key,
                    "conj",
                    rf"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                )

            self.add_edge(body_key, constr_key, edge_weight=body_sign)

            for literal in head_literals:
                self.add_edge(
                    literal,
                    constr_key,
                    edge_weight=-1.0,
                )

            self.add_edge(
                constr_key,
                self.false_const,
            )

            if statement not in self.choices:
                # TODO: necessary ???
                self.choices.add(statement)
                # TODO: best way to store choices?
                self.choice_edges[statement] = choice_edges

        elif isinstance(statement, (ChoiceRule, NPPRule)):
            choice_aggr = AggrLiteral(
                AggrCount(),
                tuple(count_elements),
                choice.guards,
            )

            # TODO: guard encoding

            # encode choice aggregate
            if choice_aggr not in self.node_id_dict:
                self.add_node(
                    choice_aggr,
                    "count",
                    f"\#count_{{{len(self.node_dict['count']['x'])}}}",
                    guards=tuple(self.encode_guards(choice_aggr.guards)),
                )

                for literal, cond_key in cond_map.items():
                    if cond_key is None:
                        self.add_edge(
                            literal,
                            choice_aggr,
                        )
                    else:
                        conj_key = Conjunction(literal, cond_key)

                        if conj_key not in self.node_id_dict:
                            self.add_node(
                                conj_key,
                                "conj",
                                f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                            )

                            self.add_edge(
                                cond_key,
                                conj_key,
                            )
                            self.add_edge(
                                literal,
                                conj_key,
                            )

                        self.add_edge(
                            conj_key,
                            choice_aggr,
                        )

            # add choice constraint
            constr_key = Conjunction(*body_literals, Naf(choice_aggr))

            if constr_key not in self.node_id_dict:
                self.add_node(
                    constr_key,
                    "conj",
                    f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                )

                self.add_edge(
                    body_key,
                    constr_key,
                    edge_weight=body_sign,
                )
                self.add_edge(
                    choice_aggr,
                    constr_key,
                    edge_weight=-1,
                )

            self.add_edge(constr_key, self.false_const)

            if isinstance(statement, ChoiceRule) and statement not in self.choices:
                # TODO: necessary ???
                self.choices.add(statement)
                # TODO: best way to store choices?
                self.choice_edges[statement] = choice_edges

            if isinstance(statement, NPPRule) and statement not in self.choices:
                # TODO: necessary ???
                self.npps.add(statement)
                # TODO: best way to store choices?
                self.npp_edges[statement] = choice_edges

    def encode_literal(
        self,
        literal: Literal,
        certain_atoms: Optional[Set[PredLiteral]] = None,
    ) -> int:
        if certain_atoms is None:
            certain_atoms = set()

        if isinstance(literal, BuiltinLiteral):
            # nothing to do here
            return 0
        elif isinstance(literal, AggrLiteral):
            aggr = abs(literal)

            # positive or negative aggregate
            self.encode_aggregate(aggr)
        else:
            atom = abs(literal)

            # initialize probability with 1.0 if atom is certain (i.e., fact)
            p = float(atom in certain_atoms)

            # register literal if not exits
            try:
                # update existing node
                # use maximum possible value (a fact is not invalidated by a rule)
                node_type, literal_id = self.node_id_dict[atom]
                self.node_dict[node_type]["x"][literal_id] = max(
                    self.node_dict[node_type]["x"][literal_id], p
                )
            # update value if it does
            except KeyError:
                # create new atom node
                # since 'False' already registed, we can safely assume that all new head
                # literals are atoms
                self.add_node(
                    atom,
                    "atom",
                    label=str(atom),
                    x=p,
                )

                # check if strong negation is also encoded in the graph
                neg_atom = Neg(atom, not atom.neg)

                if neg_atom in self.node_id_dict:
                    # add constraint that both cannot be true at the same time
                    # NOTE: since 'atom' is just encoded, we know there is no conj. yet

                    conj_key = Conjunction(atom, neg_atom)

                    self.add_node(
                        conj_key,
                        "conj",
                        f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                    )

                    self.add_edge(
                        atom,
                        conj_key,
                    )
                    self.add_edge(
                        neg_atom,
                        conj_key,
                    )
                    self.add_edge(conj_key, self.false_const)

        return -1 if literal.naf else 1

    def encode_aggregate(
        self,
        aggr: AggrLiteral,
        certain_atoms: Optional[Set[PredLiteral]] = None,
    ) -> int:
        """Encodes an aggregate in the graph.

        Args:
            aggr: `AggrLiteral` instance.

        Raises:
            TODO
        """
        # TODO: optimize & update to use new 'encode_literal' function!

        if certain_atoms is None:
            certain_atoms = set()

        aggr_type = str(aggr.func)[1:]

        if aggr not in self.node_id_dict:
            # create new aggregate node
            self.add_node(
                aggr,
                aggr_type,
                label=f"{str(aggr.func)}_{{{len(self.node_dict[aggr_type]['x'])-1}}}",
                guards=tuple(self.encode_guards(aggr.guards)),
            )

        # ---------- sort elements ----------
        # dictionary mapping a tuple to possible conditions satisfying it
        # (multiple possible; only one needs to hold)
        cond_dict = defaultdict(lambda: defaultdict(lambda: None))

        for elem in aggr.elements:
            predicate_literals = []
            for literal in elem.literals:
                if isinstance(literal, BuiltinLiteral):
                    if not literal.eval():
                        # false built-in literal (i.e., body never satisfied)
                        break
                else:
                    # keep classical literals
                    predicate_literals.append(literal)
            # run if loop did not break early
            else:
                # keep track of condition for tuple
                literals = Conjunction(*predicate_literals)
                cond_dict[elem.terms][literals]

                # register atoms in aggregate element (if not already)
                for literal in literals:
                    self.encode_literal(abs(literal), certain_atoms)

        # unconditional tuples (i.e., always satisfied)
        uncond_tuples = []

        # ---------- process tuples and conditions ----------
        for (
            tup,
            cond_candidates,
        ) in cond_dict.items():
            # condition always satisfied
            # (check here to avoid construction of minimal collections)
            if Conjunction() in cond_candidates:
                uncond_tuples.append(tup)
                continue

            # get minimal conditions
            # (supersets irrelevant if a subset already satisfies condition)
            minimal_cond_candidates = get_minimal_collections(*cond_candidates)

            # keep track of condition conjunctions
            tuple_signature = Disjunction(
                Conjunction(*condition) for condition in minimal_cond_candidates
            )

            if tuple_signature not in self.node_id_dict:
                # create auxiliary atom representing satisfied tuple
                self.add_node(
                    tuple_signature,
                    "disj",
                    label=f"{self.__unicode_symbols['disj']}_{{{len(self.node_dict['disj']['x'])-1}}}",
                )

            # connect auxiliary node to aggregate node
            # TODO: better way to compute tuple weight?
            self.add_edge(
                tuple_signature, aggr, edge_weight=float(aggr.func.eval({tup}).eval())
            )

            # process tuple conditions
            for condition in minimal_cond_candidates:
                if len(condition) == 1:
                    literal = condition[0]
                    pos_literal = abs(literal)

                    if pos_literal not in self.node_id_dict:
                        self.add_node(
                            pos_literal,
                            "atom",
                            str(pos_literal),
                        )

                    self.add_edge(
                        pos_literal,
                        tuple_signature,
                        edge_weight=1.0 if not literal.naf else -1.0,
                    )
                else:
                    # NOTE: already handled non-conditional tuples earlier
                    conj_signature = Conjunction(*condition)

                    # check if equivalent conjunction exists
                    if conj_signature not in self.node_id_dict:
                        # create new conjunction node
                        self.add_node(
                            conj_signature,
                            "conj",
                            f"{self.__unicode_symbols['conj']}_{{{len(self.node_dict['conj']['x'])-1}}}",
                        )

                        # connect literals to conjunction node
                        for literal in condition:
                            self.add_edge(
                                abs(literal),
                                conj_signature,
                                edge_weight=1.0 if not literal.naf else -1.0,
                            )

                    self.add_edge(
                        conj_signature,
                        tuple_signature,
                    )

        # edge from 'True' to aggregate auxiliary atom
        # (weight based on aggregate of all certain tuples)
        if uncond_tuples:
            self.add_edge(
                self.true_const,
                aggr,
                edge_weight=float(aggr.func.eval(set(uncond_tuples)).eval()),
            )

    def encode_query(
        self, query: Union[Constraint, ComplexQuery], certain_atoms: Optional[Set[PredLiteral]] = None
    ) -> int:
        """Adds a query to the reasoning graph.

        Args:
            query: `Constraint` instance.
            certain_atoms: optional set of atoms (`PredLiteral` instances) whose nodes
                are initialized to `True`. Can be used to reduce number of iterations.

        Raises:
            TODO
        """
        try:
            _, sink_id = self.node_id_dict[query]
            self.query_sinks.append(sink_id)

            return sink_id
        except KeyError:
            # add new query-specific sink
            sink_id = self.add_node(
                query,
                "disj",
                label=str(query),
            )

        # keep track of global sink
        global_sink = self.false_const

        # set new sink to query
        self.false_const = query
        self.query_sinks.append(sink_id)

        # simple query (single constraint)
        if isinstance(query, Constraint):
            # process query as a regular constraint (new sink is used instead)
            self.encode_statement(query, certain_atoms)
        # complex query (multiple constraints)
        else:
            for q in query.constraints:
                # process query as a regular constraint (new sink is used instead)
                self.encode_statement(q, certain_atoms)

        # reset sink to global sink
        self.false_const = global_sink

        # connect global sink to query sink
        self.add_edge(
            self.false_const,
            query,
        )

        return sink_id

    def add_node(
        self,
        expr: Expr,
        node_type: str,
        label: str,
        **attrs: Dict[str, Any],
    ) -> None:
        """TODO"""

        # if node_type == "conj" and len(self.node_dict['conj']['x']) == 2:
        #    raise Exception(label)

        # get node ID
        node_id = len(self.node_dict[node_type]["x"])

        if expr in self.node_id_dict:
            raise ValueError(f"Node representing '{str(expr)}' already exists.")

        # add node attributes
        self.node_dict[node_type]["label"].append(label)

        if "x" not in attrs:
            attrs["x"] = 0.0

        for attr, val in attrs.items():
            self.node_dict[node_type][attr].append(val)

        # track expression encoded by node
        self.node_id_dict[expr] = (node_type, node_id)

        return node_id

    def get_node(
        self,
        expr: Expr,
    ) -> Optional[Tuple[str, int]]:
        """TODO"""

        try:
            return self.node_id_dict[expr]
        except KeyError:
            return None

    def add_edge(
        self,
        src_expr: Expr,
        dst_expr: Expr,
        **attrs: Dict[str, Any],
    ) -> Tuple[Tuple[str, str, str], int]:
        """TODO"""
        try:
            src_type, src_id = self.node_id_dict[src_expr]
        except KeyError:
            raise ValueError(f"No node representing expression {str(src_expr)}")

        try:
            dst_type, dst_id = self.node_id_dict[dst_expr]
        except KeyError:
            raise ValueError(f"No node representing expression {str(dst_expr)}")

        edge_type = (src_type, "to", dst_type)

        # get edge ID
        edge_id = len(self.edge_dict[edge_type]["edge_index"])

        # add edge attributes
        if (src_id, dst_id) in self.edge_dict[edge_type]["edge_index"]:
            # TODO
            raise ValueError(
                f"Edge from {str(src_expr)} to {str(dst_expr)} already exists."
            )

        self.edge_dict[edge_type]["edge_index"].append((src_id, dst_id))

        if "edge_weight" not in attrs:
            attrs["edge_weight"] = 1.0

        for attr, val in attrs.items():
            self.edge_dict[edge_type][attr].append(val)

        return edge_type, edge_id

    def get_edge(
        self,
        edge_type: Tuple[str, str, str],
        src_expr: Expr,
        dst_expr: Expr,
    ) -> Optional[int]:
        """TODO"""

        try:
            src_id = self.node_id_dict[src_expr]
            dst_id = self.node_id_dict[dst_expr]

            return self.edge_dict[edge_type]["edge_index"].index((src_id, dst_id))
        except KeyError:
            return None
        except ValueError:
            return None

    def encode_guards(self, guards: Tuple[Guard, Guard]) -> Tuple[int, int, int, int]:
        guard_encoding = []

        # parse guards
        for guard in guards:
            if guard is None:
                guard_encoding += [-1, -1]
            else:
                if isinstance(guard.bound, Number):
                    bound = guard.bound.eval()
                else:
                    # infimum is the only object that precedes numbers in
                    # the total ordering for terms
                    # use +-infinity respectively
                    bound = (
                        -float("inf")
                        if isinstance(guard.bound, Infimum)
                        else float("inf")
                    )

                guard_encoding += [
                    relop_dict[guard.op],
                    bound,
                ]

        return guard_encoding

    def to_pyg(
        self,
        device: Optional[torch.device] = None,
        hard: bool = True,
        copies: int = 1,
    ) -> HeteroData:
        """TODO"""

        if copies < 1:
            raise ValueError(
                f"Number of copies for reasoning graph must be larger than zero, but was: {copies}."
            )

        # TODO: use quantized int8 for 'soft' values?

        # number of nodes per type
        node_types = ("atom", "disj", "conj", "count", "sum", "min", "max")
        num_nodes = tuple(
            [len(self.node_dict[node_type]["label"]) for node_type in node_types]
        )
        num_nodes_dict = dict(zip(node_types, num_nodes))

        # initialize heterogeneous PyG graph
        data = HeteroData()
        data.hard = hard
        data.device = device
        data.copies = copies

        # ----- node features -----

        for node_type in node_types:
            # keep track of number of nodes
            data[node_type].num_nodes = num_nodes_dict[node_type]

            if num_nodes_dict[node_type]:
                # NOTE: we repeat the tensor to represent different copies of the same graph
                data[node_type].x = (
                    torch.tensor(
                        self.node_dict[node_type]["x"],
                        device=device
                    ).type(dtype=torch.int8 if hard else torch.get_default_dtype())
                    .unsqueeze(1)
                    .repeat(1, copies)
                )

                if node_type in self.node_types[3:]:
                    data[node_type].guards = torch.tensor(
                        self.node_dict[node_type]["guards"],
                        device=device,
                    )
            else:
                # empty data
                data[node_type].x = torch.empty(
                    0,
                    copies,
                    dtype=torch.int8 if data.hard else torch.get_default_dtype(),
                    device=device,
                )

                if node_type in self.node_types[3:]:
                    data[node_type].guards = torch.empty(
                        0,
                        4,
                        device=device,
                    )

        # ----- edge indices and weights -----

        # atom / disj. / conj. -> *
        for src_type in self.node_types[:3]:
            for dst_type in self.node_types:
                edge_type = (src_type, "to", dst_type)

                # existing edges
                if len(self.edge_dict[edge_type]["edge_weight"]):
                    data[edge_type].edge_index = torch.tensor(
                        self.edge_dict[edge_type]["edge_index"],
                        dtype=torch.long,
                        device=device,
                    ).T.contiguous()
                    # NOTE: we repeat the tensor to represent different copies of the same graph
                    data[edge_type].edge_weight = (
                        torch.tensor(
                            self.edge_dict[edge_type]["edge_weight"],
                            device=device,
                        ).type(dtype=torch.int8
                            if hard and dst_type not in self.node_types[3:]
                            else torch.get_default_dtype(),)
                        .unsqueeze(1)
                        .repeat(1, copies)
                    )
                else:
                    # empty data
                    data[edge_type].edge_index = torch.empty(
                        2,
                        0,
                        dtype=torch.long,
                        device=device,
                    )
                    data[edge_type].edge_weight = torch.empty(
                        0,
                        copies,
                        dtype=torch.int8
                        if data.hard and dst_type not in node_types[3:]
                        else torch.get_default_dtype(),
                        device=device,
                    )

        # count / sum / min / max -> *
        for src_type in self.node_types[3:]:
            for dst_type in self.node_types[:3]:
                edge_type = (src_type, "to", dst_type)

                # existing edges
                if len(self.edge_dict[edge_type]["edge_weight"]):
                    data[edge_type].edge_index = torch.tensor(
                        self.edge_dict[edge_type]["edge_index"],
                        dtype=torch.long,
                        device=device,
                    ).T.contiguous()
                    # NOTE: we repeat the tensor to represent different copies of the same graph
                    data[edge_type].edge_weight = (
                        torch.tensor(
                            self.edge_dict[edge_type]["edge_weight"],
                            dtype=torch.int8 if hard else torch.get_default_dtype(),
                            device=device,
                        )
                        .unsqueeze(1)
                        .repeat(1, copies)
                    )
                else:
                    # empty data
                    data[edge_type].edge_index = torch.empty(
                        2,
                        0,
                        dtype=torch.long,
                        device=device,
                    )
                    data[edge_type].edge_weight = torch.empty(
                        0,
                        copies,
                        dtype=torch.int8
                        if data.hard and edge_type[2] not in node_types[3:]
                        else torch.get_default_dtype(),
                        device=device,
                    )

        return data

    def draw(
        self,
        save_as: Optional[str] = None,
        direction: str = "TB",
    ) -> None:
        pgv_graph = self.to_graphviz(direction=direction)

        if save_as is not None:
            pgv_graph.draw(path=save_as, prog="dot")

        # draw without specifying a path (returns bytes object of image)
        img = Image.open(
            io.BytesIO(pgv_graph.draw(prog="dot", format="png")), formats=("PNG",)
        )

        try:
            # check if __IPYTHON__ is defined (a bit of a hack)
            # see https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/7
            __IPYTHON__

            # display using IPython
            from IPython.display import display

            display(img)
        except NameError:
            # show in external window
            img.show()

    def to_graphviz(self, direction: str = "TB") -> pgv.AGraph:
        # TODO: automatically test for self-loops and choose strictness

        # initialize directed graph
        graph = pgv.AGraph(directed=True, rankdir=direction)

        # add nodes
        graph.add_node(
            self.node_dict["disj"]["label"][0],
            style="filled",
            fillcolor="lightgoldenrod",
            shape="circle",
            label=self.__unicode_symbols["true"],
        )
        graph.add_nodes_from(
            self.node_dict["disj"]["label"][1:],
            style="filled",
            fillcolor="gray40",
            shape="circle",
            label=self.__unicode_symbols["disj"],
            fontcolor="white",
        )
        graph.add_nodes_from(
            self.node_dict["atom"]["label"],
            style="filled",
            fillcolor="darkslategray3",
            shape="oval",
        )
        graph.add_node(
            self.node_dict["conj"]["label"][0],
            shape="circle",
            label=self.__unicode_symbols["false"],
        )
        graph.add_nodes_from(
            self.node_dict["conj"]["label"][1:],
            shape="circle",
            label=self.__unicode_symbols["conj"],
        )

        # map encoded relation operators to a symbol
        symbol_dict = {
            0: "=",
            1: self.__unicode_symbols["neq"],
            2: "<",
            3: ">",
            4: self.__unicode_symbols["leq"],
            5: self.__unicode_symbols["geq"],
        }

        for node_type in ("count", "sum", "min", "max"):
            for node_key, guards in zip(
                self.node_dict[node_type]["label"], self.node_dict[node_type]["guards"]
            ):
                # TODO: clearner way?
                # NOTE: not a high priority as plotting does not need to be performant
                label = f"\#{node_type}"

                if guards[0] != -1:
                    bound = int(guards[1]) if isfinite(guards[1]) else guards[1]
                    label = f"{bound}{symbol_dict[guards[0]]}" + label
                if guards[2] != -1:
                    bound = int(guards[3]) if isfinite(guards[3]) else guards[3]
                    label = label + f"{symbol_dict[guards[2]]}{bound}"

                graph.add_node(
                    node_key,
                    shape="rectangle",
                    label=label,
                )

        choice_edges = []

        for _, edges in itertools.chain(
            self.choice_edges.items(), self.npp_edges.items()
        ):
            for _, edge_type, edge_id in edges:
                src_id, dst_id = self.edge_dict[edge_type]["edge_index"][edge_id]
                src_key = self.node_dict[edge_type[0]]["label"][src_id]
                dst_key = self.node_dict[edge_type[-1]]["label"][dst_id]

                choice_edges.append((src_key, dst_key))

        # add edges
        for src_type, dst_type in itertools.product(
            ("atom", "disj", "conj", "count", "sum", "min", "max"),
            ("atom", "disj", "conj", "count", "sum", "min", "max"),
        ):
            edge_type = (src_type, "to", dst_type)

            for (src, dst), w in zip(
                self.edge_dict[edge_type]["edge_index"],
                self.edge_dict[edge_type]["edge_weight"],
            ):
                if dst_type in ("count", "sum", "min", "max") or w == 1:
                    color = "black"
                elif w == 0:
                    color = "gray65"
                else:
                    color = "orangered"

                if isfinite(w):
                    w = int(w)

                src_key = self.node_dict[src_type]["label"][src]
                dst_key = self.node_dict[dst_type]["label"][dst]

                graph.add_edge(
                    src_key,
                    dst_key,
                    color=color,
                    style="dashed" if (src_key, dst_key) in choice_edges else "",
                    label=str(w) if dst_type in ("sum", "min", "max") else "",
                )

        return graph
