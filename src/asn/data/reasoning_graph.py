import io
import itertools
from collections import defaultdict
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

from .expression import Conjunction, Disjunction, MultiConstraint


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
            "true": "\u22a4",  # ⊤
            "false": "\u22a5",  # ⊥
            "disj": "\u2228",  # ∨
            "conj": "\u2227",  # ∧
            "neq": "\u2260",  # ≠
            "leq": "\u2264",  # ≤
            "geq": "\u2265",  # ≥
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

        # Encode a statement of form:
        #
        #        H :- B.
        #
        # #################### process body ####################
        #
        #        B = b1, ..., bN, not bN+1, ..., not bM
        #
        # Connect all body literals to a conjunction representing
        # the body 'B' with positive/negative edges
        #
        #       ┌──┐┌───┐┌──┐┌────┐┌───┐┌──┐
        #       │b1││...││bN││bN+1││...││bM│
        #       └┬─┘└─┬─┘└┬─┘└─┬──┘└─┬─┘└┬─┘
        #        +    +   +    -     -   -
        #        │    │   │    │     │   │
        #       ┌▽────▽───▽────▽─────▽───▽┐
        #       │           B (∧)         │
        #       └─────────────────────────┘
        #
        # In case of facts (M=0), we treat it as
        #       B = Truth
        #
        # NOTE: if the body consists of a single literal 'b',
        # then we can skip encoding the conjunction and just
        # use 'b' directly as representative for 'B'

        body_literals = []

        for literal in statement.body:
            if isinstance(literal, BuiltinLiteral):
                if not literal.eval():
                    # false built-in literal (i.e., body never satisfied)
                    # no need to process rule
                    return
            else:
                # encode literal (if not already) and keep track of it
                self.encode_literal(literal, certain_atoms)
                body_literals.append(literal)

        # in case of facts, treat 'truth' as only body literal
        if not body_literals:
            body_literals.append(self.true_const)

        # create conjunction of body literals
        if len(body_literals) > 1:
            body_signature = Conjunction(*body_literals)
            self.encode_conjunction(*body_literals)
        # use single body literal directly
        else:
            body_signature = body_literals[0]

        # #################### process head ####################
        #
        # Different cases:
        #   Constraint:         H empty (K=0, treat as H = Falsity)
        #   Normal rule:        H = a
        #   Disjunctive rule:   H = a1 | ... | aK
        #   Choice rule:        H = g1 © {a1:li1,...,lj1 ; ... ; aK:liK,...,ljK} ® g2
        #   NPP rule:           H = #NPP{a, [o1,...,oK]}
        #
        # The 'a's in the head are the consequents;
        # these are the atoms that can follow from the rule
        #
        #       ┌──┐     ┌──┐
        #       │c1│ ... │cK│
        #       └┬─┘     └┬─┘
        #        ?        ?
        #        │        │
        #       ┌▽─┐     ┌▽─┐
        #       │a1│ ... │aK│
        #       └──┘     └──┘
        #
        # Here 'ci' represents that the conditions for 'ai' is fullfilled
        #
        # In the case of constraints & normal/disjunctive/NPP rules:
        #   Conditions are simply the rule body, i.e.:
        #
        #       ci=B for all i=1,...,K
        #
        # In the case of choice rules, some consequents may have (possibly multiple)
        # extra conditions

        # ----- 2. encode conditions and connect them to consequents -----

        # normal/disjunctive rule,
        if isinstance(statement, (NormalRule, DisjunctiveRule)):
            # no extra conditions
            conditions = [(atom, Conjunction()) for atom in statement.head]
        #  constraint
        elif isinstance(statement, Constraint):
            # no extra conditions
            conditions = [(self.false_const, Conjunction())]
        # choice/NPP rule
        else:
            choice = (
                statement.head
                if isinstance(statement, ChoiceRule)
                else statement.npp.as_choice()
            )
            conditions = [(elem.atom, elem.literals) for elem in choice.elements]

        # encode conditions
        conditional_signatures, conditions_dict = self.encode_conditions(
            conditions,
            external_condition=body_signature,
            certain_atoms=certain_atoms,
            return_conditions_dict=True,
        )

        # dictionary to store the edges corresponding to a choice/disjunction
        # NOTE: used later if statement actually non-deterministic
        choice_edges = list()

        # connect conditions to consequent literals
        for consequent_literal, condition_signature in conditional_signatures:

            # encode consequent literal (if not already)
            # the condition is satisfyable to begin with
            if consequent_literal is not self.false_const:
                self.encode_literal(consequent_literal, certain_atoms)

            choice_edges.append(
                (
                    consequent_literal,
                    *self.add_edge(
                        abs(condition_signature),
                        consequent_literal,
                        # NOTE: -1 is only relevant if the condition (extra + body) consists of a single
                        # literal, in which case we use the literal directly and the sign becomes relevant
                        edge_weight=1.0 if not condition_signature.naf else -1.0,
                    ),
                )
            )

        # track edges representing choices
        # TODO: clean up (messy) !!!
        # TODO: what for disjunctive rules?
        match statement:
            case DisjunctiveRule() | ChoiceRule():
                # TODO: necessary ???
                self.choices.add(statement)
                # TODO: best way to store choices?
                self.choice_edges[statement] = choice_edges
            case NPPRule():
                # TODO: necessary ???
                self.npps.add(statement)
                # TODO: best way to store choices?
                self.npp_edges[statement] = choice_edges

        # ----- 3. (optional) encode choice constraint -----
        #
        #       ┌────────┐     ┌───────┐ ┌───┐
        #       │   c1   │ ... │   cK  │ │ B │
        #       └─┬────┬─┘     └─┬────┬┘ └─┬─┘
        #         ?    │         ?    │    │
        #         │    │         │    │    │
        #       ┌─▽──┐ │       ┌─▽──┐ │    │
        #       │ a1 │ │   ... │ aK │ │    │
        #       └─┬──┘ │       └─┬──┘ │    │
        #       ┌─▽────▽─┐     ┌─▽────▽─┐  │
        #       │ e1 (∧) │ ... | e2 (∧) │  │
        #       └───┬────┘     └───┬────┘  │
        #         ┌─▽──────────────▽─┐     │
        #         │ g1 © #count ® g2 │     │
        #         └────────┬─────────┘     │
        #                  -               +
        #                  │               │
        #                 ┌▽───────────────▽┐
        #                 │   Bconstr (∧)   |
        #                 └────────┬────────┘
        #                        ┌─▽─┐
        #                        │ ⊥ │
        #                        └───┘
        #
        # NOTE: if 'ai' has no extra condition, then 'ei' can be skipped
        # and 'ai' be used directly as input to the aggregate node
        # Furthermore, if 'B' is Truth, then the conjunction 'Bconstr' can
        # be skipped and the aggregate node directly connected to Falsity

        # TODO: encode constraints for disjunctive/NPP rule?
        # -> currently implicitely handled if choice edges are set correctly
        if isinstance(statement, ChoiceRule):
            count_member_signatures = []
            aggr_elements = []

            for consequent_literal, condition_signature in conditional_signatures:
                # NOTE: if the condition is simply the body or truth, we can skip
                # encoding conjunction (ei) and just use consequent literal directly
                if condition_signature in (self.true_const, body_signature):
                    count_member_signatures.append(consequent_literal)
                else:
                    # create conjunction of condition and consequent (ei)
                    self.encode_conjunction(consequent_literal, condition_signature)
                    count_member_signatures.append(
                        Conjunction(consequent_literal, condition_signature)
                    )

                # create aggregate elements corresponding to the conditions
                for condition in conditions_dict[consequent_literal]:
                    aggr_elements.append(
                        AggrElement(
                            # NOTE: consequent literal is also a valid term
                            TermTuple(consequent_literal),
                            (
                                LiteralCollection(*condition, *body_literals)
                                if body_literals[0] is not [self.true_const]
                                else LiteralCollection(*condition)
                            ),
                        )
                    )

            # signature of aggregate literal
            aggr_signature = AggrLiteral(
                AggrCount(), tuple(aggr_elements), guards=statement.choice.guards
            )

            # create new aggregate node
            self.add_node(
                aggr_signature,
                "count",
                label=f"#count_{{{len(self.node_dict['count']['x'])-1}}}",
                guards=tuple(self.encode_guards(statement.choice.guards)),
            )
            # NOTE: we set NaF to true here, so that it is not part of
            # the signature when adding the node above
            # TODO: cleaner way?
            aggr_signature.naf = True

            # connect elements to aggregate nodes
            for member_signature in count_member_signatures:
                self.add_edge(member_signature, aggr_signature)

            if body_signature is not self.true_const:
                # create conjunction (Bconstr) of aggregate and statement body
                self.encode_conjunction(
                    aggr_signature,
                    body_signature,
                    # signs=[-1, 1],
                )
                constr_body_signature = Conjunction(aggr_signature, body_signature)
            else:
                constr_body_signature = aggr_signature

            # connect to sink node (falsity)
            self.add_edge(
                constr_body_signature,
                self.false_const,
                edge_weight=-1.0 if constr_body_signature.naf else 1.0,
            )

    def __encode_junction(
        self,
        junction_type: str,
        keys: Iterable,
        signs: Optional[int | Iterable[int]] = None,
    ) -> None:
        if junction_type not in ("conj", "disj"):
            raise ValueError(
                f"'junction_type' must be one of 'conj', 'disj', but was {junction_type}."
            )

        abs_keys = []

        # get non-default-negated keys
        for k in keys:
            abs_k = abs(k)

            # ensure that expression is already encoded in the graph
            if abs_k not in self.node_id_dict:
                raise ValueError(
                    f"Encoding conjunction requires all keys to be encoded, but '{str(k)}' could not be found."
                )

            abs_keys.append(abs_k)

        num_keys = len(abs_keys)

        # automatically infer signs
        if signs is None:
            signs = [-1 if k.naf else 1 for k in keys]
            num_signs = num_keys
        # broadcast sign
        elif isinstance(signs, int):
            signs = [signs] * num_keys
            num_signs = num_keys
        # check number of specified signs
        else:
            try:
                num_signs = len(signs)  # type: ignore
            except TypeError:
                num_signs = sum(1 for _ in signs)

        # make sure that the number of keys and signs matches
        if num_keys != num_signs:
            raise ValueError("Specified number of signs does not match number of keys.")

        # singleton or empty junction (no need to encode as new node)
        if num_keys < 2:
            return

        Junction = Conjunction if junction_type == "conj" else Disjunction

        # create junction node
        junction_key = Junction(*keys)

        if junction_key not in self.node_id_dict:
            self.add_node(
                junction_key,
                junction_type,
                f"{self.__unicode_symbols[junction_type]}_{{{len(self.node_dict['conj']['x'])-1}}}",
            )

            # connect members to conjunction node
            for k, s in zip(abs_keys, signs, strict=True):
                self.add_edge(
                    k,
                    junction_key,
                    edge_weight=torch.tensor(s),
                )

    def encode_conjunction(
        self, *keys: Expr, signs: Optional[int | Iterable[int]] = None
    ) -> None:
        self.__encode_junction("conj", keys, signs)

    def encode_disjunction(
        self, *keys: Expr, signs: Optional[int | Iterable[int]] = None
    ) -> None:
        self.__encode_junction("disj", keys, signs)

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
            aggr: AggrLiteral = abs(literal)  # type: ignore

            # positive or negative aggregate
            self.encode_aggregate(aggr)
        elif isinstance(literal, (TrueConstant, FalseConstant)):
            return 1
        elif isinstance(literal, PredLiteral):
            atom: PredLiteral = abs(literal)  # type: ignore

            # initialize probability with 1.0 if atom is certain (i.e., fact)
            p = float(atom in certain_atoms)
            p = torch.tensor(p)

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
                # since 'False' already registed, we can safely assume that all new
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
                    self.encode_statement(
                        Constraint(atom, neg_atom)
                    )  # should easily take care of things

        return -1 if literal.naf else 1

    def encode_conditions(
        self,
        conditions: Iterable[Tuple[Expr, LiteralCollection]],
        external_condition: Optional[Expr] = None,
        certain_atoms: Optional[Set[PredLiteral]] = None,
        return_conditions_dict: bool = False,
    ) -> Union[
        List[Tuple[Expr, Expr]], Tuple[List[Tuple[Expr, Expr]], Dict[Expr, Expr]]
    ]:

        # A conditional may have multiple conditions that can satisfy it,
        # where any condition is sufficient
        #
        #       ┌───┐┌───┐┌───┐        ┌───┐┌───┐┌───┐
        #       │li1││...││lj1│        │liJ││...││ljJ│
        #       └─┬─┘└─┬─┘└─┬─┘        └─┬─┘└─┬─┘└─┬─┘
        #         ±    ±    ±            ±    ±    ±
        #         │    │    │            │    │    │
        #       ┌─▽────▽────▽──┐       ┌─▽────▽────▽──┐  ┌──────────┐
        #       │  cond1 (∧)   │  ...  │  cond2 (∧)   │  │ ext_cond │
        #       └──────┬───────┘       └──────┬───────┘  └────┬─────┘
        #          ┌───▽──────────────────────▽─────┐         │
        #          │           anycond (∨)          │         │
        #          └───────────────┬────────────────┘         │
        #                        ┌─▽──────────────────────────▽─┐
        #                        │             c (∧)            │
        #                        └───────────────┬──────────────┘
        #                                        ?
        #                                        │
        #                                 ┌──────▽──────┐
        #                                 │ conditional │
        #                                 └─────────────┘
        #
        # NOTE: if 'conditional' only has one possible condition, we can
        # skip the disjunction and directly treat 'cond1' as 'c'.
        # Additionally, if there are no extra conditions, then we can
        # skip the conjunction of 'c' and use 'ext_cond' directly!
        #
        # Since consequents may have multiple conditions, we first filter for minimality
        # since if 'ci' is fullfilled, then all 'cj' that are subsets of 'ci' are also fullfilled,
        # so it is sufficient to regard inclusion-minimal conditions

        if external_condition is None:
            external_condition = self.true_const

        if certain_atoms is None:
            certain_atoms = set()

        # ---------- group elements ----------

        # dictionary mapping a term tuple to possible conditions satisfying it
        # (multiple possible; only one needs to hold)
        conditions_dict = defaultdict(list)

        for conditional, condition in conditions:
            # keep track of predicate_literals only (no built-in literals)
            predicate_literals = []

            for literal in condition:
                if isinstance(literal, BuiltinLiteral):
                    if not literal.eval():
                        # false built-in literal (i.e., condition unsatisfiable)
                        # no need to encode this element
                        break
                else:
                    # keep track of literal
                    predicate_literals.append(literal)
            # run if loop did not break early (i.e., condition is satisfiable)
            else:
                # keep track of condition for tuple
                # NOTE: conjunction may be empty if there are no predicate literals and
                # no false built-in literals (in which case the element is unconditional)
                conditions_dict[conditional].append(Conjunction(*predicate_literals))

        # ---------- process tuples and conditions ----------

        # get minimal conditions
        # (supersets irrelevant if a subset already satisfies condition)
        conditions_dict = {
            # NOTE: we check for empty condition first to avoid having to unnecessarily filter
            consequent_literal: (
                (Conjunction(),)
                if Conjunction() in condition_candidates
                else get_minimal_collections(*condition_candidates)
            )
            for consequent_literal, condition_candidates in conditions_dict.items()
        }

        conditional_signatures = []

        # process consequents and conditions
        for consequent_literal, minimal_conditions in conditions_dict.items():
            if len(minimal_conditions) == 1:
                condition = minimal_conditions[0]

                # encode all literals in conditions (if not already)
                for literal in condition:
                    self.encode_literal(literal, certain_atoms)  # type: ignore

                # directly join 'ext_cond' here
                # NOTE: can be skipped if 'ext_cond' is 'Truth' and condition is non-empty,
                # in that case adding 'ext_cond' is unnecessary;
                # however for empty conditions it is needed
                if not (external_condition is self.true_const and len(condition) > 0):
                    condition = [*condition, external_condition]

                # encode conjunction representing this condition ('condi')
                self.encode_conjunction(*condition)
                full_condition_signature = (
                    Conjunction(*condition) if len(condition) > 1 else condition[0]
                )
            elif len(minimal_conditions) > 1:
                condition_signatures = []

                # encode consequent conditions
                for condition in minimal_conditions:
                    # encode all literals in conditions (if not already)
                    for literal in condition:
                        self.encode_literal(literal, certain_atoms)  # type: ignore

                    # encode conjunction representing this condition ('condi')
                    self.encode_conjunction(*condition)
                    conj_signature = (
                        Conjunction(*condition) if len(condition) > 1 else condition[0]
                    )
                    condition_signatures.append(conj_signature)

                # combine conditions using disjunction ('anycond')
                self.encode_disjunction(*condition_signatures)
                full_condition_signature = Disjunction(*condition_signatures)

                # combine together with external condition in conjunction (if it is non-empty)
                if external_condition is not self.true_const:
                    self.encode_conjunction(
                        full_condition_signature, external_condition
                    )
                    full_condition_signature = Conjunction(
                        full_condition_signature, external_condition
                    )

            conditional_signatures.append(
                (consequent_literal, full_condition_signature)
            )

        return (
            conditional_signatures
            if not return_conditions_dict
            else (conditional_signatures, conditions_dict)
        )

    def encode_aggregate(
        self,
        aggr: AggrLiteral,
        certain_atoms: Optional[Set[PredLiteral]] = None,
    ) -> None:
        """Encodes an aggregate in the graph.

        Args:
            aggr: `AggrLiteral` instance. Is expected to be non-default-negated.

        Raises:
            TODO
        """

        # Encode an aggregate literal of form:
        #   g1 © #aggr{t1:li1,...,lj1 ; ... ; tK:liK,...,ljK} ® g2
        #
        #        ┌────┐     ┌────┐
        #        │ e1 │ ... │ eK │
        #        └─┬──┘     └─┬──┘
        #        w(t1)      w(tK)
        #          │          │
        #       ┌──▽──────────▽───┐
        #       │ g1 © #aggr ® g2 │
        #       └─────────────────┘

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

        condition_signatures = self.encode_conditions(
            [(elem.terms, elem.literals) for elem in aggr.elements],
            certain_atoms=certain_atoms,
        )

        # multiple distinct term tuples may have the same condition
        # we can therefore aggregate these together into a single edge

        # NOTE: some conditionals may have the same condition (signature),
        # so to not have redundant extra edges (which would also break 'draw'),
        # we keep track of term tuples for each condition signature and
        # create aggregated edges
        tuple_cond_dict = defaultdict(list)

        for tup, condition_signature in condition_signatures:
            tuple_cond_dict[condition_signature].append(tup)

        for tuple_condition_signature, term_tuples in tuple_cond_dict.items():
            self.add_edge(
                tuple_condition_signature,
                aggr,
                edge_weight=float(aggr.func.eval(set(term_tuples)).eval()),
            )

    def encode_query(
        self,
        query: Union[Constraint, Iterable[Constraint]],
        certain_atoms: Optional[Set[PredLiteral]] = None,
    ) -> int:
        """Adds a query to the reasoning graph.

        Args:
            query: `Constraint` instance.
            certain_atoms: optional set of atoms (`PredLiteral` instances) whose nodes
                are initialized to `True`. Can be used to reduce number of iterations.

        Raises:
            TODO
        """
        if not isinstance(query, Constraint):
            query = MultiConstraint(*query)

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
        **attrs: Any,
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
                    torch.tensor(self.node_dict[node_type]["x"], device=device)
                    .type(dtype=torch.int8 if hard else torch.get_default_dtype())
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
                        )
                        .type(
                            dtype=(
                                torch.int8
                                if hard and dst_type not in self.node_types[3:]
                                else torch.get_default_dtype()
                            ),
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
                        dtype=(
                            torch.int8
                            if data.hard and dst_type not in node_types[3:]
                            else torch.get_default_dtype()
                        ),
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
                        dtype=(
                            torch.int8
                            if data.hard and edge_type[2] not in node_types[3:]
                            else torch.get_default_dtype()
                        ),
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

        # ----- add nodes -----

        # global sink node
        graph.add_node(
            self.node_dict["disj"]["label"][0],
            style="filled",
            fillcolor="lightgoldenrod",
            shape="circle",
            label=self.__unicode_symbols["false"],
        )

        # map disj. node ID to its query sink ID
        query_sink_dict = {
            node_id: query_sink_id
            for query_sink_id, node_id in enumerate(self.query_sinks)
        }

        for disj_id, disj in enumerate(self.node_dict["disj"]["label"][1:], start=1):
            # query sink
            if disj_id in query_sink_dict:
                fillcolor = "lightgoldenrod"
                shape = "oval"
                label = rf"{self.__unicode_symbols['false']}{query_sink_dict[disj_id]}"
                fontcolor = "black"
            # "regular" disjunction
            else:
                fillcolor = "gray40"
                shape = "circle"
                label = self.__unicode_symbols["disj"]
                fontcolor = "white"

            graph.add_node(
                disj,
                style="filled",
                fillcolor=fillcolor,
                shape=shape,
                label=label,
                fontcolor=fontcolor,
            )

        # atoms
        graph.add_nodes_from(
            self.node_dict["atom"]["label"],
            style="filled",
            fillcolor="darkslategray3",
            shape="oval",
        )

        # source node
        graph.add_node(
            self.node_dict["conj"]["label"][0],
            shape="circle",
            label=self.__unicode_symbols["true"],
        )

        # conjunctions
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

        # aggregates
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

        # ----- add edges -----
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
                    label=str(w) if dst_type in ("count", "sum", "min", "max") else "",
                )

        return graph
