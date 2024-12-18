import unittest
from collections import namedtuple
from typing import Any, Dict, List

from ground_slash.program import (
    NPP,
    AggrCount,
    AggrElement,
    AggrLiteral,
    AggrMax,
    AggrMin,
    AggrSum,
    Choice,
    ChoiceElement,
    FalseConstant,
    Functional,
    Guard,
    LiteralCollection,
    Naf,
    Number,
    PredLiteral,
    Program,
    RelOp,
    SymbolicConstant,
    TrueConstant,
)

from asn.data.reasoning_graph import ReasoningGraph

# can then also conveniently add defaults !!! (really nice to generate/manipulate rg???)
# TODO: id ???

AtomNode = namedtuple("atom", ["label", "x", "aux"])
ConjNode = namedtuple("conj", ["label", "x"])

CountNode = namedtuple("count", ["label", "x", "guards"])
SumNode = namedtuple("sum", ["label", "x", "guards"])
MinNode = namedtuple("min", ["label", "x", "guards"])
MaxNode = namedtuple("max", ["label", "x", "guards"])

# atom -> conj
Atom2ConjEdge = namedtuple("atom_in_conj", ["edge_index", "edge_weight"])
# conj -> atom
Conj2AtomEdge = namedtuple("conj_defines_atom", ["edge_index", "edge_weight"])
# atom -> aggr
Atom2CountEdge = namedtuple("atom_in_count", ["edge_index", "edge_weight"])
Atom2SumEdge = namedtuple("atom_in_sum", ["edge_index", "edge_weight"])
Atom2MinEdge = namedtuple("atom_in_min", ["edge_index", "edge_weight"])
Atom2MaxEdge = namedtuple("atom_in_max", ["edge_index", "edge_weight"])
# aggr -> atom
Count2AtomEdge = namedtuple("count_defines_atom", ["edge_index", "edge_weight"])
Sum2AtomEdge = namedtuple("sum_defines_atom", ["edge_index", "edge_weight"])
Min2AtomEdge = namedtuple("min_defines_atom", ["edge_index", "edge_weight"])
Max2AtomEdge = namedtuple("max_defines_atom", ["edge_index", "edge_weight"])


def zip_nodes(node_dict: Dict[str, Dict[str, List[Any]]]) -> Dict:
    return {
        node.__name__: list(
            node(*tup)
            for tup in zip(*[node_dict[node.__name__][attr] for attr in node._fields])
        )
        for node in (AtomNode, ConjNode, CountNode, SumNode, MinNode, MaxNode)
    }


def zip_edges(edge_dict: Dict[str, Dict[str, List[Any]]]) -> Dict:
    return {
        tuple(edge.__name__.split("_")): list(
            edge(*tup)
            for tup in zip(
                *[
                    edge_dict[tuple(edge.__name__.split("_"))][attr]
                    for attr in edge._fields
                ]
            )
        )
        for edge in (
            Atom2ConjEdge,
            Conj2AtomEdge,
            Atom2CountEdge,
            Atom2SumEdge,
            Atom2MinEdge,
            Atom2MaxEdge,
            Count2AtomEdge,
            Sum2AtomEdge,
            Min2AtomEdge,
            Max2AtomEdge,
        )
    }


class TestRegionGraph(unittest.TestCase):
    def test_init(self):
        # empty program
        prog = Program(())

        # create reasoning graph
        # (empty) except for basic initialization
        rg = ReasoningGraph(prog)

        # basic attributes
        self.assertEqual(rg.aux_counter, 0)
        self.assertEqual(rg.atom_in_conj, ("atom", "in", "conj"))
        self.assertEqual(rg.conj_defines_atom, ("conj", "defines", "atom"))
        self.assertEqual(rg.atom_in_count, ("atom", "in", "count"))
        self.assertEqual(rg.atom_in_sum, ("atom", "in", "sum"))
        self.assertEqual(rg.atom_in_min, ("atom", "in", "min"))
        self.assertEqual(rg.atom_in_max, ("atom", "in", "max"))
        self.assertEqual(rg.count_defines_atom, ("count", "defines", "atom"))
        self.assertEqual(rg.sum_defines_atom, ("sum", "defines", "atom"))
        self.assertEqual(rg.min_defines_atom, ("min", "defines", "atom"))
        self.assertEqual(rg.max_defines_atom, ("max", "defines", "atom"))
        self.assertEqual(
            rg.aggr_map,
            {
                AggrCount(): ("count", r"\#"),
                AggrSum(): ("sum", r"\Sigma"),
                AggrMin(): ("min", "MIN"),
                AggrMax(): ("max", "MAX"),
            },
        )
        self.assertEqual(rg.true_const, TrueConstant())
        self.assertEqual(rg.false_const, FalseConstant())

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            set(nodes["atom"]),
            {AtomNode(r"$\top$", 1.0, True), AtomNode(r"$\bot$", 0.0, True)},
        )
        # conjunction nodes
        self.assertFalse(nodes["conj"])
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertFalse(edges[("atom", "in", "conj")])
        # conj -> atom
        self.assertFalse(edges[("conj", "defines", "atom")])
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(len(rg.conj_ids), 0)
        self.assertEqual(len(rg.aggr_ids), 0)

        # choice tracking
        self.assertEqual(len(rg.choices), 0)
        self.assertEqual(len(rg.choice_edges), 0)

    def test_normal_fact(self):
        prog = Program.from_string(
            r"""

        a.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            set(nodes["atom"]),
            {
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 1.0, False),
            },
        )
        # conjunction nodes
        self.assertEqual(set(nodes["conj"]), {ConjNode(r"$\wedge_0$", 0.0)})
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]), {Atom2ConjEdge((0, 0), 1.0)}
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]), {Conj2AtomEdge((0, 2), 1.0)}
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(rg.conj_ids[LiteralCollection(TrueConstant())], 0)
        self.assertEqual(len(rg.aggr_ids), 0)

        # choice tracking
        self.assertEqual(len(rg.choices), 0)
        self.assertEqual(len(rg.choice_edges), 0)

    def test_normal_facts(self):
        prog = Program.from_string(
            r"""

        a.
        b.

        """
        )
        # conj. should be reused for both facts

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            set(nodes["atom"]),
            {
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 1.0, False),
                AtomNode("b", 1.0, False),
            },
        )
        # conjunction nodes
        self.assertEqual(set(nodes["conj"]), {ConjNode(r"$\wedge_0$", 0.0)})
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]), {Atom2ConjEdge((0, 0), 1.0)}
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {Conj2AtomEdge((0, 2), 1.0), Conj2AtomEdge((0, 3), 1.0)},
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(rg.conj_ids[LiteralCollection(TrueConstant())], 0)
        self.assertEqual(len(rg.aggr_ids), 0)

        # choice tracking
        self.assertEqual(len(rg.choices), 0)
        self.assertEqual(len(rg.choice_edges), 0)

    def test_normal_rule(self):
        prog = Program.from_string(
            r"""

        a :- b, not c.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode("b", 0.0, False),
                AtomNode("c", 0.0, False),
            ],
        )
        # conjunction nodes
        self.assertEqual(nodes["conj"], [ConjNode(r"$\wedge_0$", 0.0)])
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {Atom2ConjEdge((3, 0), 1.0), Atom2ConjEdge((4, 0), -1.0)},
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {Conj2AtomEdge((0, 2), 1.0)},
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                PredLiteral("b"): 3,
                PredLiteral("c"): 4,
            },
        )
        self.assertEqual(
            rg.conj_ids, {LiteralCollection(PredLiteral("b"), Naf(PredLiteral("c"))): 0}
        )
        self.assertEqual(len(rg.aggr_ids), 0)

        # choice tracking
        self.assertEqual(len(rg.choices), 0)
        self.assertEqual(len(rg.choice_edges), 0)

    def test_disjunctive_fact(self):
        prog = Program.from_string(
            r"""

        a | b.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            set(nodes["atom"]),
            {
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode("b", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
            },
        )
        # conjunction nodes
        self.assertEqual(
            set(nodes["conj"]),
            {ConjNode(r"$\wedge_0$", 0.0), ConjNode(r"$\wedge_1$", 0.0)},
        )
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                Atom2ConjEdge((0, 0), 1.0),
                Atom2ConjEdge((2, 1), -1.0),
                Atom2ConjEdge((3, 1), -1.0),
                Atom2ConjEdge((4, 1), 1.0),
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                Conj2AtomEdge((0, 2), 1.0),
                Conj2AtomEdge((0, 3), 1.0),
                Conj2AtomEdge((0, 4), 1.0),
                Conj2AtomEdge((1, 1), 1.0),
            },
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(rg.conj_ids[LiteralCollection(TrueConstant())], 0)
        self.assertEqual(len(rg.aggr_ids), 0)

        # choice tracking
        self.assertEqual(
            rg.choices, {LiteralCollection(PredLiteral("a"), PredLiteral("b"))}
        )
        self.assertEqual(
            rg.choice_edges,
            {
                LiteralCollection(PredLiteral("a"), PredLiteral("b")): [
                    0,  # (0, 2)
                    1,  # (0, 3)
                ]
                # '$\wedge_0$' to 'a', 'b'
            },
        )

    def test_disjunctive_rule(self):
        prog = Program.from_string(
            r"""

        a | b :- c, not d.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode("b", 0.0, False),
                AtomNode("c", 0.0, False),
                AtomNode("d", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            nodes["conj"], [ConjNode(r"$\wedge_0$", 0.0), ConjNode(r"$\wedge_1$", 0.0)]
        )
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # body conjunction
                Atom2ConjEdge((4, 0), 1.0),
                Atom2ConjEdge((5, 0), -1.0),
                # disjunction constraint
                Atom2ConjEdge((2, 1), -1.0),
                Atom2ConjEdge((3, 1), -1.0),
                Atom2ConjEdge((6, 1), 1.0),
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # body to head
                Conj2AtomEdge((0, 2), 1.0),
                Conj2AtomEdge((0, 3), 1.0),
                # body to aux. atom
                Conj2AtomEdge((0, 6), 1.0),
                # disjunction constraint
                Conj2AtomEdge((1, 1), 1.0),
            },
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                PredLiteral("b"): 3,
                PredLiteral("c"): 4,
                PredLiteral("d"): 5,
                LiteralCollection(PredLiteral("c"), Naf(PredLiteral("d"))): 6,
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(PredLiteral("c"), Naf(PredLiteral("d"))): 0,
                LiteralCollection(
                    Naf(PredLiteral("a")),
                    Naf(PredLiteral("b")),
                    PredLiteral("c"),
                    Naf(PredLiteral("d")),
                ): 1,
            },
        )
        self.assertEqual(len(rg.aggr_ids), 0)

        # choice tracking
        self.assertEqual(
            rg.choices, {LiteralCollection(PredLiteral("a"), PredLiteral("b"))}
        )
        self.assertEqual(
            rg.choice_edges,
            {LiteralCollection(PredLiteral("a"), PredLiteral("b")): [0, 1]},
        )

    def test_constraint(self):
        prog = Program.from_string(
            r"""

        :- a, not b.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode("b", 0.0, False),
            ],
        )
        # conjunction nodes
        self.assertEqual(nodes["conj"], [ConjNode(r"$\wedge_0$", 0.0)])
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {Atom2ConjEdge((2, 0), 1.0), Atom2ConjEdge((3, 0), -1.0)},
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {Conj2AtomEdge((0, 1), 1.0)},
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                PredLiteral("b"): 3,
            },
        )
        self.assertEqual(
            rg.conj_ids, {LiteralCollection(PredLiteral("a"), Naf(PredLiteral("b"))): 0}
        )
        self.assertEqual(len(rg.aggr_ids), 0)

        # choice tracking
        self.assertEqual(len(rg.choices), 0)
        self.assertEqual(len(rg.choice_edges), 0)

    def test_empty_constraint(self):
        # TODO: not supported by ground_slash yet
        pass

    def test_choice_fact(self):
        prog = Program.from_string(
            r"""

        {a;b:d;b:e,not f;b:not f;c}.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode("b", 0.0, False),
                AtomNode("c", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode(r"$\vee_1$", 0.0, True),
                AtomNode("d", 0.0, False),
                AtomNode("e", 0.0, False),
                AtomNode("f", 0.0, False),
                AtomNode(r"$\vee_2$", 0.0, True),
                AtomNode(r"$\vee_3$", 0.0, True),
                AtomNode(r"$\vee_4$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            nodes["conj"],
            [
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
                ConjNode(r"$\wedge_3$", 0.0),
                ConjNode(r"$\wedge_4$", 0.0),
                ConjNode(r"$\wedge_5$", 0.0),
                ConjNode(r"$\wedge_6$", 0.0),
            ],
        )
        # aggregate nodes
        self.assertEqual(set(nodes["count"]), {CountNode(r"$\#_0$", 0.0, (-1,) * 4)})
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # body to conj.
                Atom2ConjEdge((0, 0), 1.0),  # 'True' to 'conj0' ('True')
                # global constraint for choice (given sat. body)
                Atom2ConjEdge((5, 1), 1.0),  # 'disj0' (body) to 'conj1'
                Atom2ConjEdge((6, 1), -1.0),  # 'disj1' (aggr) to 'conj1'
                # cond. constraint for 'a'
                Atom2ConjEdge((5, 2), 1.0),  # 'disj0' (body) to 'conj4'
                Atom2ConjEdge((2, 2), 1.0),  # 'a' to 'conj4'
                Atom2ConjEdge((10, 2), -1.0),  # 'disj5' ('a' condition) to 'conj4'
                # cond. constraint for 'b'
                Atom2ConjEdge((5, 5), 1.0),  # 'disj0' (body) o 'conj5'
                Atom2ConjEdge((3, 5), 1.0),  # 'b' to 'conj5'
                Atom2ConjEdge((11, 5), -1.0),  # 'disj6' ('b' condition) to 'conj5'
                # cond. candidates for 'b' choice
                Atom2ConjEdge((7, 3), 1.0),  # 'd' to 'conj2'
                Atom2ConjEdge((9, 4), -1.0),  # 'not f' to 'conj3'
                # cond. constraint for 'c'
                Atom2ConjEdge((5, 6), 1.0),  # 'disj0' (body) to 'conj6'
                Atom2ConjEdge((4, 6), 1.0),  # 'c' to 'conj6'
                Atom2ConjEdge((12, 6), -1.0),  # 'disj7' ('c' condition) to 'conj6'
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # conj. to head
                Conj2AtomEdge((0, 2), 1.0),  # 'conj0' ('True') to 'a'
                Conj2AtomEdge((0, 3), 1.0),  # 'conj0' ('True') to 'b'
                Conj2AtomEdge((0, 4), 1.0),  # 'conj0' ('True') to 'c'
                # conj. to aux. body atom
                Conj2AtomEdge((0, 5), 1.0),  # 'conj0' ('True') to 'disj0'
                # cond. for 'a'
                Conj2AtomEdge((0, 10), 1.0),  # 'conj0' ('True') to 'disj10'
                # cond. for 'b'
                Conj2AtomEdge((3, 11), 1.0),  # 'conj3' ('True') to 'disj11'
                Conj2AtomEdge((4, 11), 1.0),  # 'conj4' ('True') to 'disj11'
                # cond. for 'c'
                Conj2AtomEdge((0, 12), 1.0),  # 'conj0' ('True') to 'disj12'
                # global choice constraint
                Conj2AtomEdge((1, 1), 1.0),  # 'conj1' to 'False'
                # local constr. for 'a'
                Conj2AtomEdge((2, 1), 1.0),  # 'conj1' to 'False'
                # local constr. for 'b'
                Conj2AtomEdge((5, 1), 1.0),  # 'conj5' to 'False'
                # local constr. for 'c'
                Conj2AtomEdge((6, 1), 1.0),  # 'conj6' to 'False'
            },
        )
        # atom -> aggr
        self.assertEqual(
            set(edges[("atom", "in", "count")]),
            {
                Atom2CountEdge((2, 0), 1.0),  # 'a' to choice count
                Atom2CountEdge((3, 0), 1.0),  # 'b' to choice count
                Atom2CountEdge((4, 0), 1.0),  # 'c' to choice count
            },
        )
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertEqual(
            edges[("count", "defines", "atom")],
            [
                Count2AtomEdge((0, 6), 1.0),  # choice count to 'disj1' (aggr)
            ],
        )
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                PredLiteral("b"): 3,
                PredLiteral("c"): 4,
                LiteralCollection(rg.true_const): 5,
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement([Functional("a")], [PredLiteral("a")]),
                        AggrElement([Functional("b")], [PredLiteral("b")]),
                        AggrElement([Functional("c")], [PredLiteral("c")]),
                    ),
                    guards=(None, None),
                ): 6,
                PredLiteral("d"): 7,
                PredLiteral("e"): 8,
                PredLiteral("f"): 9,
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(rg.true_const): 0,
                LiteralCollection(PredLiteral("d")): 3,
                LiteralCollection(Naf(PredLiteral("f"))): 4,
                # NOTE: constraint conjunctions are not re-used
            },
        )
        self.assertEqual(
            rg.aggr_ids,
            {
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement([Functional("a")], [PredLiteral("a")]),
                        AggrElement([Functional("b")], [PredLiteral("b")]),
                        AggrElement([Functional("c")], [PredLiteral("c")]),
                    ),
                    guards=(None, None),
                ): 0,
            },
        )

        # choice tracking
        self.assertEqual(
            rg.choices,
            {
                Choice(
                    (
                        ChoiceElement(PredLiteral("a")),
                        ChoiceElement(PredLiteral("b"), [PredLiteral("d")]),
                        ChoiceElement(
                            PredLiteral("b"), [PredLiteral("e"), Naf(PredLiteral("f"))]
                        ),
                        ChoiceElement(PredLiteral("b"), [Naf(PredLiteral("f"))]),
                        ChoiceElement(PredLiteral("c")),
                    ),
                    guards=(None, None),
                ),
            },
        )
        self.assertEqual(
            rg.choice_edges,
            {
                Choice(
                    (
                        ChoiceElement(PredLiteral("a")),
                        ChoiceElement(PredLiteral("b"), [PredLiteral("d")]),
                        ChoiceElement(
                            PredLiteral("b"), [PredLiteral("e"), Naf(PredLiteral("f"))]
                        ),
                        ChoiceElement(PredLiteral("b"), [Naf(PredLiteral("f"))]),
                        ChoiceElement(PredLiteral("c")),
                    ),
                    guards=(None, None),
                ): [
                    0,  # (0, 2)
                    1,  # (0, 3)
                    2,  # (0, 4)
                ]
                # '$\wedge_0$' to 'a', 'b', 'c'
            },
        )

    def test_count_aggregate(self):
        prog = Program.from_string(
            r"""

        a :- #count{1;2:b;2:c,not d;2:not d;3}.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode("b", 0.0, False),
                AtomNode("c", 0.0, False),
                AtomNode("d", 0.0, False),
                AtomNode(r"$\vee_1$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            set(nodes["conj"]),
            {
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
            },
        )
        # aggregate nodes
        self.assertEqual(set(nodes["count"]), {SumNode(r"$\#_0$", 0.0, (-1,) * 4)})
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # body
                Atom2ConjEdge((3, 2), 1.0),  # 'disj0' (aggr) to 'conj2' (body)
                # conditions for '2'
                Atom2ConjEdge((4, 0), 1.0),  # 'b' to 'conj0'
                Atom2ConjEdge((6, 1), -1.0),  # 'f' to 'conj1'
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # body to head
                Conj2AtomEdge((2, 2), 1.0),  # 'conj2' (body) to 'a'
                # conditions for '2'
                Conj2AtomEdge((0, 7), 1.0),  # 'conj0' to 'disj1'
                Conj2AtomEdge((1, 7), 1.0),  # 'conj1' to 'disj1'
            },
        )
        # atom -> aggr
        self.assertEqual(
            set(edges[("atom", "in", "count")]),
            {
                Atom2CountEdge((0, 0), 2.0),  # 'True' to choice count ('1','3')
                Atom2CountEdge((7, 0), 1.0),  # 'disj1' to choice count ('2')
            },
        )
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertEqual(
            edges[("count", "defines", "atom")],
            [
                Count2AtomEdge((0, 3), 1.0),  # choice count to 'disj0' (aggr)
            ],
        )
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 3,
                PredLiteral("b"): 4,
                PredLiteral("c"): 5,
                PredLiteral("d"): 6,
                # TODO: local conditions not reused
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(PredLiteral("b")): 0,
                LiteralCollection(Naf(PredLiteral("d"))): 1,
                LiteralCollection(
                    AggrLiteral(
                        AggrCount(),
                        (
                            AggrElement([Number(1)]),
                            AggrElement([Number(2)], [PredLiteral("b")]),
                            AggrElement(
                                [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                            ),
                            AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                            AggrElement([Number(3)]),
                        ),
                        guards=(None, None),
                    )
                ): 2,
            },
        )
        self.assertEqual(
            rg.aggr_ids,
            {
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 0
            },
        )

        # choice tracking
        self.assertFalse(rg.choices)
        self.assertFalse(rg.choice_edges)

    def test_sum_aggregate(self):
        prog = Program.from_string(
            r"""

        a :- #sum{1;2:b;2:c,not d;2:not d;3}.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode("b", 0.0, False),
                AtomNode("c", 0.0, False),
                AtomNode("d", 0.0, False),
                AtomNode(r"$\vee_1$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            set(nodes["conj"]),
            {
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
            },
        )
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertEqual(set(nodes["sum"]), {SumNode(r"$\Sigma_0$", 0.0, (-1,) * 4)})
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # body
                Atom2ConjEdge((3, 2), 1.0),  # 'disj0' (aggr) to 'conj2' (body)
                # conditions for '2'
                Atom2ConjEdge((4, 0), 1.0),  # 'b' to 'conj0'
                Atom2ConjEdge((6, 1), -1.0),  # 'f' to 'conj1'
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # body to head
                Conj2AtomEdge((2, 2), 1.0),  # 'conj2' (body) to 'a'
                # conditions for '2'
                Conj2AtomEdge((0, 7), 1.0),  # 'conj0' to 'disj1'
                Conj2AtomEdge((1, 7), 1.0),  # 'conj1' to 'disj1'
            },
        )
        # atom -> aggr
        self.assertEqual(
            set(edges[("atom", "in", "sum")]),
            {
                Atom2SumEdge((0, 0), 4.0),  # 'True' to aggregate ('1','3')
                Atom2SumEdge((7, 0), 2.0),  # 'disj1' to aggregate ('2')
            },
        )
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertEqual(
            edges[("sum", "defines", "atom")],
            [
                Sum2AtomEdge((0, 3), 1.0),  # choice count to 'disj0' (aggr)
            ],
        )
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                AggrLiteral(
                    AggrSum(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 3,
                PredLiteral("b"): 4,
                PredLiteral("c"): 5,
                PredLiteral("d"): 6,
                # TODO: local conditions not reused
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(PredLiteral("b")): 0,
                LiteralCollection(Naf(PredLiteral("d"))): 1,
                LiteralCollection(
                    AggrLiteral(
                        AggrSum(),
                        (
                            AggrElement([Number(1)]),
                            AggrElement([Number(2)], [PredLiteral("b")]),
                            AggrElement(
                                [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                            ),
                            AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                            AggrElement([Number(3)]),
                        ),
                        guards=(None, None),
                    )
                ): 2,
            },
        )
        self.assertEqual(
            rg.aggr_ids,
            {
                AggrLiteral(
                    AggrSum(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 0
            },
        )

        # choice tracking
        self.assertFalse(rg.choices)
        self.assertFalse(rg.choice_edges)

    def test_min_aggregate(self):
        prog = Program.from_string(
            r"""

        a :- #min{1;2:b;2:c,not d;2:not d;3}.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode("b", 0.0, False),
                AtomNode("c", 0.0, False),
                AtomNode("d", 0.0, False),
                AtomNode(r"$\vee_1$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            set(nodes["conj"]),
            {
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
            },
        )
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertEqual(set(nodes["min"]), {MinNode(r"$MIN_0$", 0.0, (-1,) * 4)})
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # body
                Atom2ConjEdge((3, 2), 1.0),  # 'disj0' (aggr) to 'conj2' (body)
                # conditions for '2'
                Atom2ConjEdge((4, 0), 1.0),  # 'b' to 'conj0'
                Atom2ConjEdge((6, 1), -1.0),  # 'f' to 'conj1'
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # body to head
                Conj2AtomEdge((2, 2), 1.0),  # 'conj2' (body) to 'a'
                # conditions for '2'
                Conj2AtomEdge((0, 7), 1.0),  # 'conj0' to 'disj1'
                Conj2AtomEdge((1, 7), 1.0),  # 'conj1' to 'disj1'
            },
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertEqual(
            set(edges[("atom", "in", "min")]),
            {
                Atom2MinEdge((0, 0), 1.0),  # 'True' to choice count ('1','3')
                Atom2MinEdge((7, 0), 2.0),  # 'disj1' to choice count ('2')
            },
        )
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertEqual(
            edges[("min", "defines", "atom")],
            [
                Min2AtomEdge((0, 3), 1.0),  # choice count to 'disj0' (aggr)
            ],
        )

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                AggrLiteral(
                    AggrMin(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 3,
                PredLiteral("b"): 4,
                PredLiteral("c"): 5,
                PredLiteral("d"): 6,
                # TODO: local conditions not reused
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(PredLiteral("b")): 0,
                LiteralCollection(Naf(PredLiteral("d"))): 1,
                LiteralCollection(
                    AggrLiteral(
                        AggrMin(),
                        (
                            AggrElement([Number(1)]),
                            AggrElement([Number(2)], [PredLiteral("b")]),
                            AggrElement(
                                [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                            ),
                            AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                            AggrElement([Number(3)]),
                        ),
                        guards=(None, None),
                    )
                ): 2,
            },
        )
        self.assertEqual(
            rg.aggr_ids,
            {
                AggrLiteral(
                    AggrMin(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 0
            },
        )

        # choice tracking
        self.assertFalse(rg.choices)
        self.assertFalse(rg.choice_edges)

    def test_max_aggregate(self):
        prog = Program.from_string(
            r"""

        a :- #max{1;2:b;2:c,not d;2:not d;3}.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("a", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode("b", 0.0, False),
                AtomNode("c", 0.0, False),
                AtomNode("d", 0.0, False),
                AtomNode(r"$\vee_1$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            set(nodes["conj"]),
            {
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
            },
        )
        # aggregate nodes
        self.assertFalse(nodes["count"])
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertEqual(set(nodes["max"]), {MaxNode(r"$MAX_0$", 0.0, (-1,) * 4)})

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # body
                Atom2ConjEdge((3, 2), 1.0),  # 'disj0' (aggr) to 'conj2' (body)
                # conditions for '2'
                Atom2ConjEdge((4, 0), 1.0),  # 'b' to 'conj0'
                Atom2ConjEdge((6, 1), -1.0),  # 'f' to 'conj1'
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # body to head
                Conj2AtomEdge((2, 2), 1.0),  # 'conj2' (body) to 'a'
                # conditions for '2'
                Conj2AtomEdge((0, 7), 1.0),  # 'conj0' to 'disj1'
                Conj2AtomEdge((1, 7), 1.0),  # 'conj1' to 'disj1'
            },
        )
        # atom -> aggr
        self.assertFalse(edges[("atom", "in", "count")])
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertEqual(
            set(edges[("atom", "in", "max")]),
            {
                Atom2MaxEdge((0, 0), 3.0),  # 'True' to choice count ('1','3')
                Atom2MaxEdge((7, 0), 2.0),  # 'disj1' to choice count ('2')
            },
        )
        # aggr -> atom
        self.assertFalse(edges[("count", "defines", "atom")])
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertEqual(
            edges[("max", "defines", "atom")],
            [
                Max2AtomEdge((0, 3), 1.0),  # choice count to 'disj0' (aggr)
            ],
        )

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral("a"): 2,
                AggrLiteral(
                    AggrMax(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 3,
                PredLiteral("b"): 4,
                PredLiteral("c"): 5,
                PredLiteral("d"): 6,
                # TODO: local conditions not reused
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(PredLiteral("b")): 0,
                LiteralCollection(Naf(PredLiteral("d"))): 1,
                LiteralCollection(
                    AggrLiteral(
                        AggrMax(),
                        (
                            AggrElement([Number(1)]),
                            AggrElement([Number(2)], [PredLiteral("b")]),
                            AggrElement(
                                [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                            ),
                            AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                            AggrElement([Number(3)]),
                        ),
                        guards=(None, None),
                    )
                ): 2,
            },
        )
        self.assertEqual(
            rg.aggr_ids,
            {
                AggrLiteral(
                    AggrMax(),
                    (
                        AggrElement([Number(1)]),
                        AggrElement([Number(2)], [PredLiteral("b")]),
                        AggrElement(
                            [Number(2)], [PredLiteral("c"), Naf(PredLiteral("d"))]
                        ),
                        AggrElement([Number(2)], [Naf(PredLiteral("d"))]),
                        AggrElement([Number(3)]),
                    ),
                    guards=(None, None),
                ): 0
            },
        )

        # choice tracking
        self.assertFalse(rg.choices)
        self.assertFalse(rg.choice_edges)

    def test_NPP_fact(self):
        prog = Program.from_string(
            r"""

        #npp(h(a,b), [0,1,2]).

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("h(a,b,0)", 0.0, False),
                AtomNode("h(a,b,1)", 0.0, False),
                AtomNode("h(a,b,2)", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode(r"$\vee_1$", 0.0, True),
                AtomNode(r"$\vee_2$", 0.0, True),
                AtomNode(r"$\vee_3$", 0.0, True),
                AtomNode(r"$\vee_4$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            nodes["conj"],
            [
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
                ConjNode(r"$\wedge_3$", 0.0),
                ConjNode(r"$\wedge_4$", 0.0),
            ],
        )
        # aggregate nodes
        self.assertEqual(
            set(nodes["count"]), {CountNode(r"$\#_0$", 0.0, (0, 1, -1, -1))}
        )
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # body to conj.
                Atom2ConjEdge((0, 0), 1.0),  # 'True' to 'conj0' ('True')
                # global constraint for choice (given sat. body)
                Atom2ConjEdge((5, 1), 1.0),  # 'disj0' (body) to 'conj1'
                Atom2ConjEdge((6, 1), -1.0),  # 'disj1' (aggr) to 'conj1'
                # local constraint for 'h(a,b,0)'
                Atom2ConjEdge((2, 2), 1.0),  # 'h(a,b,0)' to 'conj2'
                Atom2ConjEdge((5, 2), 1.0),  # 'disj0' (body) to 'conj2'
                Atom2ConjEdge((7, 2), -1.0),  # 'disj1' (aggr) to 'conj2'
                # local constraint for 'h(a,b,1)'
                Atom2ConjEdge((3, 3), 1.0),  # 'h(a,b,0)' to 'conj3'
                Atom2ConjEdge((5, 3), 1.0),  # 'disj0' (body) to 'conj3'
                Atom2ConjEdge((8, 3), -1.0),  # 'disj1' (aggr) to 'conj3'
                # local constraint for 'h(a,b,2)'
                Atom2ConjEdge((4, 4), 1.0),  # 'h(a,b,0)' to 'conj4'
                Atom2ConjEdge((5, 4), 1.0),  # 'disj0' (body) to 'conj4'
                Atom2ConjEdge((9, 4), -1.0),  # 'disj1' (aggr) to 'conj4'
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # conj. to head
                Conj2AtomEdge((0, 2), 1.0),  # 'conj0' ('True') to 'h(a,b,0)'
                Conj2AtomEdge((0, 3), 1.0),  # 'conj0' ('True') to 'h(a,b,1)'
                Conj2AtomEdge((0, 4), 1.0),  # 'conj0' ('True') to 'h(a,b,2)'
                # conj. to aux. body atom
                Conj2AtomEdge((0, 5), 1.0),  # 'conj0' ('True') to 'disj0'
                # global constraint
                Conj2AtomEdge((1, 1), 1.0),  # 'conj1' ('True') to 'False'
                # cond. for 'h(a,b,0)'
                Conj2AtomEdge((2, 1), 1.0),  # 'conj2' ('True') to 'False'
                Conj2AtomEdge((0, 7), 1.0),  # 'conj0' ('True') to 'disj2'
                # cond. for 'h(a,b,1)'
                Conj2AtomEdge((3, 1), 1.0),  # 'conj3' ('True') to 'False'
                Conj2AtomEdge((0, 8), 1.0),  # 'conj0' ('True') to 'disj3'
                # cond. for 'h(a,b,2)'
                Conj2AtomEdge((4, 1), 1.0),  # 'conj4' ('True') to 'False'
                Conj2AtomEdge((0, 9), 1.0),  # 'conj0' ('True') to 'disj4'
            },
        )
        # atom -> aggr
        self.assertEqual(
            set(edges[("atom", "in", "count")]),
            {
                Atom2CountEdge((2, 0), 1.0),  # 'a' to choice count
                Atom2CountEdge((3, 0), 1.0),  # 'b' to choice count
                Atom2CountEdge((4, 0), 1.0),  # 'c' to choice count
            },
        )
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertEqual(
            edges[("count", "defines", "atom")],
            [
                Count2AtomEdge((0, 6), 1.0),  # choice count to 'disj1' (aggr)
            ],
        )
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral(
                    "h", SymbolicConstant("a"), SymbolicConstant("b"), Number(0)
                ): 2,
                PredLiteral(
                    "h", SymbolicConstant("a"), SymbolicConstant("b"), Number(1)
                ): 3,
                PredLiteral(
                    "h", SymbolicConstant("a"), SymbolicConstant("b"), Number(2)
                ): 4,
                LiteralCollection(rg.true_const): 5,
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                        ),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ): 6,
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(rg.true_const): 0,
                # NOTE: constraint conjunctions are not re-used
            },
        )
        self.assertEqual(
            rg.aggr_ids,
            {
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                        ),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ): 0
            },
        )

        # choice tracking
        self.assertFalse(rg.choices)
        self.assertFalse(rg.choice_edges)

        # NPP tracking
        self.assertEqual(
            rg.npps,
            {
                NPP(
                    "h",
                    [SymbolicConstant("a"), SymbolicConstant("b")],
                    [Number(0), Number(1), Number(2)],
                ),
            },
        )
        self.assertEqual(
            rg.npp_edges,
            {
                NPP(
                    "h",
                    [SymbolicConstant("a"), SymbolicConstant("b")],
                    [Number(0), Number(1), Number(2)],
                ): [
                    0,  # (0, 2)
                    1,  # (0, 3)
                    2,  # (0, 4)
                ]
                # '$\wedge_0$' to 'a', 'b', 'c'
            },
        )

    def test_NPP_rule(self):
        prog = Program.from_string(
            r"""

        #npp(h(a,b), [0,1,2]) :- a, not b.

        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("h(a,b,0)", 0.0, False),
                AtomNode("h(a,b,1)", 0.0, False),
                AtomNode("h(a,b,2)", 0.0, False),
                AtomNode("a", 0.0, False),
                AtomNode("b", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode(r"$\vee_1$", 0.0, True),
                AtomNode(r"$\vee_2$", 0.0, True),
                AtomNode(r"$\vee_3$", 0.0, True),
                AtomNode(r"$\vee_4$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            nodes["conj"],
            [
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
                ConjNode(r"$\wedge_3$", 0.0),
                ConjNode(r"$\wedge_4$", 0.0),
                ConjNode(r"$\wedge_5$", 0.0),
            ],
        )
        # aggregate nodes
        self.assertEqual(
            set(nodes["count"]), {CountNode(r"$\#_0$", 0.0, (0, 1, -1, -1))}
        )
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                # condition for 'choices'
                Atom2ConjEdge((0, 2), 1.0),  # 'True' to 'conj2'
                # body to conj.
                Atom2ConjEdge((5, 0), 1.0),  # 'a' to 'conj2' ('a, not b')
                Atom2ConjEdge((6, 0), -1.0),  # 'b' to 'conj2' ('a, not b')
                # global constraint for choice (given sat. body)
                Atom2ConjEdge((7, 1), 1.0),  # 'disj0' (body) to 'conj1'
                Atom2ConjEdge((8, 1), -1.0),  # 'disj1' (aggr) to 'conj1'
                # local constraint for 'h(a,b,0)'
                Atom2ConjEdge((2, 3), 1.0),  # 'h(a,b,0)' to 'conj2'
                Atom2ConjEdge((7, 3), 1.0),  # 'disj0' (body) to 'conj2'
                Atom2ConjEdge((9, 3), -1.0),  # 'disj1' (aggr) to 'conj2'
                # local constraint for 'h(a,b,1)'
                Atom2ConjEdge((3, 4), 1.0),  # 'h(a,b,0)' to 'conj3'
                Atom2ConjEdge((7, 4), 1.0),  # 'disj0' (body) to 'conj3'
                Atom2ConjEdge((10, 4), -1.0),  # 'disj1' (aggr) to 'conj3'
                # local constraint for 'h(a,b,2)'
                Atom2ConjEdge((4, 5), 1.0),  # 'h(a,b,0)' to 'conj4'
                Atom2ConjEdge((7, 5), 1.0),  # 'disj0' (body) to 'conj4'
                Atom2ConjEdge((11, 5), -1.0),  # 'disj1' (aggr) to 'conj4'
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # conj. to head
                Conj2AtomEdge((0, 2), 1.0),  # 'conj0' ('True') to 'h(a,b,0)'
                Conj2AtomEdge((0, 3), 1.0),  # 'conj0' ('True') to 'h(a,b,1)'
                Conj2AtomEdge((0, 4), 1.0),  # 'conj0' ('True') to 'h(a,b,2)'
                # conj. to aux. body atom
                Conj2AtomEdge((0, 7), 1.0),  # 'conj0' ('True') to 'disj0'
                # global constraint
                Conj2AtomEdge((1, 1), 1.0),  # 'conj1' ('True') to 'False'
                # cond. for 'h(a,b,0)'
                Conj2AtomEdge((3, 1), 1.0),  # 'conj2' ('True') to 'False'
                Conj2AtomEdge((2, 9), 1.0),  # 'conj0' ('True') to 'disj2'
                # cond. for 'h(a,b,1)'
                Conj2AtomEdge((4, 1), 1.0),  # 'conj3' ('True') to 'False'
                Conj2AtomEdge((2, 10), 1.0),  # 'conj0' ('True') to 'disj3'
                # cond. for 'h(a,b,2)'
                Conj2AtomEdge((5, 1), 1.0),  # 'conj4' ('True') to 'False'
                Conj2AtomEdge((2, 11), 1.0),  # 'conj0' ('True') to 'disj4'
            },
        )
        # atom -> aggr
        self.assertEqual(
            set(edges[("atom", "in", "count")]),
            {
                Atom2CountEdge((2, 0), 1.0),  # 'a' to choice count
                Atom2CountEdge((3, 0), 1.0),  # 'b' to choice count
                Atom2CountEdge((4, 0), 1.0),  # 'c' to choice count
            },
        )
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertEqual(
            edges[("count", "defines", "atom")],
            [
                Count2AtomEdge((0, 8), 1.0),  # choice count to 'disj1' (aggr)
            ],
        )
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # node id dictionaries
        self.assertEqual(rg.atom_ids[rg.true_const], 0)
        self.assertEqual(rg.atom_ids[rg.false_const], 1)
        self.assertEqual(
            rg.atom_ids,
            {
                rg.true_const: 0,
                rg.false_const: 1,
                PredLiteral(
                    "h", SymbolicConstant("a"), SymbolicConstant("b"), Number(0)
                ): 2,
                PredLiteral(
                    "h", SymbolicConstant("a"), SymbolicConstant("b"), Number(1)
                ): 3,
                PredLiteral(
                    "h", SymbolicConstant("a"), SymbolicConstant("b"), Number(2)
                ): 4,
                PredLiteral("a"): 5,
                PredLiteral("b"): 6,
                LiteralCollection(PredLiteral("a"), Naf(PredLiteral("b"))): 7,
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                        ),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ): 8,
            },
        )
        self.assertEqual(
            rg.conj_ids,
            {
                LiteralCollection(PredLiteral("a"), Naf(PredLiteral("b"))): 0,
                LiteralCollection(rg.true_const): 2,
                # NOTE: constraint conjunctions are not re-used
            },
        )
        self.assertEqual(
            rg.aggr_ids,
            {
                AggrLiteral(
                    AggrCount(),
                    (
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(0),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(1),
                                )
                            ],
                        ),
                        AggrElement(
                            [
                                Functional(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                            [
                                PredLiteral(
                                    "h",
                                    SymbolicConstant("a"),
                                    SymbolicConstant("b"),
                                    Number(2),
                                )
                            ],
                        ),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ): 0
            },
        )

        # choice tracking
        self.assertFalse(rg.choices)
        self.assertFalse(rg.choice_edges)

        # NPP tracking
        self.assertEqual(
            rg.npps,
            {
                NPP(
                    "h",
                    [SymbolicConstant("a"), SymbolicConstant("b")],
                    [Number(0), Number(1), Number(2)],
                ),
            },
        )
        self.assertEqual(
            rg.npp_edges,
            {
                NPP(
                    "h",
                    [SymbolicConstant("a"), SymbolicConstant("b")],
                    [Number(0), Number(1), Number(2)],
                ): [
                    0,  # (0, 2)
                    1,  # (0, 3)
                    2,  # (0, 4)
                ]
                # '$\wedge_0$' to 'a', 'b', 'c'
            },
        )

    def test_4_queens(self):
        prog = Program.from_string(
            r"""
        n(0). n(1). n(2). n(3).

        % choose one row per queen
        1={q(0,0);q(0,1);q(0,2);q(0,3)} :- n(0).
        1={q(1,0);q(1,1);q(1,2);q(1,3)} :- n(1).
        1={q(2,0);q(2,1);q(2,2);q(2,3)} :- n(2).
        1={q(3,0);q(3,1);q(3,2);q(3,3)} :- n(3).

        % check columns
        :- q(0,0), q(1,0), 0 < 1.
        :- q(0,0), q(2,0), 0 < 2.
        :- q(0,0), q(3,0), 0 < 3.
        :- q(1,0), q(2,0), 1 < 2.
        :- q(1,0), q(3,0), 1 < 3.
        :- q(2,0), q(3,0), 2 < 3.

        :- q(0,1), q(1,1), 0 < 1.
        :- q(0,1), q(2,1), 0 < 2.
        :- q(0,1), q(3,1), 0 < 3.
        :- q(1,1), q(2,1), 1 < 2.
        :- q(1,1), q(3,1), 1 < 3.
        :- q(2,1), q(3,1), 2 < 3.

        :- q(0,2), q(1,2), 0 < 1.
        :- q(0,2), q(2,2), 0 < 2.
        :- q(0,2), q(3,2), 0 < 3.
        :- q(1,2), q(2,2), 1 < 2.
        :- q(1,2), q(3,2), 1 < 3.
        :- q(2,2), q(3,2), 2 < 3.

        :- q(0,3), q(1,3), 0 < 1.
        :- q(0,3), q(2,3), 0 < 2.
        :- q(0,3), q(3,3), 0 < 3.
        :- q(1,3), q(2,3), 1 < 2.
        :- q(1,3), q(3,3), 1 < 3.
        :- q(2,3), q(3,3), 2 < 3.

        % check diagonals 1
        :- q(0,0), q(1,1), n(1), 1=0+1, 1=0+1, 1 > 0.
        :- q(1,1), q(2,2), n(1), 2=1+1, 2=1+1, 1 > 0.
        :- q(2,2), q(3,3), n(1), 3=2+1, 3=2+1, 1 > 0.

        :- q(0,1), q(1,2), n(1), 1=0+1, 2=1+1, 1 > 0.
        :- q(1,2), q(2,3), n(1), 2=1+1, 3=2+1, 1 > 0.

        :- q(0,2), q(1,3), n(1), 1=0+1, 3=2+1, 1 > 0.

        :- q(1,0), q(2,1), n(1), 2=1+1, 1=0+1, 1 > 0.
        :- q(2,1), q(3,2), n(1), 3=2+1, 2=1+1, 1 > 0.

        :- q(2,0), q(3,1), n(1), 3=2+1, 1=0+1, 1 > 0.

        :- q(0,0), q(2,2), n(2), 2=0+2, 2=0+2, 2 > 0.
        :- q(1,1), q(3,3), n(2), 3=1+2, 3=1+2, 2 > 0.

        :- q(0,1), q(2,3), n(2), 2=0+2, 3=1+2, 2 > 0.

        :- q(1,0), q(3,2), n(2), 3=1+2, 2=0+2, 2 > 0.

        :- q(0,0), q(3,3), n(3), 3=0+3, 3=0+3, 3 > 0.

        % check diagonals 2
        :- q(0,3), q(1,2), n(1), 1=0+1, 3=2+1, 1 > 0.
        :- q(1,2), q(2,1), n(1), 2=1+1, 2=1+1, 1 > 0.
        :- q(2,1), q(3,0), n(1), 3=2+1, 1=0+1, 1 > 0.

        :- q(0,2), q(1,1), n(1), 1=0+1, 2=1+1, 1 > 0.
        :- q(1,1), q(2,0), n(1), 2=1+1, 1=0+1, 1 > 0.

        :- q(0,1), q(1,0), n(1), 1=0+1, 1=0+1, 1 > 0.

        :- q(1,3), q(2,2), n(1), 2=1+1, 3=2+1, 1 > 0.
        :- q(2,2), q(3,1), n(1), 3=2+1, 2=1+1, 1 > 0.

        :- q(2,3), q(3,2), n(1), 3=2+1, 3=2+1, 1 > 0.

        :- q(0,3), q(2,1), n(2), 2=0+2, 3=1+2, 2 > 0.
        :- q(1,2), q(3,0), n(2), 3=1+2, 2=0+2, 2 > 0.

        :- q(0,2), q(2,0), n(2), 2=0+2, 2=0+2, 2 > 0.

        :- q(1,3), q(3,1), n(2), 3=1+2, 3=1+2, 2 > 0.

        :- q(0,3), q(3,0), n(3), 3=0+3, 3=0+3, 3 > 0.
        """
        )

        # create reasoning graph
        rg = ReasoningGraph(prog)

        # zip node and edges attributes for convenience (since order may differ)
        nodes = zip_nodes(rg.node_dict)
        edges = zip_edges(rg.edge_dict)

        # atom nodes
        self.assertEqual(
            nodes["atom"],
            [
                AtomNode(r"$\top$", 1.0, True),
                AtomNode(r"$\bot$", 0.0, True),
                AtomNode("n(0)", 1.0, False),
                AtomNode("n(1)", 1.0, False),
                AtomNode("n(2)", 1.0, False),
                AtomNode("n(3)", 1.0, False),
                AtomNode("q(0,0)", 0.0, False),
                AtomNode("q(0,1)", 0.0, False),
                AtomNode("q(0,2)", 0.0, False),
                AtomNode("q(0,3)", 0.0, False),
                AtomNode(r"$\vee_0$", 0.0, True),
                AtomNode(r"$\vee_1$", 0.0, True),
                AtomNode(r"$\vee_2$", 0.0, True),
                AtomNode(r"$\vee_3$", 0.0, True),
                AtomNode(r"$\vee_4$", 0.0, True),
                AtomNode(r"$\vee_5$", 0.0, True),
                AtomNode("q(1,0)", 0.0, False),
                AtomNode("q(1,1)", 0.0, False),
                AtomNode("q(1,2)", 0.0, False),
                AtomNode("q(1,3)", 0.0, False),
                AtomNode(r"$\vee_6$", 0.0, True),
                AtomNode(r"$\vee_7$", 0.0, True),
                AtomNode(r"$\vee_8$", 0.0, True),
                AtomNode(r"$\vee_9$", 0.0, True),
                AtomNode(r"$\vee_10$", 0.0, True),
                AtomNode(r"$\vee_11$", 0.0, True),
                AtomNode("q(2,0)", 0.0, False),
                AtomNode("q(2,1)", 0.0, False),
                AtomNode("q(2,2)", 0.0, False),
                AtomNode("q(2,3)", 0.0, False),
                AtomNode(r"$\vee_12$", 0.0, True),
                AtomNode(r"$\vee_13$", 0.0, True),
                AtomNode(r"$\vee_14$", 0.0, True),
                AtomNode(r"$\vee_15$", 0.0, True),
                AtomNode(r"$\vee_16$", 0.0, True),
                AtomNode(r"$\vee_17$", 0.0, True),
                AtomNode("q(3,0)", 0.0, False),
                AtomNode("q(3,1)", 0.0, False),
                AtomNode("q(3,2)", 0.0, False),
                AtomNode("q(3,3)", 0.0, False),
                AtomNode(r"$\vee_18$", 0.0, True),
                AtomNode(r"$\vee_19$", 0.0, True),
                AtomNode(r"$\vee_20$", 0.0, True),
                AtomNode(r"$\vee_21$", 0.0, True),
                AtomNode(r"$\vee_22$", 0.0, True),
                AtomNode(r"$\vee_23$", 0.0, True),
            ],
        )
        # conjunction nodes
        self.assertEqual(
            nodes["conj"],
            [
                ConjNode(r"$\wedge_0$", 0.0),
                ConjNode(r"$\wedge_1$", 0.0),
                ConjNode(r"$\wedge_2$", 0.0),
                ConjNode(r"$\wedge_3$", 0.0),
                ConjNode(r"$\wedge_4$", 0.0),
                ConjNode(r"$\wedge_5$", 0.0),
                ConjNode(r"$\wedge_6$", 0.0),
                ConjNode(r"$\wedge_7$", 0.0),
                ConjNode(r"$\wedge_8$", 0.0),
                ConjNode(r"$\wedge_9$", 0.0),
                ConjNode(r"$\wedge_10$", 0.0),
                ConjNode(r"$\wedge_11$", 0.0),
                ConjNode(r"$\wedge_12$", 0.0),
                ConjNode(r"$\wedge_13$", 0.0),
                ConjNode(r"$\wedge_14$", 0.0),
                ConjNode(r"$\wedge_15$", 0.0),
                ConjNode(r"$\wedge_16$", 0.0),
                ConjNode(r"$\wedge_17$", 0.0),
                ConjNode(r"$\wedge_18$", 0.0),
                ConjNode(r"$\wedge_19$", 0.0),
                ConjNode(r"$\wedge_20$", 0.0),
                ConjNode(r"$\wedge_21$", 0.0),
                ConjNode(r"$\wedge_22$", 0.0),
                ConjNode(r"$\wedge_23$", 0.0),
                ConjNode(r"$\wedge_24$", 0.0),
                # columns constraints
                ConjNode(r"$\wedge_25$", 0.0),
                ConjNode(r"$\wedge_26$", 0.0),
                ConjNode(r"$\wedge_27$", 0.0),
                ConjNode(r"$\wedge_28$", 0.0),
                ConjNode(r"$\wedge_29$", 0.0),
                ConjNode(r"$\wedge_30$", 0.0),
                ConjNode(r"$\wedge_31$", 0.0),
                ConjNode(r"$\wedge_32$", 0.0),
                ConjNode(r"$\wedge_33$", 0.0),
                ConjNode(r"$\wedge_34$", 0.0),
                ConjNode(r"$\wedge_35$", 0.0),
                ConjNode(r"$\wedge_36$", 0.0),
                ConjNode(r"$\wedge_37$", 0.0),
                ConjNode(r"$\wedge_38$", 0.0),
                ConjNode(r"$\wedge_39$", 0.0),
                ConjNode(r"$\wedge_40$", 0.0),
                ConjNode(r"$\wedge_41$", 0.0),
                ConjNode(r"$\wedge_42$", 0.0),
                ConjNode(r"$\wedge_43$", 0.0),
                ConjNode(r"$\wedge_44$", 0.0),
                ConjNode(r"$\wedge_45$", 0.0),
                ConjNode(r"$\wedge_46$", 0.0),
                ConjNode(r"$\wedge_47$", 0.0),
                ConjNode(r"$\wedge_48$", 0.0),
                # diagonal constraints 1
                ConjNode(r"$\wedge_49$", 0.0),
                ConjNode(r"$\wedge_50$", 0.0),
                ConjNode(r"$\wedge_51$", 0.0),
                ConjNode(r"$\wedge_52$", 0.0),
                ConjNode(r"$\wedge_53$", 0.0),
                ConjNode(r"$\wedge_54$", 0.0),
                ConjNode(r"$\wedge_55$", 0.0),
                ConjNode(r"$\wedge_56$", 0.0),
                ConjNode(r"$\wedge_57$", 0.0),
                ConjNode(r"$\wedge_58$", 0.0),
                ConjNode(r"$\wedge_59$", 0.0),
                ConjNode(r"$\wedge_60$", 0.0),
                ConjNode(r"$\wedge_61$", 0.0),
                ConjNode(r"$\wedge_62$", 0.0),
                # diagonal constraints 2
                ConjNode(r"$\wedge_63$", 0.0),
                ConjNode(r"$\wedge_64$", 0.0),
                ConjNode(r"$\wedge_65$", 0.0),
                ConjNode(r"$\wedge_66$", 0.0),
                ConjNode(r"$\wedge_67$", 0.0),
                ConjNode(r"$\wedge_68$", 0.0),
                ConjNode(r"$\wedge_69$", 0.0),
                ConjNode(r"$\wedge_70$", 0.0),
                ConjNode(r"$\wedge_71$", 0.0),
                ConjNode(r"$\wedge_72$", 0.0),
                ConjNode(r"$\wedge_73$", 0.0),
                ConjNode(r"$\wedge_74$", 0.0),
                ConjNode(r"$\wedge_75$", 0.0),
                ConjNode(r"$\wedge_76$", 0.0),
            ],
        )
        # aggregate nodes
        self.assertEqual(
            set(nodes["count"]), {CountNode(r"$\#_0$", 0.0, (0, 1, -1, -1))}
        )
        self.assertFalse(nodes["sum"])
        self.assertFalse(nodes["min"])
        self.assertFalse(nodes["max"])

        # atom -> conj
        self.assertEqual(
            set(edges[("atom", "in", "conj")]),
            {
                Atom2ConjEdge((0, 0), 1.0),  # 'True' to 'conj0' ('True')
                Atom2ConjEdge((2, 1), 1.0),
                Atom2ConjEdge((6, 3), 1.0),
                Atom2ConjEdge((7, 4), 1.0),
                Atom2ConjEdge((8, 5), 1.0),
                Atom2ConjEdge((9, 6), 1.0),
                Atom2ConjEdge((10, 2), 1.0),
                Atom2ConjEdge((10, 3), 1.0),
                Atom2ConjEdge((10, 4), 1.0),
                Atom2ConjEdge((10, 5), 1.0),
                Atom2ConjEdge((10, 6), 1.0),
                Atom2ConjEdge((11, 2), -1.0),
                Atom2ConjEdge((12, 3), -1.0),
                Atom2ConjEdge((13, 4), -1.0),
                Atom2ConjEdge((14, 5), -1.0),
                Atom2ConjEdge((15, 6), -1.0),
                Atom2ConjEdge((3, 7), 1.0),
                Atom2ConjEdge((16, 9), 1.0),
                Atom2ConjEdge((17, 10), 1.0),
                Atom2ConjEdge((18, 11), 1.0),
                Atom2ConjEdge((19, 12), 1.0),
                Atom2ConjEdge((20, 8), 1.0),
                Atom2ConjEdge((20, 9), 1.0),
                Atom2ConjEdge((20, 10), 1.0),
                Atom2ConjEdge((20, 11), 1.0),
                Atom2ConjEdge((20, 12), 1.0),
                Atom2ConjEdge((21, 8), -1.0),
                Atom2ConjEdge((22, 9), -1.0),
                Atom2ConjEdge((23, 10), -1.0),
                Atom2ConjEdge((24, 11), -1.0),
                Atom2ConjEdge((25, 12), -1.0),
                Atom2ConjEdge((4, 13), 1.0),
                Atom2ConjEdge((26, 15), 1.0),
                Atom2ConjEdge((27, 16), 1.0),
                Atom2ConjEdge((28, 17), 1.0),
                Atom2ConjEdge((29, 18), 1.0),
                Atom2ConjEdge((30, 14), 1.0),
                Atom2ConjEdge((30, 15), 1.0),
                Atom2ConjEdge((30, 16), 1.0),
                Atom2ConjEdge((30, 17), 1.0),
                Atom2ConjEdge((30, 18), 1.0),
                Atom2ConjEdge((31, 14), -1.0),
                Atom2ConjEdge((32, 15), -1.0),
                Atom2ConjEdge((33, 16), -1.0),
                Atom2ConjEdge((34, 17), -1.0),
                Atom2ConjEdge((35, 18), -1.0),
                Atom2ConjEdge((5, 19), 1.0),
                Atom2ConjEdge((36, 21), 1.0),
                Atom2ConjEdge((37, 22), 1.0),
                Atom2ConjEdge((38, 23), 1.0),
                Atom2ConjEdge((39, 24), 1.0),
                Atom2ConjEdge((40, 20), 1.0),
                Atom2ConjEdge((40, 21), 1.0),
                Atom2ConjEdge((40, 22), 1.0),
                Atom2ConjEdge((40, 23), 1.0),
                Atom2ConjEdge((40, 24), 1.0),
                Atom2ConjEdge((41, 20), -1.0),
                Atom2ConjEdge((42, 21), -1.0),
                Atom2ConjEdge((43, 22), -1.0),
                Atom2ConjEdge((44, 23), -1.0),
                Atom2ConjEdge((45, 24), -1.0),
                # column constraints
                Atom2ConjEdge((6, 25), 1.0),
                Atom2ConjEdge((16, 25), 1.0),
                Atom2ConjEdge((6, 26), 1.0),
                Atom2ConjEdge((26, 26), 1.0),
                Atom2ConjEdge((6, 27), 1.0),
                Atom2ConjEdge((36, 27), 1.0),
                Atom2ConjEdge((16, 28), 1.0),
                Atom2ConjEdge((26, 28), 1.0),
                Atom2ConjEdge((16, 29), 1.0),
                Atom2ConjEdge((36, 29), 1.0),
                Atom2ConjEdge((26, 30), 1.0),
                Atom2ConjEdge((36, 30), 1.0),
                Atom2ConjEdge((7, 31), 1.0),
                Atom2ConjEdge((17, 31), 1.0),
                Atom2ConjEdge((7, 32), 1.0),
                Atom2ConjEdge((27, 32), 1.0),
                Atom2ConjEdge((7, 33), 1.0),
                Atom2ConjEdge((37, 33), 1.0),
                Atom2ConjEdge((17, 34), 1.0),
                Atom2ConjEdge((27, 34), 1.0),
                Atom2ConjEdge((17, 35), 1.0),
                Atom2ConjEdge((37, 35), 1.0),
                Atom2ConjEdge((27, 36), 1.0),
                Atom2ConjEdge((37, 36), 1.0),
                Atom2ConjEdge((8, 37), 1.0),
                Atom2ConjEdge((18, 37), 1.0),
                Atom2ConjEdge((8, 38), 1.0),
                Atom2ConjEdge((28, 38), 1.0),
                Atom2ConjEdge((8, 39), 1.0),
                Atom2ConjEdge((38, 39), 1.0),
                Atom2ConjEdge((18, 40), 1.0),
                Atom2ConjEdge((28, 40), 1.0),
                Atom2ConjEdge((18, 41), 1.0),
                Atom2ConjEdge((38, 41), 1.0),
                Atom2ConjEdge((28, 42), 1.0),
                Atom2ConjEdge((38, 42), 1.0),
                Atom2ConjEdge((9, 43), 1.0),
                Atom2ConjEdge((19, 43), 1.0),
                Atom2ConjEdge((9, 44), 1.0),
                Atom2ConjEdge((29, 44), 1.0),
                Atom2ConjEdge((9, 45), 1.0),
                Atom2ConjEdge((39, 45), 1.0),
                Atom2ConjEdge((19, 46), 1.0),
                Atom2ConjEdge((29, 46), 1.0),
                Atom2ConjEdge((19, 47), 1.0),
                Atom2ConjEdge((39, 47), 1.0),
                Atom2ConjEdge((29, 48), 1.0),
                Atom2ConjEdge((39, 48), 1.0),
                # diagonal constraints 1
                # 1
                Atom2ConjEdge((3, 49), 1.0),
                Atom2ConjEdge((6, 49), 1.0),
                Atom2ConjEdge((17, 49), 1.0),
                Atom2ConjEdge((3, 50), 1.0),
                Atom2ConjEdge((17, 50), 1.0),
                Atom2ConjEdge((28, 50), 1.0),
                Atom2ConjEdge((3, 51), 1.0),
                Atom2ConjEdge((28, 51), 1.0),
                Atom2ConjEdge((39, 51), 1.0),
                Atom2ConjEdge((3, 52), 1.0),
                Atom2ConjEdge((7, 52), 1.0),
                Atom2ConjEdge((18, 52), 1.0),
                Atom2ConjEdge((3, 53), 1.0),
                Atom2ConjEdge((18, 53), 1.0),
                Atom2ConjEdge((29, 53), 1.0),
                Atom2ConjEdge((3, 54), 1.0),
                Atom2ConjEdge((8, 54), 1.0),
                Atom2ConjEdge((19, 54), 1.0),
                Atom2ConjEdge((3, 55), 1.0),
                Atom2ConjEdge((16, 55), 1.0),
                Atom2ConjEdge((27, 55), 1.0),
                Atom2ConjEdge((3, 56), 1.0),
                Atom2ConjEdge((27, 56), 1.0),
                Atom2ConjEdge((38, 56), 1.0),
                Atom2ConjEdge((3, 57), 1.0),
                Atom2ConjEdge((26, 57), 1.0),
                Atom2ConjEdge((37, 57), 1.0),
                # 2
                Atom2ConjEdge((4, 58), 1.0),
                Atom2ConjEdge((6, 58), 1.0),
                Atom2ConjEdge((28, 58), 1.0),
                Atom2ConjEdge((4, 59), 1.0),
                Atom2ConjEdge((17, 59), 1.0),
                Atom2ConjEdge((39, 59), 1.0),
                Atom2ConjEdge((4, 60), 1.0),
                Atom2ConjEdge((7, 60), 1.0),
                Atom2ConjEdge((29, 60), 1.0),
                Atom2ConjEdge((4, 61), 1.0),
                Atom2ConjEdge((16, 61), 1.0),
                Atom2ConjEdge((38, 61), 1.0),
                # 3
                Atom2ConjEdge((5, 62), 1.0),
                Atom2ConjEdge((6, 62), 1.0),
                Atom2ConjEdge((39, 62), 1.0),
                # diagonal constraints 2
                # 1
                Atom2ConjEdge((3, 63), 1.0),
                Atom2ConjEdge((9, 63), 1.0),
                Atom2ConjEdge((18, 63), 1.0),
                Atom2ConjEdge((3, 64), 1.0),
                Atom2ConjEdge((18, 64), 1.0),
                Atom2ConjEdge((27, 64), 1.0),
                Atom2ConjEdge((3, 65), 1.0),
                Atom2ConjEdge((27, 65), 1.0),
                Atom2ConjEdge((36, 65), 1.0),
                Atom2ConjEdge((3, 66), 1.0),
                Atom2ConjEdge((8, 66), 1.0),
                Atom2ConjEdge((17, 66), 1.0),
                Atom2ConjEdge((3, 67), 1.0),
                Atom2ConjEdge((17, 67), 1.0),
                Atom2ConjEdge((26, 67), 1.0),
                Atom2ConjEdge((3, 68), 1.0),
                Atom2ConjEdge((7, 68), 1.0),
                Atom2ConjEdge((16, 68), 1.0),
                Atom2ConjEdge((3, 69), 1.0),
                Atom2ConjEdge((19, 69), 1.0),
                Atom2ConjEdge((28, 69), 1.0),
                Atom2ConjEdge((3, 70), 1.0),
                Atom2ConjEdge((28, 70), 1.0),
                Atom2ConjEdge((37, 70), 1.0),
                Atom2ConjEdge((3, 71), 1.0),
                Atom2ConjEdge((29, 71), 1.0),
                Atom2ConjEdge((38, 71), 1.0),
                # 2
                Atom2ConjEdge((4, 72), 1.0),
                Atom2ConjEdge((9, 72), 1.0),
                Atom2ConjEdge((27, 72), 1.0),
                Atom2ConjEdge((4, 73), 1.0),
                Atom2ConjEdge((18, 73), 1.0),
                Atom2ConjEdge((36, 73), 1.0),
                Atom2ConjEdge((4, 74), 1.0),
                Atom2ConjEdge((8, 74), 1.0),
                Atom2ConjEdge((26, 74), 1.0),
                Atom2ConjEdge((4, 75), 1.0),
                Atom2ConjEdge((19, 75), 1.0),
                Atom2ConjEdge((37, 75), 1.0),
                # 3
                Atom2ConjEdge((5, 76), 1.0),
                Atom2ConjEdge((9, 76), 1.0),
                Atom2ConjEdge((36, 76), 1.0),
            },
        )
        # conj -> atom
        self.assertEqual(
            set(edges[("conj", "defines", "atom")]),
            {
                # facts
                Conj2AtomEdge((0, 2), 1.0),  # conj0 (true) -> n(0)
                Conj2AtomEdge((0, 3), 1.0),  # conj0 (true) -> n(1)
                Conj2AtomEdge((0, 4), 1.0),  # conj0 (true) -> n(2)
                Conj2AtomEdge((0, 5), 1.0),  # conj0 (true) -> n(3)
                Conj2AtomEdge((0, 12), 1.0),
                Conj2AtomEdge((0, 13), 1.0),
                Conj2AtomEdge((0, 14), 1.0),
                Conj2AtomEdge((0, 15), 1.0),
                Conj2AtomEdge((2, 1), 1.0),
                Conj2AtomEdge((3, 1), 1.0),
                Conj2AtomEdge((4, 1), 1.0),
                Conj2AtomEdge((5, 1), 1.0),
                Conj2AtomEdge((6, 1), 1.0),
                Conj2AtomEdge((1, 6), 1.0),  # conj1 (body) -> q(0,0)
                Conj2AtomEdge((1, 7), 1.0),  # conj1 (body) -> q(0,1)
                Conj2AtomEdge((1, 8), 1.0),  # conj1 (body) -> q(0,2)
                Conj2AtomEdge((1, 9), 1.0),  # conj1 (body) -> q(0,3)
                Conj2AtomEdge((1, 10), 1.0),  # conj1 (body) -> aux. body atom
                Conj2AtomEdge((0, 22), 1.0),
                Conj2AtomEdge((0, 23), 1.0),
                Conj2AtomEdge((0, 24), 1.0),
                Conj2AtomEdge((0, 25), 1.0),
                Conj2AtomEdge((8, 1), 1.0),
                Conj2AtomEdge((9, 1), 1.0),
                Conj2AtomEdge((10, 1), 1.0),
                Conj2AtomEdge((11, 1), 1.0),
                Conj2AtomEdge((12, 1), 1.0),
                Conj2AtomEdge((7, 16), 1.0),  # conj7 (body) -> q(1,0)
                Conj2AtomEdge((7, 17), 1.0),  # conj7 (body) -> q(1,1)
                Conj2AtomEdge((7, 18), 1.0),  # conj7 (body) -> q(1,2)
                Conj2AtomEdge((7, 19), 1.0),  # conj7 (body) -> q(1,3)
                Conj2AtomEdge((7, 20), 1.0),  # conj7 (body) -> aux. body atom
                Conj2AtomEdge((0, 32), 1.0),
                Conj2AtomEdge((0, 33), 1.0),
                Conj2AtomEdge((0, 34), 1.0),
                Conj2AtomEdge((0, 35), 1.0),
                Conj2AtomEdge((14, 1), 1.0),
                Conj2AtomEdge((15, 1), 1.0),
                Conj2AtomEdge((16, 1), 1.0),
                Conj2AtomEdge((17, 1), 1.0),
                Conj2AtomEdge((18, 1), 1.0),
                Conj2AtomEdge((13, 26), 1.0),  # conj13 (body) -> q(2,0)
                Conj2AtomEdge((13, 27), 1.0),  # conj13 (body) -> q(2,1)
                Conj2AtomEdge((13, 28), 1.0),  # conj13 (body) -> q(2,2)
                Conj2AtomEdge((13, 29), 1.0),  # conj13 (body) -> q(2,3)
                Conj2AtomEdge((13, 30), 1.0),  # conj13 (body) -> aux. body atom
                Conj2AtomEdge((0, 42), 1.0),
                Conj2AtomEdge((0, 43), 1.0),
                Conj2AtomEdge((0, 44), 1.0),
                Conj2AtomEdge((0, 45), 1.0),
                Conj2AtomEdge((20, 1), 1.0),
                Conj2AtomEdge((21, 1), 1.0),
                Conj2AtomEdge((22, 1), 1.0),
                Conj2AtomEdge((23, 1), 1.0),
                Conj2AtomEdge((24, 1), 1.0),
                Conj2AtomEdge((19, 36), 1.0),  # conj19 (body) -> q(3,0)
                Conj2AtomEdge((19, 37), 1.0),  # conj19 (body) -> q(3,1)
                Conj2AtomEdge((19, 38), 1.0),  # conj19 (body) -> q(3,2)
                Conj2AtomEdge((19, 39), 1.0),  # conj19 (body) -> q(3,3)
                Conj2AtomEdge((19, 40), 1.0),  # conj19 (body) -> aux. body atom
                # column constraints
                Conj2AtomEdge((25, 1), 1.0),
                Conj2AtomEdge((26, 1), 1.0),
                Conj2AtomEdge((27, 1), 1.0),
                Conj2AtomEdge((28, 1), 1.0),
                Conj2AtomEdge((29, 1), 1.0),
                Conj2AtomEdge((30, 1), 1.0),
                Conj2AtomEdge((31, 1), 1.0),
                Conj2AtomEdge((32, 1), 1.0),
                Conj2AtomEdge((33, 1), 1.0),
                Conj2AtomEdge((34, 1), 1.0),
                Conj2AtomEdge((35, 1), 1.0),
                Conj2AtomEdge((36, 1), 1.0),
                Conj2AtomEdge((37, 1), 1.0),
                Conj2AtomEdge((38, 1), 1.0),
                Conj2AtomEdge((39, 1), 1.0),
                Conj2AtomEdge((40, 1), 1.0),
                Conj2AtomEdge((41, 1), 1.0),
                Conj2AtomEdge((42, 1), 1.0),
                Conj2AtomEdge((43, 1), 1.0),
                Conj2AtomEdge((44, 1), 1.0),
                Conj2AtomEdge((45, 1), 1.0),
                Conj2AtomEdge((46, 1), 1.0),
                Conj2AtomEdge((47, 1), 1.0),
                Conj2AtomEdge((48, 1), 1.0),
                # diagonal constraints 1
                Conj2AtomEdge((49, 1), 1.0),
                Conj2AtomEdge((50, 1), 1.0),
                Conj2AtomEdge((51, 1), 1.0),
                Conj2AtomEdge((52, 1), 1.0),
                Conj2AtomEdge((53, 1), 1.0),
                Conj2AtomEdge((54, 1), 1.0),
                Conj2AtomEdge((55, 1), 1.0),
                Conj2AtomEdge((56, 1), 1.0),
                Conj2AtomEdge((57, 1), 1.0),
                Conj2AtomEdge((58, 1), 1.0),
                Conj2AtomEdge((59, 1), 1.0),
                Conj2AtomEdge((60, 1), 1.0),
                Conj2AtomEdge((61, 1), 1.0),
                Conj2AtomEdge((62, 1), 1.0),
                # diagonal constraints 2
                Conj2AtomEdge((63, 1), 1.0),
                Conj2AtomEdge((64, 1), 1.0),
                Conj2AtomEdge((65, 1), 1.0),
                Conj2AtomEdge((66, 1), 1.0),
                Conj2AtomEdge((67, 1), 1.0),
                Conj2AtomEdge((68, 1), 1.0),
                Conj2AtomEdge((69, 1), 1.0),
                Conj2AtomEdge((70, 1), 1.0),
                Conj2AtomEdge((71, 1), 1.0),
                Conj2AtomEdge((72, 1), 1.0),
                Conj2AtomEdge((73, 1), 1.0),
                Conj2AtomEdge((74, 1), 1.0),
                Conj2AtomEdge((75, 1), 1.0),
                Conj2AtomEdge((76, 1), 1.0),
            },
        )
        # atom -> aggr
        self.assertEqual(
            set(edges[("atom", "in", "count")]),
            {
                Atom2CountEdge((6, 0), 1.0),
                Atom2CountEdge((7, 0), 1.0),
                Atom2CountEdge((8, 0), 1.0),
                Atom2CountEdge((9, 0), 1.0),
                Atom2CountEdge((16, 1), 1.0),
                Atom2CountEdge((17, 1), 1.0),
                Atom2CountEdge((18, 1), 1.0),
                Atom2CountEdge((19, 1), 1.0),
                Atom2CountEdge((26, 2), 1.0),
                Atom2CountEdge((27, 2), 1.0),
                Atom2CountEdge((28, 2), 1.0),
                Atom2CountEdge((29, 2), 1.0),
                Atom2CountEdge((36, 3), 1.0),
                Atom2CountEdge((37, 3), 1.0),
                Atom2CountEdge((38, 3), 1.0),
                Atom2CountEdge((39, 3), 1.0),
            },
        )
        self.assertFalse(edges[("atom", "in", "sum")])
        self.assertFalse(edges[("atom", "in", "min")])
        self.assertFalse(edges[("atom", "in", "max")])
        # aggr -> atom
        self.assertEqual(
            edges[("count", "defines", "atom")],
            [
                Count2AtomEdge((0, 11), 1.0),
                Count2AtomEdge((1, 21), 1.0),
                Count2AtomEdge((2, 31), 1.0),
                Count2AtomEdge((3, 41), 1.0),
            ],
        )
        self.assertFalse(edges[("sum", "defines", "atom")])
        self.assertFalse(edges[("min", "defines", "atom")])
        self.assertFalse(edges[("max", "defines", "atom")])

        # choice tracking
        self.assertEqual(
            rg.choices,
            {
                Choice(
                    (
                        ChoiceElement(PredLiteral("q", Number(0), Number(0))),
                        ChoiceElement(PredLiteral("q", Number(0), Number(1))),
                        ChoiceElement(PredLiteral("q", Number(0), Number(2))),
                        ChoiceElement(PredLiteral("q", Number(0), Number(3))),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ),
                Choice(
                    (
                        ChoiceElement(PredLiteral("q", Number(1), Number(0))),
                        ChoiceElement(PredLiteral("q", Number(1), Number(1))),
                        ChoiceElement(PredLiteral("q", Number(1), Number(2))),
                        ChoiceElement(PredLiteral("q", Number(1), Number(3))),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ),
                Choice(
                    (
                        ChoiceElement(PredLiteral("q", Number(2), Number(0))),
                        ChoiceElement(PredLiteral("q", Number(2), Number(1))),
                        ChoiceElement(PredLiteral("q", Number(2), Number(2))),
                        ChoiceElement(PredLiteral("q", Number(2), Number(3))),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ),
                Choice(
                    (
                        ChoiceElement(PredLiteral("q", Number(3), Number(0))),
                        ChoiceElement(PredLiteral("q", Number(3), Number(1))),
                        ChoiceElement(PredLiteral("q", Number(3), Number(2))),
                        ChoiceElement(PredLiteral("q", Number(3), Number(3))),
                    ),
                    guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                ),
            },
        )
        self.assertEqual(len(rg.choice_edges), 4)
        self.assertEqual(
            set(
                edges[("conj", "defines", "atom")][i]
                for i in rg.choice_edges[
                    Choice(
                        (
                            ChoiceElement(PredLiteral("q", Number(0), Number(0))),
                            ChoiceElement(PredLiteral("q", Number(0), Number(1))),
                            ChoiceElement(PredLiteral("q", Number(0), Number(2))),
                            ChoiceElement(PredLiteral("q", Number(0), Number(3))),
                        ),
                        guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                    )
                ]
            ),
            {
                Conj2AtomEdge((1, 6), 1.0),
                Conj2AtomEdge((1, 7), 1.0),
                Conj2AtomEdge((1, 8), 1.0),
                Conj2AtomEdge((1, 9), 1.0),
            },
        )
        self.assertEqual(
            set(
                edges[("conj", "defines", "atom")][i]
                for i in rg.choice_edges[
                    Choice(
                        (
                            ChoiceElement(PredLiteral("q", Number(1), Number(0))),
                            ChoiceElement(PredLiteral("q", Number(1), Number(1))),
                            ChoiceElement(PredLiteral("q", Number(1), Number(2))),
                            ChoiceElement(PredLiteral("q", Number(1), Number(3))),
                        ),
                        guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                    )
                ]
            ),
            {
                Conj2AtomEdge((7, 16), 1.0),
                Conj2AtomEdge((7, 17), 1.0),
                Conj2AtomEdge((7, 18), 1.0),
                Conj2AtomEdge((7, 19), 1.0),
            },
        )
        self.assertEqual(
            set(
                edges[("conj", "defines", "atom")][i]
                for i in rg.choice_edges[
                    Choice(
                        (
                            ChoiceElement(PredLiteral("q", Number(2), Number(0))),
                            ChoiceElement(PredLiteral("q", Number(2), Number(1))),
                            ChoiceElement(PredLiteral("q", Number(2), Number(2))),
                            ChoiceElement(PredLiteral("q", Number(2), Number(3))),
                        ),
                        guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                    )
                ]
            ),
            {
                Conj2AtomEdge((13, 26), 1.0),
                Conj2AtomEdge((13, 27), 1.0),
                Conj2AtomEdge((13, 28), 1.0),
                Conj2AtomEdge((13, 29), 1.0),
            },
        )
        self.assertEqual(
            set(
                edges[("conj", "defines", "atom")][i]
                for i in rg.choice_edges[
                    Choice(
                        (
                            ChoiceElement(PredLiteral("q", Number(3), Number(0))),
                            ChoiceElement(PredLiteral("q", Number(3), Number(1))),
                            ChoiceElement(PredLiteral("q", Number(3), Number(2))),
                            ChoiceElement(PredLiteral("q", Number(3), Number(3))),
                        ),
                        guards=(Guard(RelOp.EQUAL, Number(1), False), None),
                    )
                ]
            ),
            {
                Conj2AtomEdge((19, 36), 1.0),
                Conj2AtomEdge((19, 37), 1.0),
                Conj2AtomEdge((19, 38), 1.0),
                Conj2AtomEdge((19, 39), 1.0),
            },
        )


# TODO: check that certain_atoms works
# TODO: check individual methods
# TODO: negative aggregates

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
