from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Set, Union

from ground_slash.program import Expr, LiteralCollection, Naf

if TYPE_CHECKING:
    from ground_slash.program import Constraint, Query, Statement, Term, Variable
    from ground_slash.progrm.safety_characterization import SafetyTriplet


class Disjunction(LiteralCollection):
    # TODO: generalize to Literals as well as Disjunctions/Disjunctions
    def __hash__(self) -> int:
        return hash(("disjunction", frozenset(self.literals)))

    def __eq__(self, other: "Any") -> bool:
        return (
            isinstance(other, Disjunction)
            and len(self) == len(other)
            and frozenset(self.literals) == frozenset(other.literals)
        )

    def as_conjunction(self) -> "Conjunction":
        return Conjunction(Naf(literal, ~literal.naf) for literal in self)


class Conjunction(LiteralCollection):
    # TODO: generalize to Literals as well as Disjunctions/Disjunctions
    def __hash__(self) -> int:
        return hash(("conjunction", frozenset(self.literals)))

    def __eq__(self, other: "Any") -> bool:
        return (
            isinstance(other, Conjunction)
            and len(self) == len(other)
            and frozenset(self.literals) == frozenset(other.literals)
        )


class ComplexQuery(Expr):
    def __init__(self, *constraints: Iterable["Constraint"]) -> None:
        if len(constraints) < 2:
            raise ValueError(f"Complex query must containt at least 2 constraints, but got {len(constraints)}.")

        self.constraints = tuple(constraints)

    def __str__(self) -> str:
        return "\n".join([str(constr) for constr in self.constraints])

    def __eq__(self, other: "ComplexQuery") -> bool:
        return isinstance(other, ComplexQuery) and (frozenset(self.constraints) == frozenset(other.constraints))

    def __hash__(self) -> int:
        return hash(frozenset(self.constraints))

    def vars(self) -> Set["Variable"]:
        raise NotImplementedError()

    def global_vars(self) -> Set["Variable"]:
        raise NotImplementedError()

    def safety(self, statement: Optional[Union["Statement", "Query"]]=None) -> "SafetyTriplet":
        raise NotImplementedError()

    def substitute(self, subst: Dict[str, "Term"]) -> "Expr":
        raise NotImplementedError()