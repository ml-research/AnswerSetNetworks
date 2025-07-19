from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Set, Union

from ground_slash.program import Expr, Literal, LiteralCollection

if TYPE_CHECKING:
    from ground_slash.program import Constraint, Query, Statement, Term, Variable
    from ground_slash.progrm.safety_characterization import SafetyTriplet


class Conjunction(LiteralCollection):
    def __init__(self, *literals: Literal, naf: bool = False) -> None:
        super().__init__(*literals)
        self.naf = naf

    def __hash__(self) -> int:
        return hash(("conjunction", self.naf, frozenset(self.literals)))

    def __eq__(self, other: "Any") -> bool:
        return (
            isinstance(other, Conjunction)
            and self.naf == other.naf
            and len(self) == len(other)
            and frozenset(self.literals) == frozenset(other.literals)
        )

    def __str__(self) -> str:
        return f"\u2227({','.join(str(literal) for literal in self.literals)})"

    def __abs__(self) -> "Conjunction":
        """TODO"""
        if self.naf:
            copy = deepcopy(self)
            copy.naf = False

            return copy

        return self


class Disjunction(LiteralCollection):
    def __init__(self, *literals: Literal, naf: bool = False) -> None:
        super().__init__(*literals)
        self.naf = naf

    def __hash__(self) -> int:
        return hash(("disjunction", self.naf, frozenset(self.literals)))

    def __eq__(self, other: "Any") -> bool:
        return (
            isinstance(other, Disjunction)
            and self.naf == other.naf
            and len(self) == len(other)
            and frozenset(self.literals) == frozenset(other.literals)
        )

    def __str__(self) -> str:
        return f"\u2228({','.join(str(literal) for literal in self.literals)})"

    def __abs__(self) -> "Disjunction":
        """TODO"""
        if self.naf:
            copy = deepcopy(self)
            copy.naf = False

            return copy

        return self


class MultiConstraint(Expr):
    def __init__(self, *constraints: "Constraint") -> None:
        if len(constraints) < 2:
            raise ValueError(
                f"Multi-constraint must containt at least 2 constraints, but got {len(constraints)}."
            )

        self.constraints = tuple(constraints)

    def __str__(self) -> str:
        return "\n".join([str(constr) for constr in self.constraints])

    def __eq__(self, other: "MultiConstraint") -> bool:
        return isinstance(other, MultiConstraint) and (
            frozenset(self.constraints) == frozenset(other.constraints)
        )

    def __hash__(self) -> int:
        return hash(frozenset(self.constraints))

    def vars(self) -> Set["Variable"]:
        raise NotImplementedError()

    def global_vars(self) -> Set["Variable"]:
        raise NotImplementedError()

    def safety(
        self, statement: Optional[Union["Statement", "Query"]] = None
    ) -> "SafetyTriplet":
        raise NotImplementedError()

    def substitute(self, subst: Dict[str, "Term"]) -> "Expr":
        raise NotImplementedError()
