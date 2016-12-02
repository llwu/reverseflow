"""These are arrows for control flow of input"""
from reverseflow.arrows.primitivearrow import PrimitiveArrow
from typing import List, MutableMapping, Set
from sympy import Expr, Eq, Rel



class DuplArrow(PrimitiveArrow):
    """
    Duplicate input
    f(x) = (x, x, ..., x)
    """
    def __init__(self, n_duplications=2):
        name = 'Dupl'
        super().__init__(n_in_ports=1, n_out_ports=n_duplications, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        constraints = []
        for i in output_expr.keys():
            for j output_expr.keys():
                if i != j:
                    constraints.append(Eq(output_expr[i], output_expr[j]))
        return constraints


class IdentityArrow(PrimitiveArrow):
    """
    Identity input
    f(x) = x
    """

    def __init__(self) -> None:
        name = 'Identity'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)
