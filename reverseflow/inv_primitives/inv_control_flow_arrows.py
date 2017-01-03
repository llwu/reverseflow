from arrows.primitivearrow import PrimitiveArrow
from typing import List, MutableMapping
from sympy import Expr, Eq, Rel


class InvDuplArrow(PrimitiveArrow):
    """InvDupl f(x1,...,xn) = x"""

    def __init__(self, n_duplications=2):
        name = "InvDupl"
        super().__init__(n_in_ports=n_duplications, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> List[Rel]:
        assert 0 in output_expr
        constraints = []
        for i in input_expr.keys():
            constraints.append(Eq(output_expr[0], input_expr[i]))
        return constraints
