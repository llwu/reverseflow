"""These are arrows for control flow of input"""

from reverseflow.arrows.port import InPort, OutPort
from reverseflow.arrows.primitivearrow import PrimitiveArrow
from typing import List
from sympy import Expr

class DuplArrow(PrimitiveArrow):
    """
    Duplicate input
    f(x) = (x, x, ..., x)
    """

    def __init__(self, n_duplications=2) -> None:
        self.name = "Dupl"
        self.n_duplications = n_duplications
        self.in_ports = [InPort(self, 0)]
        self.out_ports = [OutPort(self, i) for i in range(n_duplications)]

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        constraints = []
        if 0 in output_expr and 1 in output_expr:
            constraints.append(output_expr[0] - output_expr[1])
        if 0 in output_expr and 0 in input_expr:
            constraints.append(output_expr[0] - input_expr[0])
        if 1 in output_expr and 0 in input_expr:
            constraints.append(output_expr[1] - input_expr[0])
        return constraints

    def invert(self):
        pass


class IdentityArrow(PrimitiveArrow):
    """
    Identity input
    f(x) = x
    """

    def __init__(self) -> None:
        self.name = "Identity"
        self.in_ports = [InPort(self, 0)]
        self.out_ports = [OutPort(self, 0)]

    def invert(self):
        pass
