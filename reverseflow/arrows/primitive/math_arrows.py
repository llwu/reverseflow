from reverseflow.arrows.primitivearrow import PrimitiveArrow
from typing import Dict, List
from sympy import Expr, Rel, Gt


class AddArrow(PrimitiveArrow):
    """Addition"""

    def __init__(self):
        name = 'Add'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)


class SubArrow(PrimitiveArrow):
    """Subtraction. Out[1] = In[0] - In[1]"""

    def __init__(self):
        name = 'Sub'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)


class MulArrow(PrimitiveArrow):
    """Multiplication"""

    def __init__(self):
        name = 'Mul'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)


class DivArrow(PrimitiveArrow):
    """Division"""

    def __init__(self):
        name = 'Div'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[int, Expr], output_expr: Dict[int, Expr]) -> List[Rel]:
        constraints = []
        if 1 in input_expr:
            constraints.append(Ne(input_expr[1], 0))
        return constraints


class ExpArrow(PrimitiveArrow):
    """Exponentiaion"""

    def __init__(self):
        name = 'Exp'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[int, Expr], output_expr: Dict[int, Expr]) -> List[Rel]:
        constraints = []
        if 0 in input_expr:
            constraints.append(Gt(input_expr[0], 0))
        if 0 in output_expr:
            constraints.append(Gt(output_expr[0], 0))
        return constraints


class LogArrow(PrimitiveArrow):
    """Logarithm"""

    def __init__(self):
        name = 'Log'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[int, Expr], output_expr: Dict[int, Expr]) -> List[Rel]:
        constraints = []
        if 0 in input_expr:
            constraints.append(Gt(input_expr[0], 0))
        if 1 in input_expr:
            constraints.append(Gt(input_expr[1], 0))
        return constraints


class NegArrow(PrimitiveArrow):
    """Negation"""

    def __init__(self):
        name = 'Neg'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)
