from arrows.primitivearrow import PrimitiveArrow
from typing import Dict, List, MutableMapping, Set
from sympy import Expr, Rel, Gt, Ne


def same_shape(shape): return shape

class AddArrow(PrimitiveArrow):
    """Addition"""

    def __init__(self):
        name = 'Add'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = ptv[i[0]] + ptv[i[1]]
        if i[0] in ptv and o[0] in ptv:
            ptv[i[1]] = ptv[o[0]] - ptv[i[0]]
        if i[1] in ptv and o[0] in ptv:
            ptv[i[0]] = ptv[o[0]] - ptv[i[1]]
        return ptv


class SubArrow(PrimitiveArrow):
    """Subtraction. Out[1] = In[0] - In[1]"""

    def __init__(self):
        name = 'Sub'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = ptv[i[0]] - ptv[i[1]]
        if i[0] in ptv and o[0] in ptv:
            ptv[i[1]] = ptv[i[0]] - ptv[o[0]]
        if i[1] in ptv and o[0] in ptv:
            ptv[i[0]] = ptv[i[1]] + ptv[o[0]]
        return ptv


class MulArrow(PrimitiveArrow):
    """Multiplication"""

    def __init__(self):
        name = 'Mul'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = ptv[i[0]] * ptv[i[1]]
        if i[0] in ptv and o[0] in ptv:
            ptv[i[1]] = ptv[o[0]] / ptv[i[0]]
        if i[1] in ptv and o[0] in ptv:
            ptv[i[0]] = ptv[o[0]] / ptv[i[1]]
        return ptv


class DivArrow(PrimitiveArrow):
    """Division"""

    def __init__(self):
        name = 'Div'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        constraints = []
        if 1 in input_expr:
            constraints.append(Ne(input_expr[1], 0))
        return constraints

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = ptv[i[0]] / ptv[i[1]]
        if i[0] in ptv and o[0] in ptv:
            ptv[i[1]] = ptv[i[0]] / ptv[o[0]]
        if i[1] in ptv and o[0] in ptv:
            ptv[i[0]] = ptv[i[1]] * ptv[o[0]]
        return ptv


class PowArrow(PrimitiveArrow):
    """Computes the power of one value to another.

    Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
    corresponding elements in `x` and `y`. For example"""

    def __init__(self):
        name = 'Pow'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        constraints = []
        if 0 in input_expr:
            constraints.append(Gt(input_expr[0], 0))
        if 0 in output_expr:
            constraints.append(Gt(output_expr[0], 0))
        return constraints

class ExpArrow(PrimitiveArrow):
    """Exponential e^x"""

    def __init__(self):
        name = 'Exp'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)


class LogArrow(PrimitiveArrow):
    """Log_e(x)"""

    def __init__(self):
        name = 'Log'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

class LogBaseArrow(PrimitiveArrow):
    """Log_y(x)"""

    def __init__(self):
        name = 'LogBase'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
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


class AddNArrow(PrimitiveArrow):
    """Element wise sum of n tensors"""

    def __init__(self, n: int):
        name = 'AddN'
        super().__init__(n_in_ports=n, n_out_ports=1, name=name)


class SinArrow(PrimitiveArrow):
    """Sin"""

    def __init__(self):
        name = 'Sin'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)


class ASinArrow(PrimitiveArrow):
    """Sin"""

    def __init__(self):
        name = 'ASin'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)


class CosArrow(PrimitiveArrow):
    """Cos"""

    def __init__(self):
        name = 'Cos'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)


class ClipArrow(PrimitiveArrow):
    """Cos"""

    def __init__(self):
        name = 'Clip'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

class ACosArrow(PrimitiveArrow):
    """ACos"""

    def __init__(self):
        name = 'ACos'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)


class AbsArrow(PrimitiveArrow):
    """Abs(x)"""

    def __init__(self):
        name = 'Abs'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        constraints = []
        if 1 in input_expr:
            constraints.append(Ge(input_expr[0], 0))
        return constraints


class MaxArrow(PrimitiveArrow):
    """Returns the max of x and y (i.e. x > y ? x : y) element-wise."""

    def __init__(self):
        name = 'Max'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)


class ReduceMeanArrow(PrimitiveArrow):
    """Computes the mean of elements across dimensions of a tensor.
    Port0: input
    Port1: reduction_indices
    """

    def __init__(self,
                 n_inputs,
                 axis=None,
                 keep_dims=False,
                 reduction_indices=None):

        name = 'ReduceMean'
        self.axis = axis
        self.kee_dims = keep_dims
        self.reduction_indices = reduction_indices
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)
