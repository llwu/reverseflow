import math
from typing import Dict, List, MutableMapping, Set

import numpy as np
from sympy import Expr, Rel, Gt, Ne

from arrows.primitivearrow import PrimitiveArrow


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

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = ptv[i[0]] ** ptv[i[1]]
        if i[0] in ptv and o[0] in ptv:
            ptv[i[1]] = math.log(ptv[o[0]],  ptv[i[0]])
        if i[1] in ptv and o[0] in ptv:
            ptv[i[0]] = ptv[o[0]] ** (1 / ptv[i[1]])
        return ptv

class ExpArrow(PrimitiveArrow):
    """Exponential e^x"""

    def __init__(self):
        name = 'Exp'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = math.exp(ptv[i[0]])
        if o[0] in ptv:
            ptv[i[1]] = math.log(ptv[o[0]])
        return ptv


class LogArrow(PrimitiveArrow):
    """Log_e(x)"""

    def __init__(self):
        name = 'Log'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = math.log(ptv[i[0]])
        if o[0] in ptv:
            ptv[i[1]] = math.exp(ptv[o[0]])
        return ptv

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

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = math.log(ptv[i[0]], ptv[i[1]])
        if i[0] in ptv and o[0] in ptv:
            ptv[i[1]] = ptv[i[0]] ** (1 / ptv[o[0]])
        if i[1] in ptv and o[0] in ptv:
            ptv[i[0]] = ptv[i[1]] ** (ptv[o[0]])
        return ptv


class NegArrow(PrimitiveArrow):
    """Negation"""

    def __init__(self):
        name = 'Neg'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = -ptv[i[0]]
        if o[0] in ptv:
            ptv[i[1]] = -ptv[o[0]]
        return ptv


class AddNArrow(PrimitiveArrow):
    """Element wise sum of n tensors"""

    def __init__(self, n: int):
        name = 'AddN'
        super().__init__(n_in_ports=n, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        last_unknown = None
        n_known = 0
        sum_known = 0
        for in_port in i:
            if in_port not in ptv:
                last_unknown = in_port
            else:
                n_known += 1
                sum_known += ptv[in_port]
        if n_known == len(i):
            ptv[o[0]] = sum_known
        if n_known == len(i) - 1 and o[0] in ptv:
            ptv[last_unknown] = ptv[o[0]] - sum_known
        return ptv


class SinArrow(PrimitiveArrow):
    """Sin"""

    def __init__(self):
        name = 'Sin'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = math.sin(ptv[i[0]])
        return ptv


class ASinArrow(PrimitiveArrow):
    """ASin"""

    def __init__(self):
        name = 'ASin'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = math.asin(ptv[i[0]])
        if o[0] in ptv:
            ptv[i[0]] = math.sin(ptv[o[0]])
        return ptv


class CosArrow(PrimitiveArrow):
    """Cos"""

    def __init__(self):
        name = 'Cos'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = math.cos(ptv[i[0]])
        return ptv


class ClipArrow(PrimitiveArrow):
    """Clip"""

    def __init__(self):
        name = 'Clip'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        if i[0] in ptv and i[1] in ptv and i[2] in ptv:
            ptv[o[0]] = np.clip(ptv[i[0]], ptv[i[1]], ptv[i[2]])
        return ptv

class ACosArrow(PrimitiveArrow):
    """ACos"""

    def __init__(self):
        name = 'ACos'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = math.acos(ptv[i[0]])
        if o[0] in ptv:
            ptv[i[0]] = math.cos(ptv[o[0]])
        return ptv


class AbsArrow(PrimitiveArrow):
    """Abs(x)"""

    def __init__(self):
        name = 'Abs'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        constraints = []
        if 0 in output_expr:
            constraints.append(Ge(output_expr[0], 0))
        return constraints

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        if i[0] in ptv:
            ptv[o[0]] = abs(ptv[i[0]])
        return ptv


class MaxArrow(PrimitiveArrow):
    """Returns the max of x and y (i.e. x > y ? x : y) element-wise."""

    def __init__(self):
        name = 'Max'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = np.max(ptv[i[0]], ptv[i[1]])
        return ptv


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

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        if i[0] in ptv and i[1] in ptv:
            i1 = tuple(ptv[i[1]]) if isinstance(ptv[i[1]], np.ndarray) else ptv[i[1]]
            ptv[o[0]] = np.mean(ptv[i[0]], i1)
        return ptv
