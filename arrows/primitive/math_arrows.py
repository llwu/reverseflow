from arrows.primitivearrow import PrimitiveArrow
from typing import Dict, List, MutableMapping, Set, Sequence
from sympy import Expr, Rel, Gt, Ne
import math
from arrows.port_attributes import (PortAttributes, port_has, ports_has,
    extract_attribute)
from arrows.port import Port
from arrows.arrow import Arrow
from arrows.apply.shapes import *
from arrows.apply.constants import constant_pred, constant_dispatch

def same_shape(shape): return shape


def add_pred1(arr: "AddArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def add_dispatch1(arr: "AddArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {o[0] : {'value': ptv[i[0]] + ptv[i[1]]}}

def add_pred2(arr: "AddArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[0], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def add_dispatch2(arr: "AddArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[1] : {'value': ptv[o[0]] - ptv[i[0]]}}

def add_pred3(arr: "AddArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[1], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def add_dispatch3(arr: "AddArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[0] : {'value': ptv[o[0]] - ptv[i[1]]}}


class AddArrow(PrimitiveArrow):
    """Addition"""

    def __init__(self):
        name = 'Add'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            add_pred1: add_dispatch1,
            add_pred2: add_dispatch2,
            add_pred3: add_dispatch3
            })
        return disp


def sub_pred1(arr: "SubArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def sub_dispatch1(arr: "SubArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {o[0] : {'value': ptv[i[0]] - ptv[i[1]]}}

def sub_pred2(arr: "SubArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[0], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def sub_dispatch2(arr: "SubArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[1] : {'value': ptv[i[0]] - ptv[o[0]]}}

def sub_pred3(arr: "SubArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[1], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def sub_dispatch3(arr: "SubArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[0] : {'value': ptv[o[0]] + ptv[i[1]]}}


class SubArrow(PrimitiveArrow):
    """Subtraction. Out[1] = In[0] - In[1]"""

    def __init__(self):
        name = 'Sub'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            sub_pred1: sub_dispatch1,
            sub_pred2: sub_dispatch2,
            sub_pred3: sub_dispatch3
            })
        return disp


def mul_pred1(arr: "MulArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def mul_dispatch1(arr: "MulArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {o[0] : {'value': ptv[i[0]] * ptv[i[1]]}}

def mul_pred2(arr: "MulArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[0], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def mul_dispatch2(arr: "MulArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[1] : {'value': ptv[o[0]] / ptv[i[0]]}}

def mul_pred3(arr: "MulArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[1], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def mul_dispatch3(arr: "MulArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[0] : {'value': ptv[o[0]] / ptv[i[1]]}}


class MulArrow(PrimitiveArrow):
    """Multiplication"""

    def __init__(self):
        name = 'Mul'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            mul_pred1: mul_dispatch1,
            mul_pred2: mul_dispatch2,
            mul_pred3: mul_dispatch3
            })
        return disp


def div_pred1(arr: "DivArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def div_dispatch1(arr: "DivArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {o[0] : {'value': ptv[i[0]] / ptv[i[1]]}}

def div_pred2(arr: "DivArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[0], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def div_dispatch2(arr: "DivArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[1] : {'value': ptv[i[0]] / ptv[o[0]]}}

def div_pred3(arr: "DivArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[1], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def div_dispatch3(arr: "DivArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[0] : {'value': ptv[o[0]] * ptv[i[1]]}}


class DivArrow(PrimitiveArrow):
    """Division"""

    def __init__(self):
        name = 'Div'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            div_pred1: div_dispatch1,
            div_pred2: div_dispatch2,
            div_pred3: div_dispatch3
            })
        return disp

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        constraints = []
        if 1 in input_expr:
            constraints.append(Ne(input_expr[1], 0))
        return constraints


def pow_pred1(arr: "PowArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def pow_dispatch1(arr: "PowArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {o[0] : {'value': ptv[i[0]] ** ptv[i[1]]}}

def pow_pred2(arr: "PowArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[0], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def pow_dispatch2(arr: "PowArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[1] : {'value': math.log(ptv[i[0]], ptv[o[0]])}}

def pow_pred3(arr: "PowArrow", port_attr: PortAttributes):
    ports = [arr.in_ports()[1], arr.out_ports()[0]]
    return ports_has(ports, 'value', port_attr)

def pow_dispatch3(arr: "PowArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {i[0] : {'value': ptv[o[0]] ** (1.0 / ptv[i[1]])}}


class PowArrow(PrimitiveArrow):
    """Computes the power of one value to another.

    Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
    corresponding elements in `x` and `y`. For example"""

    def __init__(self):
        name = 'Pow'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            pow_pred1: pow_dispatch1,
            pow_pred2: pow_dispatch2,
            pow_pred3: pow_dispatch3
            })
        return disp

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

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch
            })
        return disp


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

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch
            })
        return disp


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

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch
            })
        return disp



class ASinArrow(PrimitiveArrow):
    """ASin"""

    def __init__(self):
        name = 'ASin'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch
            })
        return disp


class CosArrow(PrimitiveArrow):
    """Cos"""

    def __init__(self):
        name = 'Cos'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch
            })

class FloorDivArrow(PrimitiveArrow):
    """Floor Division"""

    def __init__(self):
        name = 'FloorDiv'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)


class ClipArrow(PrimitiveArrow):
    """Clip"""

    def __init__(self):
        name = 'Clip'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

class ACosArrow(PrimitiveArrow):
    """ACos"""

    def __init__(self):
        name = 'ACos'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        return {constant_pred: constant_dispatch,
                shape_pred: shape_dispatch}


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
