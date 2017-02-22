from arrows.primitivearrow import PrimitiveArrow
import arrows.compositearrow as compositearrows
import arrows.primitive.control_flow as cfarrows
from reverseflow.util.mapping import Bimap
from typing import Dict, List, MutableMapping, Set, Sequence
from sympy import Expr, Rel, Gt, Ne
import math
from arrows.port_attributes import (PortAttributes, port_has, ports_has,
    extract_attribute, any_port_has)
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


def add_symbt_pred(arr: "AddArrow", port_attr: PortAttributes):
    return any_port_has(arr.in_ports(), 'symbolic_tensor', port_attr)


def add_symbt_disp(arr: "AddArrow", port_attr: PortAttributes):
    ptv = extract_attribute('symbolic_tensor', port_attr)
    if arr.in_port(0) and arr.in_port(1) in ptv:
        assert False, "Figure this out"
    elif arr.in_port(0) in ptv:
        return {arr.out_port(0): {'symbolic_tensor': ptv[arr.in_port(0)]}}
    elif arr.in_port(1) in ptv:
        return {arr.out_port(1): {'symbolic_tensor': ptv[arr.in_port(1)]}}
    else:
        assert False, "why am i here"

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
            add_pred3: add_dispatch3,
            add_symbt_pred: add_symbt_disp,
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

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch
            })
        return disp


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
        return disp


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

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({shape_pred: shape_dispatch})
        return disp

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

class SquareArrow(PrimitiveArrow):
    """Square(x)"""

    def __init__(self):
        name = 'Square'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({shape_pred: shape_dispatch})
        return disp


class MaxArrow(PrimitiveArrow):
    """Returns the max of x and y (i.e. x > y ? x : y) element-wise."""

    def __init__(self):
        name = 'Max'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({shape_pred: shape_dispatch})
        return disp


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


class SquaredDifference(PrimitiveArrow):
    """ Returns (x - y)(x - y) element-wise.

    Args:
      x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      y: A `Tensor`. Must have the same type as `x`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `x`.
    """
    def __init__(self):
        name = 'SquaredDifference'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({shape_pred: shape_dispatch})
        return disp


def broadcast_pred(arr: Arrow, port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'shape', port_attr)


def broadcast_dispatch(arr: Arrow, port_attr: PortAttributes):
    """Decide output shape."""
    pts = extract_attribute('shape', port_attr)
    shapes = list(pts.values())
    shape = ()
    for s in shapes:
        if len(s) >= len(shape):
            if len(shape) > 0:
                assert s[-len(shape):] == shape, "Shapes incompatible %s %s %s" % (s, s[-len(shape):], shape)
            shape = s
    print("Broadcasting %s" % pts)
    return {port: {'shape': shape} for port in arr.out_ports()}


class BroadcastArithArrow(compositearrows.CompositeArrow):
    """
    Broadcasts the inputs.
    """

    def __init__(self, arith_arrow: PrimitiveArrow):
        name = "BroadcastArith"
        edges = Bimap()
        in_ports = []
        out_ports = arith_arrow.out_ports()
        for in_port in arith_arrow.in_ports():
            broadcast = cfarrows.BroadcastArrow()
            in_ports += broadcast.in_ports()
            edges.add(broadcast.out_ports()[0], in_port)
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            broadcast_pred: broadcast_dispatch
            })
        return disp
