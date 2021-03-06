"""These are arrows for control flow of input"""
from typing import List, MutableMapping, Set, Dict

import numpy as np
from sympy import Expr, Eq, Rel

from arrows.primitivearrow import PrimitiveArrow
from arrows.apply.shapes import *
from arrows.apply.constants import constant_pred, constant_dispatch, CONST

closure_available = set([
    "AddArrow",
    "SubArrow",
    "MulArrow",
    "DivArrow",
    "ExpArrow",
    "NegArrow",
    "CosArrow",
    "SinArrow",
#    "GatherArrow"
    ])

def dupl_pred(arr: "DuplArrow", port_attr: PortAttributes):
    for port in arr.ports():
        if port in port_attr and 'value' in port_attr[port]:
            return True
    return False

def dupl_disp(arr: "DuplArrow", port_attr: PortAttributes):
    known_value = None
    for port in arr.ports():
        if port in port_attr and 'value' in port_attr[port]:
            known_value = port_attr[port]['value']
            break
    return {port: {'value': known_value} for port in arr.ports()}

# def closure_pred(arr: "DuplArrow", port_attr: PortAttributes):
#     return hasattr(arr, 'topo_order')

# def closure_disp(arr: "DuplArrow", port_attr: PortAttributes):
#     if arr.parent is None:
#         return {}
#     o = arr.out_ports()
#     n = 0
#     const_dict = {}
#     neighs_dict = {}
#     neigh_list = []
#     for out_port in o:
#         neighs_dict[out_port] = len(out_port.arrow.parent.neigh_ports(out_port))
#         for neigh in out_port.arrow.parent.neigh_ports(out_port):
#             neigh_list.append((neigh.arrow, neigh.arrow.get_topo_order(), out_port))
#     print(neighs_dict)
#     print(neigh_list)
#     for arrow, _, out_port in sorted(neigh_list, key=lambda x:x[1]):
#         if arrow.__class__.__name__ not in closure_available:
#             break
#         else:
#             neighs_dict[out_port] -= 1
#             if neighs_dict[out_port] == 0:
#                 const_dict[out_port] = {'constant': CONST}
#                 n += 1
#                 if n >= len(o) - 1:
#                     break
#     print(const_dict)
#     return const_dict


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
            for j in output_expr.keys():
                if i != j:
                    constraints.append(Eq(output_expr[i], output_expr[j]))
        return constraints

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            dupl_pred: dupl_disp,
            # closure_pred: closure_disp
            })
        return disp


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

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            dupl_pred: dupl_disp
            })
        return disp


class IdentityArrow(PrimitiveArrow):
    """
    Identity input
    f(x) = x
    """

    def __init__(self) -> None:
        name = 'Identity'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            shape_pred: shape_dispatch,
            dupl_pred: dupl_disp
            })
        return disp


def ig_in_value_pred(arr: "IgnoreInputArrow", port_attr: PortAttributes):
    """Fire if second value present, dont care about first"""
    return arr.in_port(1) in port_attr and 'value' in port_attr[arr.in_port(1)]


def ig_in_shape_pred(arr: "IgnoreInputArrow", port_attr: PortAttributes):
    """Fire if second value present, dont care about first"""
    return arr.in_port(1) in port_attr and 'shape' in port_attr[arr.in_port(1)]


def ig_in_value_disp(arr: "IgnoreInputArrow", port_attr: PortAttributes):
    """Just propagate value"""
    value = port_attr[arr.in_port(1)]['value']
    return {arr.out_port(0): {'value': value}}


def ig_in_shape_disp(arr: "IgnoreInputArrow", port_attr: PortAttributes):
    """Just propagate shape"""
    shape = port_attr[arr.in_port(1)]['shape']
    return {arr.out_port(0): {'shape': shape}}


class IgnoreInputArrow(PrimitiveArrow):
    """
    Just ignores first input and passess through second
    f(x, y) = y
    """

    def __init__(self):
      super().__init__(n_in_ports=2, n_out_ports=1, name="IgnoreInput")

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            ig_in_value_pred: ig_in_value_disp,
            ig_in_shape_pred: ig_in_shape_disp,
            })
        return disp


class IfArrow(PrimitiveArrow):
    """
    IfArrow with 3 inputs: input[0] is a boolean indicating which one
    of input[1] (True), input[2] (False) will be propogated to the output.
    """

    def __init__(self) -> None:
        name = 'If'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> List[Rel]:
        assert 0 in input_expr
        constraints = []
        if input_expr[0]:
            constraints.append(Eq(output_expr[0], input_expr[1]))
        else:
            constraints.append(Eq(output_expr[0], input_expr[2]))
        return constraints

# FIXME: Why is this here?
class GreaterArrow(PrimitiveArrow):
    """
    Bool valued arrow for the operation '>':
    output[0] = (input[0] > input[1])
    """

    def __init__(self) -> None:
        name = 'Greater'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> List[Rel]:
        assert 0 in output_expr
        constraints = []
        if output_expr[0]:
            constraints.append(Gt(input_expr[0], input_expr[1]))
        else:
            constraints.append(Ge(input_expr[1], input_expr[0]))
        return constraints


def broadcast_bwd_pred(arr: "BroadcastArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'shape', port_attr) and ports_has(arr.out_ports(), 'value', port_attr)

def broadcast_bwd_disp(arr: "BroadcastArrow", port_attr: PortAttributes):
    out_val = port_attr[arr.out_port(0)]['value']
    in_shape = port_attr[arr.in_port(0)]['shape']
    out_shape = constant_to_shape(out_val)
    if len(out_shape) <= len(in_shape):
        return {arr.in_port(0): {'value': out_val}}
    else:
        ext_shape = out_shape[:len(out_shape) - len(in_shape)]
        idx = tuple(0 for i in ext_shape)
        return {arr.in_port(0): {'value': out_val[idx]}}

def broadcast_fwd_pred(arr: "BroadcastArrow", port_attr: PortAttributes):
    return ports_has(arr.out_ports(), 'shape', port_attr) and ports_has(arr.in_ports(), 'value', port_attr)

def broadcast_fwd_disp(arr: "BroadcastArrow", port_attr: PortAttributes):
    in_val = port_attr[arr.in_port(0)]['value']
    out_shape = port_attr[arr.out_port(0)]['shape']
    return {arr.out_port(0): {'value': np.broadcast_to(in_val, out_shape)}}

## FIXME Add assertion to test that shapes are broadcast compatibl
class BroadcastArrow(PrimitiveArrow):
    """
    Broadcast an op
    """

    def __init__(self):
        name = 'Broadcast'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            broadcast_bwd_pred: broadcast_bwd_disp,
            broadcast_fwd_pred: broadcast_fwd_disp
            })
        return disp

## FIXME Add assertion to test that shapes are broadcast compatibl
class InvBroadcastArrow(PrimitiveArrow):
    """
    Broadcast an op
    """

    def __init__(self, in_shape, out_shape):
        name = 'InvBroadcast'
        self.ext_shape = out_shape[:len(out_shape) - len(in_shape)]
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    # def get_dispatches(self):
    #     disp = super().get_dispatches()
    #     disp.update({
    #         broadcast_bwd_pred: broadcast_bwd_disp,
    #         })
    #     return disp
