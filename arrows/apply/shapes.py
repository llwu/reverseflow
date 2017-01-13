"""Compute shapes of outputs"""
from typing import Tuple, Dict, Sequence

from overloading import overload
from numpy import ndarray

from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.std_arrows import *
from arrows.port import Port
from arrows.apply.propagate import propagate
from reverseflow.inv_primitives.inv_control_flow_arrows import *
from reverseflow.inv_primitives.inv_math_arrows import *

ShapeList = Sequence
PortValues = Dict


@overload
def constant_to_shape(x: int):
    return ()


@overload
def constant_to_shape(x: float):
    return ()


@overload
def constant_to_shape(x: ndarray):
    return x.shape


def generic_prop(a: Arrow, port_to_known: PortValues, state=None):
    known_shape = None
    for port, shape in port_to_known.items():
        assert shape == known_shape or known_shape is None
        known_shape = shape
    if known_shape is None:
        return {}
    return {port: known_shape for port in a.get_ports()}


@overload
def sub_propagate(a: Arrow, port_to_known: PortValues, state=None) -> ShapeList:
    assert False, "Error, no sub_propagation for %s implemented" % a.__class__.__name__


@overload
def sub_propagate(a: SourceArrow, port_to_known: PortValues, state=None) -> ShapeList:
    known_shape = constant_to_shape(a.value)
    return {port: known_shape for port in a.get_ports()}


@overload
def sub_propagate(a: AddArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: SubArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: NegArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: PowArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: ExpArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: LogArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: LogBaseArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: MulArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: DivArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)


@overload
def sub_propagate(a: DuplArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)

@overload
def sub_propagate(a: AddNArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)

@overload
def sub_propagate(a: CastArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)

@overload
def sub_propagate(a: AbsArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)

@overload
def sub_propagate(a: InvDuplArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return generic_prop(a, port_to_known)

@overload
def sub_propagate(a: RankArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return {port: () for port in a.get_out_ports()}

@overload
def sub_propagate(a: RangeArrow, port_to_known: PortValues, state=None) -> ShapeList:
    assert False

@overload
def sub_propagate(a: ReduceMeanArrow, port_to_known: PortValues, state=None) -> ShapeList:
    assert False


@overload
def sub_propagate(a: CompositeArrow, port_to_known: PortValues, state=None) -> ShapeList:
    return propagate(sub_propagate, a, port_to_known, state=None)


def propagate_shapes(comp_arrow: CompositeArrow, port_to_known: PortValues):
    return propagate(sub_propagate, comp_arrow, port_to_known)
