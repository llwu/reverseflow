"""Compute shapes of outputs"""
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.std_arrows import *
from arrows.port import Port
from arrows.apply.propagate import propagate

from overloading import overload
from numpy import ndarray
from typing import Tuple, Dict, Sequence, Union, TypeVar, Type, Callable


Shape = Sequence[int]
PortValues = Dict[Port, Shape]


# Predicate dispatch
pred_to_dispatch = {}  # type: Dict[Type, Callable]

def get_dispatches(a: Arrow):
    return pred_to_dispatch[a.__class__]

def register_dispiatch(a: Type, pred: Callable, dispatch: Callable):
    if a in pred_to_dispatch:
        pred_to_dispatch[a] = [(pred, dispatch)]
    else:
        pred_to_dispatch[a].append([pred, dispatch])


def sub_propagate(a: Arrow, port_values: Dict[Port, Dict], state=None):
    dispatches = get_dispatches(a)
    for predicate, dispatch in dispatches.items():
        if predicate(a, port_values):
            dispatch(a, port_values)


def port_has(port_values: Dict[Port, Dict], port: Port, type: str) -> bool:
    return port in port_values and type in port_values[port]


def rank_predicate(a: Arrow, port_values: PortValues, state=None) -> bool:
    return port_has(port_values, 0, "name") and has_port_value(port_values, 1, "shape")


def rank_dispatch(a: Arrow, port_values: PortValues, state=None):
    ...

register_dispiatch(Arrow, rank_predicate, rank_dispatch)


##########################
AllSame = Union[AddArrow,
                SubArrow,
                NegArrow,
                PowArrow,
                ExpArrow,
                LogArrow,
                LogBaseArrow,
                MulArrow,
                DivArrow,
                DuplArrow,
                AddNArrow,
                CastArrow,
                AbsArrow,
                InvDuplArrow]


@overload
def constant_to_shape(x: int):
    return ()


@overload
def constant_to_shape(x: float):
    return ()


@overload
def constant_to_shape(x: ndarray):
    return x.shape


def all_same_shape(a: Arrow, port_values: PortValues, state=None):
    # All ports have the same shape
    # if any port has a known shape then propagate that to others
    known_shape = None
    for port, shape in port_values.items():
        assert shape == known_shape or known_shape is None
        known_shape = shape
    if known_shape is None:
        return {}
    return {port: known_shape for port in a.get_ports()}


@overload
def sub_propagate(a: Arrow, port_values: PortValues, state=None) -> PortValues:
    assert False, "Error, no sub_propagation for %s implemented" % a.__class__.__name__


@overload
def sub_propagate(a: SourceArrow, port_values: PortValues, state=None) -> PortValues:
    shape = constant_to_shape(a.value)
    val = Value(shape=shape, value=a.value)
    return {port: val for port in a.get_ports()}


@overload
def sub_propagate(a: AllSame, port_values: PortValues, state=None) -> PortValues:
    return all_same_shape(a, port_values)


@overload
def sub_propagate(a: RankArrow, port_values: PortValues, state=None) -> PortValues:
    return {port: () for port in a.get_out_ports()}


@overload
def sub_propagate(a: RangeArrow, port_values: PortValues, state=None) -> PortValues:
    assert False


@overload
def sub_propagate(a: ReduceMeanArrow, port_values: PortValues, state=None) -> PortValues:
    # The shape depends on the values
    if set(a.get_ports()) == set(port_values.keys()):
        ...
    # There's no way to do this without propagating the shapes

@overload
def sub_propagate(a: CompositeArrow, port_values: PortValues, state=None) -> PortValues:
    return propagate(sub_propagate, a, port_values, state=None)


def propagate_shapes(comp_arrow: CompositeArrow, port_values: PortValues):
    return propagate(sub_propagate, comp_arrow, port_values)
