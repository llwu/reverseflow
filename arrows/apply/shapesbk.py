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

def register_dispatch(a: Type, pred: Callable, dispatch: Callable):
    if a in pred_to_dispatch:
        pred_to_dispatch[a] = [(pred, dispatch)]
    else:
        pred_to_dispatch[a].append((pred, dispatch))


def eval_predicate(a: Arrow, port_values: PortValues, state=None) -> bool:
    return all([port_has(port_values, port, 'value') for port in a.get_in_ports()])


def eval_dispatch(a: Arrow, port_values: PortValues, state=None):
    flattened = dict([(port, value['value']) for port, value in port_values.items()])
    for port, value in a.eval(flattened):
        port_values[port]['value'] = value


def sub_propagate(a: Arrow, port_values: Dict[Port, Dict], state=None):
    dispatches = get_dispatches(a)
    for predicate, dispatch in dispatches.items():
        if predicate(a, port_values):
            dispatch(a, port_values)
    if eval_predicate(a, port_values):
        eval_dispatch(a, port_values)


def port_has(port_values: Dict[Port, Dict], port: Port, type: str) -> bool:
    return port in port_values and type in port_values[port]


def rank_predicate_shape(a: Arrow, port_values: PortValues, state=None) -> bool:
    assert len(a.get_in_ports()) == 1
    return True


def rank_dispatch_shape(a: Arrow, port_values: PortValues, state=None):
    assert len(a.get_out_ports()) == 1
    port_values[a.get_out_ports()[0]]['shape'] = ()


register_dispatch(RankArrow, rank_predicate_shape, rank_dispatch_shape)


@overload
def constant_to_shape(x: int):
    return ()


@overload
def constant_to_shape(x: float):
    return ()


@overload
def constant_to_shape(x: ndarray):
    return x.shape


def generic_predicate(a: Arrow, port_values: PortValues, state=None):
    known_shape = None
    for port, value in port_values.items():
        if 'value' in value:
            value['shape'] = constant_to_shape[value['value']]
        if 'shape' not in value:
            continue
        assert value['shape'] == known_shape or known_shape is None
        known_shape = value['shape']
    return known_shape is not None


def generic_dispatch(a: Arrow, port_to_known: PortValues, state=None):
    known_shape = None
    for port, value in port_values.items():
        if 'shape' not in value:
            continue
        known_shape = value['shape']
        break
    for port, value in port_values.items():
        value['shape'] = known_shape


for arrow_type in [AddArrow,
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
                   InvDuplArrow]:
    register_dispatch(arrow_type, generic_predicate, generic_dispatch)

# @overload
# def constant_to_shape(x: int):
#     return ()


# @overload
# def constant_to_shape(x: float):
#     return ()


# @overload
# def constant_to_shape(x: ndarray):
#     return x.shape


# def all_same_shape(a: Arrow, port_values: PortValues, state=None):
#     # All ports have the same shape
#     # if any port has a known shape then propagate that to others
#     known_shape = None
#     for port, shape in port_values.items():
#         assert constant_to_shape(shape) == known_shape or known_shape is None
#         known_shape = shape
#     if known_shape is None:
#         return {}
#     return {port: known_shape for port in a.get_ports()}


# @overload
# def sub_propagate(a: Arrow, port_values: PortValues, state=None) -> PortValues:
#     assert False, "Error, no sub_propagation for %s implemented" % a.__class__.__name__


# @overload
# def sub_propagate(a: SourceArrow, port_values: PortValues, state=None) -> PortValues:
#     shape = constant_to_shape(a.value)
#     val = Value(shape=shape, value=a.value)
#     return {port: val for port in a.get_ports()}


# @overload
# def sub_propagate(a: AllSame, port_values: PortValues, state=None) -> PortValues:
#     return all_same_shape(a, port_values)


# @overload
# def sub_propagate(a: RankArrow, port_values: PortValues, state=None) -> PortValues:
#     return {port: () for port in a.get_out_ports()}


# @overload
# def sub_propagate(a: RangeArrow, port_values: PortValues, state=None) -> PortValues:
#     assert False


# @overload
# def sub_propagate(a: ReduceMeanArrow, port_values: PortValues, state=None) -> PortValues:
#     # The shape depends on the values
#     if set(a.get_ports()) == set(port_values.keys()):
#         ...
#     # There's no way to do this without propagating the shapes

@overload
def sub_propagate(a: CompositeArrow, port_values: PortValues, state=None) -> PortValues:
    return propagate(sub_propagate, a, port_values, state=None)


def propagate_shapes(comp_arrow: CompositeArrow, port_values: PortValues):
    return propagate(sub_propagate, comp_arrow, port_values)
