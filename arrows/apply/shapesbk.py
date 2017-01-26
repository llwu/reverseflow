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
PortValues = Dict


pred_to_dispatch = {}


def get_dispatches(a: Arrow):
    global pred_to_dispatch
    if pred_to_dispatch == {}:
        pred_to_dispatch = register_dispatches()
    return pred_to_dispatch[a.__class__]


def register_dispatch(out_dict: Dict, a: Type, pred: Callable, dispatch: Callable):
    if a not in pred_to_dispatch:
        out_dict[a] = [(pred, dispatch)]
    else:
        out_dict[a].append((pred, dispatch))


def eval_predicate(a: Arrow, port_values: PortValues, state=None) -> bool:
    return all([port_has(port_values, port, 'value') for port in a.get_in_ports()])


def eval_dispatch(a: Arrow, port_values: PortValues, state=None):
    flattened = dict([(port, value['value']) for port, value in port_values.items() if 'value' in value])
    for port, value in a.eval(flattened).items():
        port_values[port]['value'] = value


@overload
def sub_propagate(a: Arrow, port_values: PortValues, state=None):
    dispatches = get_dispatches(a)
    for predicate, dispatch in dispatches:
        if predicate(a, port_values):
            dispatch(a, port_values)
    if eval_predicate(a, port_values):
        eval_dispatch(a, port_values)
    return port_values


def port_has(port_values: Dict[Port, Dict], port: Port, type: str) -> bool:
    return port in port_values and type in port_values[port]


def rank_predicate_shape(a: Arrow, port_values: PortValues, state=None) -> bool:
    assert len(a.get_in_ports()) == 1
    return True


def rank_dispatch_shape(a: Arrow, port_values: PortValues, state=None):
    assert len(a.get_out_ports()) == 1
    port_values[a.get_out_ports()[0]]['shape'] = ()


def source_predicate(a: Arrow, port_values: PortValues, state=None) -> bool:
    assert len(a.get_in_ports()) == 0
    return True


def source_dispatch(a: Arrow, port_values: PortValues, state=None):
    assert len(a.get_out_ports()) == 1
    port_values[a.get_out_ports()[0]]['shape'] = constant_to_shape(a.value)
    port_values[a.get_out_ports()[0]]['value'] = a.value


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
            value['shape'] = constant_to_shape(value['value'])
        if 'shape' not in value:
            continue
        assert value['shape'] == known_shape or known_shape is None
        known_shape = value['shape']
    return known_shape is not None


def generic_dispatch(a: Arrow, port_values: PortValues, state=None):
    known_shape = None
    for port, value in port_values.items():
        if 'shape' not in value:
            continue
        known_shape = value['shape']
        break
    for port in a.get_ports():
        if port not in port_values:
            port_values[port] = {}
        port_values[port]['shape'] = known_shape


def register_dispatches():
    pred_to_dispatch = {}
    register_dispatch(pred_to_dispatch, RankArrow, rank_predicate_shape, rank_dispatch_shape)
    register_dispatch(pred_to_dispatch, SourceArrow, source_predicate, source_dispatch)
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
        register_dispatch(pred_to_dispatch, arrow_type, generic_predicate, generic_dispatch)
    return pred_to_dispatch


@overload
def sub_propagate(a: CompositeArrow, port_values: PortValues, state=None) -> PortValues:
    return propagate(sub_propagate, a, port_values, state=None)


def propagate_shapes(comp_arrow: CompositeArrow, port_values: PortValues):
    return propagate(sub_propagate, comp_arrow, port_values)
