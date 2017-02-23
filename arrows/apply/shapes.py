from arrows.arrow import Arrow
from arrows.port_attributes import *
from arrows.apply.constants import CONST, VAR
from arrows.util.misc import same
from overloading import overload
from numpy import ndarray
from arrows.compositearrow import *
from arrows.util.misc import *
import numpy


def shape_pred(arr: Arrow, port_attr: PortAttributes):
    """True if any of the ports have a shape"""
    return any((port_has(port, 'shape', port_attr) for port in arr.ports()))


def shape_dispatch(arr: Arrow, port_attr: PortAttributes):
    """Make all other ports the smae"""
    pts = extract_attribute('shape', port_attr)
    shapes = list(pts.values())

    # # broadcast
    # shape = ()
    # for s in shapes:
    #     if len(s) >= len(shape):
    #         if len(shape) > 0:
    #             assert s[-len(shape):] == shape, "Shapes incompatible %s %s %s" % (s, s[-len(shape):], shape)
    #         shape = s
    # return {port: {'shape': shape} for port in arr.out_ports()}
    #
    assert same(shapes), "All shapes should be the same %s" % pts
    shape = shapes[0]
    return {port: {'shape': shape} for port in arr.ports()}


def rank_predicate_shape(a: Arrow, port_values: PortAttributes, state=None) -> bool:
    assert len(a.in_ports()) == 1
    return True


def rank_dispatch_shape(a: Arrow, port_values: PortAttributes, state=None):
    assert len(a.out_ports()) == 1
    return {a.out_ports()[0] : {'shape': ()}}

# FIXME: We could get rid of these redundant predicates by just putting data
# on the port directly
def source_predicate(a: Arrow, port_attr: PortAttributes, state=None) -> bool:
    assert len(a.in_ports()) == 0
    return True


@overload
def constant_to_shape(x: int):
    return ()


@overload
def constant_to_shape(x: float):
    return ()


@overload
def constant_to_shape(x: ndarray):
    return x.shape


@overload
def constant_to_shape(x: numpy.number):
    return x.shape


def source_dispatch(a: Arrow, port_values: PortAttributes, state=None):
    assert len(a.out_ports()) == 1
    return {a.out_ports()[0]: {'shape': constant_to_shape(a.value),
                                   'value': a.value,
                                   'constant': CONST}}
