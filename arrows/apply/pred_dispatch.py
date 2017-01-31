from arrows.arrow import Arrow
from arrows.port_attributes import (PortAttributes, port_has, ports_has,
    extract_attribute)
from overloading import overload
from numpy import ndarray

def constant_pred(arr: Arrow, port_attr: PortAttributes):
    return all((port_has(port, 'constant', port_attr) for port in arr.get_in_ports()))


def constant_dispatch(arr: Arrow, port_attr: PortAttributes):
    assert False


def shape_pred(arr: Arrow, port_attr: PortAttributes):
    return any((port_has(port, 'shape', port_attr) for port in arr.get_ports()))


def shape_dispatch(arr: Arrow, port_attr: PortAttributes):
    pts = extract_attribute('shape', port_attr)
    shape = list(pts.values())[0]
    return {port: {'shape': shape} for port in arr.get_ports()}


def rank_predicate_shape(a: Arrow, port_values: PortAttributes, state=None) -> bool:
    assert len(a.get_in_ports()) == 1
    return True


def rank_dispatch_shape(a: Arrow, port_values: PortAttributes, state=None):
    assert len(a.get_out_ports()) == 1
    return {a.get_out_ports()[0] : {'shape': ()}}

# FIXME: We could get rid of these redundant predicates by just putting data
# on the port directly
def source_predicate(a: Arrow, port_values: PortAttributes, state=None) -> bool:
    assert len(a.get_in_ports()) == 0
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


def source_dispatch(a: Arrow, port_values: PortAttributes, state=None):
    assert len(a.get_out_ports()) == 1
    return {a.get_out_ports()[0]: {'shape': constant_to_shape(a.value),
                                    'value': a.value}}
