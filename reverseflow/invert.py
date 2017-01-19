"""Parametric Inversion"""
from arrows import Arrow, OutPort, Port, InPort
from arrows.compositearrow import CompositeArrow, RelEdgeMap, is_projecting, is_receiving
from arrows.apply.constants import PortValues, CONST, VAR
from arrows.std_arrows import *
from arrows.port_attributes import *
from arrows.apply.constants import propagate_constants
from reverseflow.defaults import default_dispatch
from typing import Dict, Callable, Set, Tuple, Type, TypeVar, Any
from overloading import overload

PortMap = Dict[int, int]
U = TypeVar('U', bound=Arrow)
# DispatchType = Dict[Any, Callable]
# FIXME: Make this type tighter (and work)
DispatchType = Any

@overload
def invert_sub_arrow(arrow: Arrow,
                     port_values: PortValues,
                     dispatch: DispatchType):
    invert_f = dispatch[arrow.__class__]
    return invert_f(arrow, port_values)

@overload
def invert_sub_arrow(source_arrow: SourceArrow,
                     port_values: PortValues,
                     dispatch: DispatchType):
    return source_arrow, {0: 0}

@overload
def invert_sub_arrow(comp_arrow: CompositeArrow,
                     port_values: PortValues,
                     dispatch: DispatchType):
    return inner_invert(comp_arrow, port_values, dispatch)

@overload
def link(out_port1: OutPort, in_port2: InPort, comp_arrow: CompositeArrow):
    comp_arrow.add_edge(out_port1, in_port2)

@overload
def link(in_port1: InPort, out_port2: OutPort, comp_arrow: CompositeArrow):
    comp_arrow.add_edge(out_port2, in_port1)

@overload
def link(in_port1: InPort, in_port2: InPort, comp_arrow: CompositeArrow):
    assert False, "Can't connect in_port to in_port"

@overload
def link(out_port1: OutPort, out_port2: OutPort, comp_arrow: CompositeArrow):
    assert False, "Can't connect out_port to out_port"


def get_inv_port(port: Port,
                 arrow_to_port_map: [Arrow, PortMap],
                 arrow_to_inv: Dict[Arrow, Arrow]):
    arrow = port.arrow
    inv = arrow_to_inv[arrow]
    port_map = arrow_to_port_map[arrow]
    inv_port_index = port_map[port.index]
    inv_port = inv.ports[inv_port_index]
    return inv_port

def inner_invert(comp_arrow: CompositeArrow,
                 port_values: PortValues,
                 dispatch: Dict[Arrow, Callable]):
    """Construct a parametric inverse of arrow
    Args:
        arrow: Arrow to invert
        dispatch: Dict mapping arrow class to invert function
    Returns:
        A (approximate) parametric inverse of `arrow`"""
    # Empty compositon for inverse
    inv_comp_arrow = CompositeArrow(name="%s_inv" % comp_arrow.name)

    # Add a port on inverse arrow for every port on arrow
    for port in comp_arrow.get_ports():
        inv_port = inv_comp_arrow.add_port()
        if is_in_port(port):
            make_out_port(inv_port)
        elif is_out_port(port):
            make_in_port(inv_port)

    # invert each sub_arrow
    arrow_to_inv = dict()
    arrow_to_port_map = dict()
    for sub_arrow in comp_arrow.get_sub_arrows():
        inv_sub_arrow, port_map = invert_sub_arrow(sub_arrow,
                                                   port_values,
                                                   dispatch)
        arrow_to_port_map[sub_arrow] = port_map
        arrow_to_inv[sub_arrow] = inv_sub_arrow

    # Add comp_arrow to inv
    arrow_to_inv[comp_arrow] = inv_comp_arrow
    comp_port_map = {i: i for i in range(comp_arrow.num_ports())}
    arrow_to_port_map[comp_arrow] = comp_port_map

    # Then, rewire up all the edges
    for out_port, in_port in comp_arrow.edges.items():
        left_inv_port = get_inv_port(out_port, arrow_to_port_map, arrow_to_inv)
        right_inv_port = get_inv_port(in_port, arrow_to_port_map, arrow_to_inv)
        if left_inv_port.arrow is inv_comp_arrow and right_inv_port.arrow is inv_comp_arrow:
            assert is_out_port(left_inv_port)
            assert is_in_port(right_inv_port)
            inv_comp_arrow.add_edge(right_inv_port, left_inv_port)
        elif left_inv_port.arrow is inv_comp_arrow:
            assert is_out_port(left_inv_port)
            assert is_out_port(right_inv_port)
            inv_comp_arrow.add_edge(right_inv_port, left_inv_port)
        elif right_inv_port.arrow is inv_comp_arrow:
            assert is_in_port(right_inv_port)
            assert is_in_port(left_inv_port)
            inv_comp_arrow.add_edge(right_inv_port, left_inv_port)
        else:
            if is_out_port(left_inv_port) and is_in_port(right_inv_port):
                inv_comp_arrow.add_edge(left_inv_port, right_inv_port)
            elif is_in_port(left_inv_port) and is_out_port(right_inv_port):
                inv_comp_arrow.add_edge(right_inv_port, left_inv_port)
            else:
                assert False, "One must be projecting and one receiving"

    # Craete new ports on inverse compositions for parametric and error ports
    for sub_arrow in inv_comp_arrow.get_sub_arrows():
        for port in sub_arrow.get_ports():
            if is_param_port(port):
                assert port not in inv_comp_arrow.edges.keys()
                assert port not in inv_comp_arrow.edges.values()
                param_port = inv_comp_arrow.add_port()
                inv_comp_arrow.add_edge(param_port, port)
                make_in_port(param_port)
                make_param_port(param_port)
            elif is_error_port(port):
                assert port not in inv_comp_arrow.edges.keys()
                assert port not in inv_comp_arrow.edges.values()
                error_port = inv_comp_arrow.add_port()
                inv_comp_arrow.add_edge(port, error_port)
                make_out_port(error_port)
                make_error_port(error_port)

    return inv_comp_arrow



def invert(comp_arrow: CompositeArrow,
           dispatch: Dict[Arrow, Callable]=default_dispatch) -> Arrow:
    """Construct a parametric inverse of comp_arrow
    Args:
        comp_arrow: Arrow to invert
        dispatch: Dict mapping comp_arrow class to invert function
    Returns:
        A (approximate) parametric inverse of `comp_arrow`"""
    port_values = propagate_constants(comp_arrow)
    comp_arrow.duplify()
    return inner_invert(comp_arrow, port_values, dispatch)
