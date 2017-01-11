"""Parametric Inversion"""

from typing import Dict, Callable, Set, Tuple, Type, TypeVar, Any

from arrows import Arrow, OutPort, Port, InPort
from arrows.compositearrow import CompositeArrow, RelEdgeMap
from arrows.std_arrows import *
from reverseflow.defaults import default_dispatch
from arrows.marking import mark_source
from reverseflow.util.mapping import Bimap, Relation
from overloading import overload

PortMap = Dict[int, int]
U = TypeVar('U', bound=Arrow)
# DispatchType = Dict[Any, Callable]
# FIXME: Make this type tighter (and work)
DispatchType = Any

@overload
def invert_sub_arrow(arrow: Arrow,
                     const_in_ports,
                     const_out_ports,
                     dispatch: DispatchType):
    invert_f = dispatch[arrow.__class__]
    return invert_f(arrow, const_in_ports)

@overload
def invert_sub_arrow(source_arrow: SourceArrow,
                     const_in_ports: Set[InPort],
                     const_out_ports: Set[OutPort],
                     dispatch: DispatchType):
    return source_arrow, {0: 0}

@overload
def invert_sub_arrow(comp_arrow: CompositeArrow,
                     const_in_ports: Set[InPort],
                     const_out_ports: Set[OutPort],
                     dispatch: DispatchType):
    return inner_invert(comp_arrow, const_in_ports, const_out_ports, dispatch)

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
                 const_in_ports: Set[InPort],
                 const_out_ports: Set[OutPort],
                 dispatch: Dict[Arrow, Callable]):
    """Construct a parametric inverse of arrow
    Args:
        arrow: Arrow to invert
        dispatch: Dict mapping arrow class to invert function
    Returns:
        A (approximate) parametric inverse of `arrow`"""
    # Empty compositon for inverse
    inv_comp_arrow = CompositeArrow(name="%s_inv" % comp_arrow.name)

    # invert each sub_arrow
    arrow_to_inv = dict()
    arrow_to_port_map = dict()
    for sub_arrow in comp_arrow.get_sub_arrows():
        inv_sub_arrow, port_map = invert_sub_arrow(sub_arrow,
                                                   const_in_ports,
                                                   const_out_ports,
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
        link(left_inv_port, right_inv_port, inv_comp_arrow)

    return inv_comp_arrow

## TODO:
## How to switch the port type?
## Inner Edge will fail, because it will be from in_port to in_port.
## - The solution, I think, is to have edges just be maps from Port to Port.
## And have whether a port is an out_port or an in_port (from the perspective) of the outside as a 'port property'
## I think this approach will simplify other algorithms too.


def invert(arrow: CompositeArrow,
           dispatch: Dict[Arrow, Callable]=default_dispatch) -> Arrow:
    """Construct a parametric inverse of arrow
    Args:
        arrow: Arrow to invert
        dispatch: Dict mapping arrow class to invert function
    Returns:
        A (approximate) parametric inverse of `arrow`"""
    const_in_ports, const_out_ports = mark_source(arrow)
    return inner_invert(arrow, const_in_ports, const_out_ports, dispatch)
