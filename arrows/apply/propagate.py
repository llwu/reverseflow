""""Generic Propagation of values around a composite arrow"""
from typing import Dict, Callable, TypeVar

from arrows.port import Port
from arrows.compositearrow import CompositeArrow


T = TypeVar('T')
PortValues = Dict[Port, T]


def update_neigh(in_dict, out_dict, context, working_set):
    for port, value in in_dict.items():
        for neigh_port in context.neigh_ports(port):
            if neigh_port not in out_dict:
                out_dict[neigh_port] = value
                working_set.add(neigh_port.arrow)
            else:
                assert value == out_dict[neigh_port], "Inconsistent"


def propagate(sub_propagate: Callable,
              comp_arrow: CompositeArrow,
              port_values: PortValues,
              state=None) -> PortValues:
    """
    Propagate values around a composite arrow to determine knowns from unknowns
    The knowns should be determined by the knowns, otherwise an error throws
    Args:
        sub_propagate: an @overloaded function which propagates from each arrow
          sub_propagate(a: ArrowType, port_to_known:Dict[Port, T], state:Any)
        comp_arrow: Composite Arrow to propagate through
        port_values: port->value map for inputs to composite arrow
        state: A value of any type that is passed around during propagation
               and can be updated by sub_propagate
    Returns:
        port->value map for all ports in composite arrow
    """
    _port_values = {}
    _port_values.update(port_values)
    updated = set(comp_arrow.get_sub_arrows())
    update_neigh(port_values, _port_values, comp_arrow, updated)
    while len(updated) > 0:
        sub_arrow = updated.pop()
        sub_port_values = {port: _port_values[port]
                           for port in sub_arrow.get_ports()
                           if port in _port_values}
        new_sub_port_values = sub_propagate(sub_arrow, sub_port_values,
                                            state)
        update_neigh(new_sub_port_values, _port_values, comp_arrow, updated)
    return _port_values
