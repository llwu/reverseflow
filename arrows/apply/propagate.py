""""Generic Propagation of values around a composite arrow"""

from copy import copy
from typing import Dict, Callable, TypeVar

from arrows.port import Port
from arrows.compositearrow import CompositeArrow
from arrows.port_attributes import get_port_attributes


T = TypeVar('T')
PortValues = Dict[Port, T]


def update_neigh(in_dict, out_dict, context, working_set):
    for port, value in in_dict.items():
        for neigh_port in context.neigh_ports(port):
            if (neigh_port.arrow != context) and (neigh_port not in out_dict or out_dict[neigh_port] != value):
                working_set.add(neigh_port.arrow)
            out_dict[neigh_port] = copy(value)
        out_dict[port] = copy(value)


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
    # update port_values with values stored on port
    for sub_arrow in comp_arrow.get_all_arrows():
        for port in sub_arrow.get_ports():
            attributes = get_port_attributes(port)
            _port_values[port] = attributes

    updated = set(comp_arrow.get_sub_arrows())
    update_neigh(port_values, _port_values, comp_arrow, updated)
    while len(updated) > 0:
        print(len(updated))
        sub_arrow = updated.pop()
        sub_port_values = {port: _port_values[port]
                           for port in sub_arrow.get_ports()
                           if port in _port_values}
        new_sub_port_values = sub_propagate(sub_arrow, sub_port_values, state)
        update_neigh(new_sub_port_values, _port_values, comp_arrow, updated)
    return _port_values

# 1. What is the correct termination condition
# 2. Is this going to take in a sub_propagate function or not
# - we might not aant to always propagate everything
# 3. lwu put in this kind of inverse propagation to handle the case of value propagation
# 4. How to reuse that from symbolic apply
