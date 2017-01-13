""""Generic Propagation of values around a composite arrow"""
from arrows.arrow import Arrow
from arrows.port import Port
from arrows.compositearrow import CompositeArrow
from typing import List, Dict, Callable, TypeVar

T = TypeVar['T']
PortValues = Dict[Port, T]


def propagate(sub_propagate: Callable,
              comp_arrow: CompositeArrow,
              port_values: PortValues,
              state: None) -> PortValues:
    """
    Propagate values around a composite arrow to determine knowns from unknowns
    The knowns should be determined by the knowns, otherwise an error throws
    Args:
        sub_propagate: an @overloaded function which propagates from each arrow
          sub_propagate(a: ArrowType, port_to_known:Dict[Port, T], state:Any)
        comp_arrow: Composite Arrow to execute
        knowns: Portlist of inputs to composite arrow
        State: A value of any type that is passed around during propagation
               and can be updated by sub_propagate
    Returns:
        List of outputs
    """
    # TODO: This could me made more efficient
    # 1. If the inputs to a sub_arrow don't change then we don't need to call
    #    sub_propagate on it (but we also need to call it at least once)

    # Don't want to modify input,
    _port_values = {}
    _port_values.update(port_values)

    arrow_to_num_unresolved = {sub_arrow: sub_arrow.num_ports()
                               for sub_arrow in comp_arrow.get_sub_arrows()}

    # FIXME: Should terminate when all ports are resolved
    while True:
        # There are two alternating phases of propagation
        # Phase 1: Query any sub_arrows to see if they have any update on their ports
        for sub_arrow, num_unresolved in arrow_to_num_unresolved.items():
            if num_unresolved > 0:
                sub_port_values = {port: value
                                   for port, value in _port_values.items()
                                   if port.arrow is sub_arrow}
                new_sub_port_values = sub_propagate(sub_arrow, sub_port_values,
                                                    state)
                _port_values.update(new_sub_port_values)
                arrow_to_num_unresolved[sub_arrow] = sub_arrow.num_ports() - len(new_sub_port_values)


        # Phase 2: propagate values from each port to any connected port
        edge_prop_occured = False
        for port, value in _port_values.items():
            for neigh_port in comp_arrow.neigh_ports(port):
                if neigh_port not in _port_values:
                    _port_values[neigh_port] = value
                    edge_prop_occured = True
                else:
                    assert value == _port_values[neigh_port], "Inconsistent"
