""""Generic Propagation of values around a composite arrow"""
from arrows.port import Port
from arrows.compositearrow import CompositeArrow
from arrows.port_attributes import get_port_attributes, PortAttributes

from copy import copy
from typing import Dict, Callable, TypeVar, Any, Set
from collections import defaultdict

def update_neigh(in_dict: PortAttributes,
                 out_dict: PortAttributes,
                 context: CompositeArrow,
                 working_set: Set[Port]):
    for port, attrs in in_dict.items():
        for neigh_port in context.neigh_ports(port):
            if (neigh_port.arrow != context):
                neigh_attr_keys = out_dict[neigh_port].keys()
                if any((attr_key not in neigh_attr_keys for attr_key in attrs.keys())):
                    working_set.add(neigh_port.arrow)
            out_dict[neigh_port].update(attrs)
        out_dict[port].update(attrs)


def propagate(comp_arrow: CompositeArrow,
              port_attr: PortAttributes=None,
              state=None) -> PortAttributes:
    """
    Propagate values around a composite arrow to determine knowns from unknowns
    The knowns should be determined by the knowns, otherwise an error throws
    Args:
        sub_propagate: an @overloaded function which propagates from each arrow
          sub_propagate(a: ArrowType, port_to_known:Dict[Port, T], state:Any)
        comp_arrow: Composite Arrow to propagate through
        port_attr: port->value map for inputs to composite arrow
        state: A value of any type that is passed around during propagation
               and can be updated by sub_propagate
    Returns:
        port->value map for all ports in composite arrow
    """
    if port_attr is None:
        port_attr = {}
    port_attr = {a: {b: c for b, c in d.items()} for a, d in port_attr.items()}

    _port_attr = defaultdict(lambda: dict())
    if comp_arrow.parent is None:
        comp_arrow.toposort()
    # update port_attr with values stored on port
    for sub_arrow in comp_arrow.get_all_arrows():
        for port in sub_arrow.ports():
            attributes = get_port_attributes(port)
            if port not in port_attr:
                port_attr[port] = {}
            port_attr[port].update(attributes)

    updated = set(comp_arrow.get_sub_arrows())
    update_neigh(port_attr, _port_attr, comp_arrow, updated)
    while len(updated) > 0:
        print(len(updated))
        sub_arrow = updated.pop()
        sub_port_attr = {port: _port_attr[port]
                           for port in sub_arrow.ports()
                           if port in _port_attr}

        if isinstance(sub_arrow, CompositeArrow):
            new_sub_port_attr = propagate(sub_arrow, sub_port_attr, state)
            update_neigh(new_sub_port_attr, _port_attr, comp_arrow, updated)
        else:
            pred_dispatches = sub_arrow.get_dispatches()
            for pred, dispatch in pred_dispatches.items():
                if pred(sub_arrow, sub_port_attr):
                    new_sub_port_attr = dispatch(sub_arrow, sub_port_attr)
                    update_neigh(new_sub_port_attr, _port_attr, comp_arrow, updated)
    return _port_attr
