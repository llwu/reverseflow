""""Generic Propagation of values around a composite arrow"""
import arrows
from arrows.arrow import Arrow
from arrows.port import Port
from arrows.compositearrow import CompositeArrow
from arrows.port_attributes import get_port_attr, PortAttributes

from arrows.port_attributes import *
from arrows.compositearrow import *
from arrows.util.misc import *


from copy import copy
from typing import Dict, Callable, TypeVar, Any, Set
from collections import defaultdict

import numpy as np

# FIXME: Really port_attr reflects two different kinds of things.
# This which are actually about the ports themselves, i.e. whether its an out
# port or inport, which we dont want to propagate, and things which are Really
# abstractios (or actually) values which should propagate along the node
# e.g. shape, type, symbolic, etc.  NO_PROP is a simple workaround, need better
# solution

# Do not propagate port attributes of this kind
DONT_PROP = set(['InOut', 'parametric', 'error'])

def resolve(x, y, fail_on_conflict=True):
    # _x = x
    # _y = y
    if isinstance(x, np.ndarray) or isinstance(x, int) or isinstance(x, float) or isinstance(x, np.number):
        # _x = np.where(np.isfinite(x), x, y)
        # _y = np.where(np.isfinite(y), y, x)
        # diff = np.abs(_x - _y)
        # diff = np.where(np.isfinite(diff), diff, 0)
        diff = np.abs(x - y)
        err = np.mean(diff)
        # if fail_on_conflict and err >= 1e-6:
        #     import pdb; pdb.set_trace()
        assert not fail_on_conflict or err < 1e-6, "conflicting: %s, %s" % (x, y)
    else:
        assert not fail_on_conflict or x == y, "conflicting: %s, %s" % (x, y)
    # if isinstance(x, int):
    #     _x = int(_x)
    # if isinstance(x, float):
    #     _x = float(_x)
    # return _x
    return x

def update_port_attr(to_update: PortAttributes,
                     with_p: PortAttributes,
                     dont_update: Set,
                     fail_on_conflict=True):
    for key, value in with_p.items():
        if key not in dont_update:
            if key in to_update:
                value = resolve(value, to_update[key], fail_on_conflict)
            to_update[key] = value

def equiv_neigh(port: Port, context):
    seen = set()
    to_see = set([port])
    equiv = set()
    # import pdb; pdb.set_trace()
    while len(to_see) > 0:
        port = to_see.pop()
        seen.add(port)
        for neigh in context.neigh_ports(port):
            equiv.add(neigh)
            if neigh not in seen:
                to_see.add(neigh)
    return equiv

def update_neigh(sub_port_attr: PortAttributes,
                 port_attr: PortAttributes,
                 context: CompositeArrow,
                 working_set: Set[Arrow]):
    """
    For every port in sub_port_attr the port_attr data all of its connected nodes
    Args:
        sub_port_attr: Port Attributes restricted to a particular arrow
        port_attr: Global PortAttributes for composition to be update
        context: The composition
        working_set: Set of arrows that need further propagation
    """
    for port, attrs in sub_port_attr.items():
        neigh_ports = equiv_neigh(port, context)
        for neigh_port in neigh_ports:
            # If the neighbouring node doesn't have a key which I have, then it will
            # have to be added to working set to propagate again
            if (neigh_port.arrow != context):
                neigh_attr_keys = port_attr[neigh_port].keys()
                if any((attr_key not in neigh_attr_keys for attr_key in attrs.keys())):
                    working_set.add(neigh_port.arrow)
            update_port_attr(port_attr[neigh_port], attrs, dont_update=DONT_PROP)
        # Update global with this port
        update_port_attr(port_attr[port], attrs, dont_update=DONT_PROP)


def extract_port_attr(comp_arrow, port_attr):
    for sub_arrow in comp_arrow.get_all_arrows():
        for port in sub_arrow.ports():
            attributes = get_port_attr(port)
            if port not in port_attr:
                port_attr[port] = {}
            update_port_attr(port_attr[port], attributes, set())

#FIXME: Does unnecessary Propagate, will do a dispatch more than once
# which is (probably) never needed
# FIXME: There is a loss of information bug from port_attr,
# remove __eq__ form symbolictensor and run voxel render to sees
def propagate(comp_arrow: CompositeArrow,
              port_attr: PortAttributes=None,
              state=None,
              already_prop=None) -> PortAttributes:
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
    already_prop = set() if already_prop is None else already_prop
    print("Propagating")
    # Copy port_attr to avoid affecting input
    port_attr = {} if port_attr is None else port_attr
    _port_attr = defaultdict(lambda: dict())
    for port, attr in port_attr.items():
        for attr_key, attr_value in attr.items():
            _port_attr[port][attr_key] = attr_value

    # if comp_arrow.parent is None:
    #     comp_arrow.toposort()

    # update port_attr with values stored on port
    extract_port_attr(comp_arrow, _port_attr)

    updated = set(comp_arrow.get_sub_arrows())
    update_neigh(_port_attr, _port_attr, comp_arrow, updated)
    while len(updated) > 0:
        # print(len(updated), " arrows updating in proapgation iteration")
        sub_arrow = updated.pop()
        sub_port_attr = {port: _port_attr[port]
                           for port in sub_arrow.ports()
                           if port in _port_attr}

        pred_dispatches = sub_arrow.get_dispatches()
        for pred, dispatch in pred_dispatches.items():
            if pred(sub_arrow, sub_port_attr) and (sub_arrow, dispatch) not in already_prop:
                new_sub_port_attr = dispatch(sub_arrow, sub_port_attr)
                update_neigh(new_sub_port_attr, _port_attr, comp_arrow, updated)
                already_prop.add((sub_arrow, dispatch))
        if isinstance(sub_arrow, CompositeArrow):
            sub_port_attr = {port: _port_attr[port]
                            for port in sub_arrow.ports()
                            if port in _port_attr}
            new_sub_port_attr = propagate(sub_arrow, sub_port_attr, state, already_prop)
            update_neigh(new_sub_port_attr, _port_attr, comp_arrow, updated)
    print("Done Propagating")
    return _port_attr
