"""Defines functions for composing arrows."""

from typing import Dict
from copy import copy
from overloading import overload
from reverseflow.arrows.primitivearrow import PrimitiveArrow
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.port import InPort, OutPort
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.util.mapping import Bimap


def edges(a: Arrow) -> EdgeMap:
    if a.is_composite():
        return a.edges
    else:
        return Bimap()


def compose_comb(l: Arrow, r: Arrow, out_to_in: Dict[int, int]) -> CompositeArrow:
    """
    Wires the ports of the primitive arrows on the boundaries.

    The dangling ports become the ports of the composite arrow.

    The ports of the composite arrow may reference l, r,
    or a subarrow of l or r.
    """
    in_ports = copy(l.in_ports)
    out_ports = copy(r.out_ports)

    r_in_connect = [False] * len(r.in_ports)
    new_edges = Bimap()  # type: EdgeMap
    for i in range(len(l.out_ports)):
        if i not in out_to_in:
            out_ports.append(l.out_ports[i])
        else:
            new_edges.add(l.out_ports[i], r.in_ports[out_to_in[i]])
            r_in_connect[out_to_in[i]] = True

    for i in range(len(r.in_ports)):
        if not r_in_connect[i]:
            in_ports.append(r.in_ports[i])

    # TODO: param_ports.append(), etc. should be trivial

    new_edges.update(edges(l))
    new_edges.update(edges(r))
    return CompositeArrow(edges=new_edges, in_ports=in_ports, out_ports=out_ports)


def compose_comb_modular(l: Arrow, r: Arrow, out_to_in: Dict[int, int]) -> CompositeArrow:
    """
    Wires the ports of the composite arrows.

    For each dangling port, an InPort or OutPort is initialized
    referencing l or r.

    The ports of the composite arrow all reference l or r.
    """
    in_ports = [InPort(l, i) for i in range(len(l.in_ports))]
    out_ports = [OutPort(r, i) for i in range(len(r.out_ports))]

    r_in_connect = [False] * len(r.in_ports)
    edges = Bimap()
    for i in range(len(l.out_ports)):
        if i not in out_to_in:
            out_ports.append(OutPort(l, i))
        else:
            edges.add(OutPort(l, i), InPort(r, out_to_in[i]))
            r_in_connect[out_to_in[i]] = True

    for i in range(len(r.in_ports)):
        if not r_in_connect[i]:
            in_ports.append(InPort(r, i))

    return CompositeArrow(edges=edges, in_ports=in_ports, out_ports=out_ports)
