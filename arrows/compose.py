"""Defines functions for composing arrows."""

from typing import Dict
from copy import copy
from overloading import overload
from arrows.primitivearrow import PrimitiveArrow
from arrows.arrow import Arrow
from arrows.port import InPort, OutPort
from arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.util.mapping import Bimap

@overload
def compose(l: PrimitiveArrow,
            r:PrimitiveArrow):
    c = CompositeArrow()

@overload
def compose(l: CompositeArrow, r: PrimitiveArrow):



def edges(a: Arrow) -> EdgeMap:
    if a.is_composite():
        return a.edges
    else:
        return Bimap()


def compose_comb(l: Arrow, r: Arrow, out_to_in: Dict[int, int], name: str=None) -> CompositeArrow:
    """
    Wires the ports of the primitive arrows on the boundaries.

    Edges are made between the inner ports of l and r.
    """
    in_ports = copy(l.inner_in_ports())
    out_ports = copy(r.inner_out_ports())

    r_in_connect = [False] * len(r.get_in_ports())
    new_edges = Bimap()  # type: EdgeMap
    for i in range(len(l.get_out_ports())):
        if i not in out_to_in:
            out_ports.append(l.inner_out_ports()[i])
        else:
            new_edges.add(l.inner_out_ports()[i], r.inner_in_ports()[out_to_in[i]])
            r_in_connect[out_to_in[i]] = True

    for i in range(len(r.get_in_ports())):
        if not r_in_connect[i]:
            in_ports.append(r.inner_in_ports()[i])

    # TODO: propagate paramport-ness

    new_edges.update(edges(l))
    new_edges.update(edges(r))
    return CompositeArrow(edges=new_edges,
                          in_ports=in_ports,
                          out_ports=out_ports,
                          name=name)


def compose_comb_modular(l: Arrow,
                         r: Arrow,
                         out_to_in: Dict[int, int],
                         name=None) -> CompositeArrow:
    """
    Wires the ports of the composite arrows.

    Edges are made between the outer ports of l and r.
    """
    in_ports = copy(l.get_in_ports())
    out_ports = copy(r.get_out_ports())

    r_in_connect = [False] * len(r.get_in_ports())
    edges = Bimap()
    for i in range(len(l.get_out_ports())):
        if i not in out_to_in:
            out_ports.append(l.get_out_ports()[i])
        else:
            edges.add(l.get_out_ports()[i], r.get_in_ports()[out_to_in[i]])
            r_in_connect[out_to_in[i]] = True

    for i in range(len(r.get_in_ports())):
        if not r_in_connect[i]:
            in_ports.append(r.get_in_ports()[i])

    # TODO: propagate paramport-ness

    return CompositeArrow(edges=edges,
                          in_ports=in_ports,
                          out_ports=out_ports,
                          name=name)
