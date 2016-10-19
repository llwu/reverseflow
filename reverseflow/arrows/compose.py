from typing import List
from reverseflow.arrows.primitivearrow import PrimitiveArrow
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.port import InPort, OutPort
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.util.mapping import Bimap


def compose(l: Arrow, r: Arrow) -> CompositeArrow:
    """Connect outputs of arrow l into inputs of arrow r"""
    assert len(l.out_ports) == len(r.in_ports), \
        "Can't compose %s outports into %s in_ports" % \
        (len(l.out_ports), len(r.in_ports))

    n_ports_compose = len(l.out_ports)
    edges = Bimap()  # type: EdgeMap
    for i in range(n_ports_compose):
        edges.add(l.out_ports[i], r.in_ports[i])
    edges.update(l.edges)
    edges.update(r.edges)
    return CompositeArrow(in_ports=l.in_ports, out_ports=r.out_ports,
                          edges=edges)


def compose_comb(l: Arrow, r: Arrow, out_to_in: List[int]) -> CompositeArrow:
    """Wires the ports of the primitive arrows on the boundaries."""
    in_ports = l.in_ports
    out_ports = r.out_ports
    assert len(l.out_ports) == len(out_to_in), \
        "Each out_port of l should have r.in_port index (or -1)"

    r_in_connect = [False]*len(r.in_ports)
    edges = Bimap()
    for i in range(len(l.out_ports)):
        if out_to_in[i] == -1:
            out_ports.append(l.out_ports[i])
        else:
            edges.add(l.out_ports[i], r.in_ports[out_to_in[i]])
            r_in_connect[out_to_in[i]] = True

    for i in range(len(r.in_ports)):
        if not r_in_connect[i]:
            in_ports.append(r.in_ports[i])

    edges.update(l.edges)
    edges.update(r.edges)
    return CompositeArrow(in_ports=in_ports, out_ports=out_ports,
                          edges=edges)

def compose_comb_modular(l: Arrow, r: Arrow, out_to_in: List[int]) -> CompositeArrow:
    """Wires the ports of the composite arrows."""
    in_ports = [InPort(l, i) for i in range(len(l.in_ports))]
    out_ports = [OutPort(r, i) for i in range(len(r.out_ports))]
    assert len(l.out_ports) == len(out_to_in), \
        "Each out_port of l should have r.in_port index (or -1)"

    r_in_connect = [False] * len(r.in_ports)
    edges = Bimap()
    for i in range(len(l.out_ports)):
        if out_to_in[i] == -1:
            out_ports.append(OutPort(l, i))
        else:
            edges.add(OutPort(l, i), InPort(r, out_to_in[i]))
            r_in_connect[out_to_in[i]] = True

    for i in range(len(r.in_ports)):
        if not r_in_connect[i]:
            in_ports.append(InPort(r, i))

    return CompositeArrow(in_ports=in_ports, out_ports=out_ports,
                          edges=edges)
