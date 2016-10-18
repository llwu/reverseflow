from overloading import overload

@overload
def compose(l: PrimitiveArrow, r: PrimitiveArrow) -> CompositeArrow:
    assert len(l.out_ports) == len(r._in_ports), \
        "Can't compose %s outports into %s in_ports" % \
        (len(l.out_ports), len(r._in_ports))

    n_ports_compose = len(l.out_ports)
    edges = Bimap
    for i in range(n_ports_compose):
        edges.add(l.out_ports[i], r.in_ports[i])
    edges.update(l.edges)
    edges.update(r.edges)
    return CompositeArrow(in_ports=l.in_ports, out_ports=a.out_ports,
                          edges=edges)

@overload
def compose(a: PrimitiveArrow, b: CompositeArrow) -> CompositeArrow:
    pass

@overload
def compose(a: PrimitiveArrow, b: CompositeArrow) -> CompositeArrow:
    pass

@overload
def compose(a: CompositeArrow, b: PrimitiveArrow) -> CompositeArrow:
    pass

@overload
def compose(a: CompositeArrow, b: CompositeArrow) -> CompositeArrow:
    pass
