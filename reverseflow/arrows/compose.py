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
def compose_comb(l: Arrow, r: Arrow, out_to_in: List[int]) -> CompositeArrow:
    in_ports = l.in_ports
    out_ports = r.out_ports

    r_in_connect = [False]*len(r.in_ports)
    assert len(l.out_ports) == len(out_to_in), \
        "Each out_port of l should have a r.in_port index to connect (or -1)"

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
    return CompositeArrow(in_ports = in_ports, out_ports = out_ports,
                            edges = edges)
