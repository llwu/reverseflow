from reverseflow.arrows.arrows import (MulArrow, AddArrow, DuplArrow, Bimap,
                                       CompositeArrow, OutPort, InPort)


def test_xyplusx() -> None:
    """f(x,y) = x * y + x"""
    a = MulArrow()
    b = AddArrow()
    c = DuplArrow()
    edges = Bimap()  # type: Bimap[OutPort, InPort]
    # change the rest
    edges.add(c.get_out_ports()[0], a.get_in_ports()[0])  # dupl -> mul
    edges.add(c.get_out_ports()[1], b.get_in_ports()[0])  # dupl -> add
    edges.add(a.get_out_ports()[0], b.get_in_ports()[1])  # mul -> add
    d = CompositeArrow([a, b, c], edges)
