
from reverseflow.arrows.port import OutPort, InPort
from reverseflow.arrows.primitive.math_arrows import MulArrow, AddArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.decode import arrow_to_graph

def test_xyplusx() -> None:
    """f(x,y) = x * y + x"""
    a = MulArrow()
    b = AddArrow()
    c = DuplArrow()
    edges = Bimap()  # type: Bimap[OutPort, InPort]
    d = CompositeArrow()
    edges.add(c.out_ports[0], a.in_ports[0])  # dupl -> mul
    edges.add(c.out_ports[1], b.in_ports[0])  # dupl -> add
    edges.add(a.out_ports[0], b.in_ports[1])  # mul -> add
    d.add_edges(edges)
    tf_d = arrow_to_graph(d)
    # d2 = graph_to_arrow(tf_d)

test_xyplusx()
