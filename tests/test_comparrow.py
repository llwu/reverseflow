from reverseflow.arrows.primitive.math_arrows import MulArrow, AddArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.bimap import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow


def test_xyplusx() -> None:
    """f(x,y) = x * y + x"""
    a = MulArrow()
    b = AddArrow()
    c = DuplArrow()
    edges = Bimap()  # type: Bimap[OutPort, InPort]
    edges.add(c.get_out_ports()[0], a.get_in_ports()[0])  # dupl -> mul
    edges.add(c.get_out_ports()[1], b.get_in_ports()[0])  # dupl -> add
    edges.add(a.get_out_ports()[0], b.get_in_ports()[1])  # mul -> add
    d = CompositeArrow([a, b, c], edges)
    tf_d = arrow_to_graph(d)
    d2 = graph_to_arrow(tf_d)
    # inv_d = invert(d)
