
from reverseflow.arrows.port import OutPort, InPort
from reverseflow.arrows.primitive.math_arrows import MulArrow, AddArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.compose import compose_comb
from reverseflow.to_graph import arrow_to_graph
from reverseflow.to_arrow import graph_to_arrow
import tensorflow as tf

def test_xyplusx() -> None:
    """f(x,y) = x * y + x"""
    tf.reset_default_graph()
    mul = MulArrow()
    add = AddArrow()
    dupl = DuplArrow()
    edges = Bimap()  # type: EdgeMap
    edges.add(dupl.out_ports[0], mul.in_ports[0])  # dupl -> mul
    edges.add(dupl.out_ports[1], add.in_ports[0])  # dupl -> add
    edges.add(mul.out_ports[0], add.in_ports[1])  # mul -> add
    d = CompositeArrow(in_ports=[dupl.in_ports[0], mul.in_ports[1]],
                       out_ports=[add.out_ports[0]], edges=edges)
    # construct same composite arrow with compose_comb
    dupl_to_mul = { 0:0 }
    c1 = compose_comb(dupl, mul, dupl_to_mul)
    c1_to_add = { 0:0 , 1:1 }
    d1 = compose_comb(c1, add, c1_to_add)
    # import pdb; pdb.set_trace()

    tf_d = arrow_to_graph(d)
    # d_2 = graph_to_arrow(tf_d)

test_xyplusx()
