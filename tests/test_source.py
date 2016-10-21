
from reverseflow.arrows.port import OutPort, InPort
from reverseflow.arrows.sourcearrow import SourceArrow
from reverseflow.arrows.primitive.math_arrows import MulArrow, AddArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.compose import compose_comb
from reverseflow.to_graph import arrow_to_graph
from reverseflow.to_arrow import graph_to_arrow
import tensorflow as tf
import numpy as np

def test_source() -> None:
    """f(x,y) = x * 2"""
    mul = MulArrow()
    two = SourceArrow(np.ones(10))
    edges = Bimap()  # type: EdgeMap
    edges.add(two.out_ports[0], mul.in_ports[0])  # dupl -> mul
    d = CompositeArrow(in_ports=[mul.in_ports[1]],
                       out_ports=[mul.out_ports[0]], edges=edges)
