import tensorflow as tf
from reverseflow.to_arrow import graph_to_arrow
from test_graphs import test_xyplusx_graph


def test_xyplusx() -> None:
    """f(x,y) = x * y + x"""
    graph, inputs, outputs = test_xyplusx_graph()
    arrow = graph_to_arrow(outputs)
