import tensorflow as tf
from tensorflow import Graph, Tensor
from typing import List, Tuple


def test_xyplusx_graph() -> Tuple[Graph, List[Tensor], List[Tensor]]:
    """f(x,y) = x * y + x"""
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder('float32')
        y = tf.placeholder('float32')
        z = x * y + x
    return graph, [x, y], [z]
