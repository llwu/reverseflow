import tensorflow as tf
from reverseflow.to_arrow import graph_to_arrow


def test_xyplusx() -> None:
    """f(x,y) = x * y + x"""
    x = tf.placeholder('float32')
    y = tf.placeholder('float32')
    z = x * y + x
    graph_to_arrow([z])
