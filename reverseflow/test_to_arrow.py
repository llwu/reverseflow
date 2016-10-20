import tensorflow as tf
from reverseflow.to_arrow import graph_to_arrow


def test_graph_to_arrow():
    x = tf.placeholder('float32', name='x')
    y = tf.placeholder('float32', name='y')
    z = x * y + x
    arrow = graph_to_arrow([z])
    return arrow
