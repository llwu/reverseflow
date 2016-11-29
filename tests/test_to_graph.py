"""Tests the conversion of composite arrows."""

import tensorflow as tf

from reverseflow.arrows.arrow import Arrow
from reverseflow.to_graph import arrow_to_new_graph
from test_arrows import test_xyplusx_flat
from util import random_arrow_test


def test_arrow_to_graph() -> None:
    """f(x,y) = x * y + x"""
    arrow = test_xyplusx_flat()
    tf.reset_default_graph()
    arrow_to_new_graph(arrow)


def reset_and_conv(arrow: Arrow) -> None:
    tf.reset_default_graph()
    graph = tf.Graph()
    input_tensors = [tf.placeholder(dtype='float32') for i in range(arrow.n_in_ports)]
    param_tensors = [tf.Variable(dtype='float32', shape=()) for i in range(arrow.n_param_ports)]
    arrow_to_new_graph(arrow, input_tensors, param_tensors, graph)

random_arrow_test(reset_and_conv, "to_graph")
