"""Tests the conversion of composite arrows."""

import tensorflow as tf

from reverseflow.to_graph import arrow_to_graph
from reverseflow.to_arrow import graph_to_arrow
from .test_arrows import test_xyplusx


def test_comparrow() -> None:
    """f(x,y) = x * y + x"""
    d = test_xyplusx()
    tf.reset_default_graph()
    tf_d = arrow_to_graph(d)
    # d_2 = graph_to_arrow(tf_d)

test_comparrow()
