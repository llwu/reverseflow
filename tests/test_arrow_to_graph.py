"""Tests the conversion of composite arrows."""

import tensorflow as tf

from reverseflow.to_graph import arrow_to_graph
from test_arrows import test_xyplusx_flat


def test_arrow_to_graph() -> None:
    """f(x,y) = x * y + x"""
    arrow = test_xyplusx_flat()
    tf.reset_default_graph()
    arrow_to_graph(arrow)

test_arrow_to_graph()
