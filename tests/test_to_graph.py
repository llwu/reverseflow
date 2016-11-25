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
    arrow_to_new_graph(arrow)

random_arrow_test(reset_and_conv, "to_graph")
