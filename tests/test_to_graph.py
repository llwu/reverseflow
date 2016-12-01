"""Tests the conversion of composite arrows."""

import tensorflow as tf

from reverseflow.arrows.port import InPort, ParamPort
from reverseflow.arrows.arrow import Arrow
from reverseflow.to_graph import arrow_to_new_graph
from test_arrows import test_xyplusx_flat, all_composites
from util import random_arrow_test
import numpy as np


def test_arrow_to_graph() -> None:
    """f(x,y) = x * y + x"""
    arrow = test_xyplusx_flat()
    tf.reset_default_graph()
    arrow_to_new_graph(arrow)

def generate_input(arrow):
    input_tensors = []
    for in_port in arrow.in_ports:
        if isinstance(in_port, ParamPort):
            # FIXME for right shape
            input_tensors.append(tf.Variable(np.random.rand(1)))
        elif isinstance(in_port, InPort):
            input_tensors.append(tf.placeholder(dtype='float32'))
        else:
            assert False, "Don't know how to handle %s" % in_port
    return input_tensors

def reset_and_conv(arrow: Arrow) -> None:
    tf.reset_default_graph()
    graph = tf.Graph()
    input_tensors = generate_input(arrow)
    arrow_to_new_graph(arrow, input_tensors, graph)

random_arrow_test(reset_and_conv, "to_graph")

# def test_all_composites() -> None:
#     composites = all_composites()
#
#     for arrow_class in composites:
#         reset_and_conv(arrow_class())
#
# test_all_composites()
