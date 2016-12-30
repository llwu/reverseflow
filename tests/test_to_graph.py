"""Tests the conversion of composite arrows."""

import tensorflow as tf

from reverseflow.config import floatX
from reverseflow.arrows.port import InPort, ParamPort
from reverseflow.arrows.arrow import Arrow
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors
from test_arrows import test_xyplusx_flat, all_composites
from util import random_arrow_test
import numpy as np


def reset_and_conv(arrow: Arrow) -> None:
    tf.reset_default_graph()
    graph = tf.Graph()
    input_tensors = gen_input_tensors(arrow)
    arrow_to_graph(arrow, input_tensors, graph)

# random_arrow_test(reset_and_conv, "to_graph")

def test_all_composites() -> None:
    composites = all_composites()

    for arrow_class in composites:
        arrow = arrow_class()
        print("Testing arrow ", arrow.name)
        reset_and_conv(arrow)

test_all_composites()
