"""Tests the conversion of composite arrows."""
import tensorflow as tf
from arrows.arrow import Arrow
from test_arrows import all_test_arrow_gens
from totality_test import totality_test
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors


def reset_and_conv(arrow: Arrow) -> None:
    tf.reset_default_graph()
    input_tensors = gen_input_tensors(arrow)
    arrow_to_graph(arrow, input_tensors)


def test_to_graph():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(reset_and_conv, all_test_arrows, test_name="to_graph")
