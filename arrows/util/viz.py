"""Arrow visualization tools."""

import tensorflow as tf

from arrows.arrow import Arrow
from arrows.primitive.array_arrows import const_to_tuple
from reverseflow.to_graph import arrow_to_graph


TENSORBOARD_LOGDIR = "tensorboard_logdir"

def show_tensorboard_graph()  -> None:
    writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR, tf.Session().graph)
    writer.flush()
    print("For graph visualization, invoke")
    print("$ tensorboard --logdir " + TENSORBOARD_LOGDIR)
    print("and click on the GRAPHS tab.")
    input("PRESS ENTER TO CONTINUE.")


def make_placeholder(port, port_attrs):
    if port in port_attrs and 'shape' in port_attrs[port]:
        return tf.placeholder(dtype='float32', shape=const_to_tuple(port_attrs[port]['shape']))
    else:
        return tf.placeholder(dtype='float32')


def show_tensorboard(arrow: Arrow, port_attrs=None) -> None:
    """Shows arrow on tensorboard."""
    if port_attrs is None:
        port_attrs = {}
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        input_tensors = [make_placeholder(port, port_attrs) for port in arrow.in_ports()]
        arrow_to_graph(arrow, input_tensors)
        show_tensorboard_graph()
