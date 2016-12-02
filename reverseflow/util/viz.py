"""Arrow visualization tools."""

import tensorflow as tf

from reverseflow.arrows.arrow import Arrow
from reverseflow.to_graph import arrow_to_new_graph


TENSORBOARD_LOGDIR = "tensorboard_logdir"


def show_tensorboard(arrow: Arrow) -> None:
    """Shows arrow on tensorboard."""
    tf.reset_default_graph()
    graph = tf.Graph()
    input_tensors = [tf.placeholder(dtype='float32') for i in range(arrow.num_in_ports())]
    arrow_to_new_graph(arrow, input_tensors, graph)
    writer = tf.train.SummaryWriter(TENSORBOARD_LOGDIR, tf.Session().graph)
    writer.flush()
    print("For graph visualization, invoke")
    print("$ tensorboard --logdir " + TENSORBOARD_LOGDIR)
    print("and click on the GRAPHS tab.")
    input("PRESS ENTER TO CONTINUE.")
