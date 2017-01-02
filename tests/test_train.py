from reverseflow.train.train_y import min_approx_error_arrow
from test_arrows import test_inv_twoxyplusx
import numpy as np
import tensorflow as tf
from reverseflow.util.viz import *


def test_train_graph():
    graph, inputs, outputs = test_xyplusx_graph()
    train_y_tf(outputs)

def test_min_approx_error():
    arrow = test_inv_twoxyplusx()
    dataset = [26.0]
    min_approx_error_arrow(arrow, dataset)

session = tf.Session()
graph = session.graph
with graph.as_default():
    test_min_approx_error()
    # show_tensorboard_graph()
    tf.train.SummaryWriter("tensorboard_logdir", graph)
