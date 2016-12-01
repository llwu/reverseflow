from reverseflow.train.train_y import train_y_arr, train_y_tf

from test_arrows import test_xyplusx_flat
from test_graphs import test_xy
import numpy as np


def test_train_graph():
    graph, inputs, outputs = test_xyplusx_graph()
    train_y_tf(outputs)

def test_train_arrow():
    arrow = test_xyplusx_flat()
    dataset = np.random.rand(100)
    train_y_arr(arrow, dataset)
