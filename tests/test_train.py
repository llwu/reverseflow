from reverseflow.train.train_y import min_approx_error_arrow
from test_arrows import all_test_arrow_gens
from totality_test import totality_test
import numpy as np
from numpy import ndarray
import tensorflow as tf
from reverseflow.util.viz import *


def gen_tensor(dtype,
               shape) -> ndarray:
    """Generate tensor of type `dtype` and shape `shape`"""
    if dtype.is_floating:
        return np.array(np.random.rand(*shape), dtype=dtype)
    elif dtype.is_integer:
        return np.random.rand(100, size=shape, dtype=dtype)
    else:
        assert False, """Unsuported dtype"""


def test_manual():
    arrow = test_inv_twoxyplusx()
    dataset = [26.0]
    min_approx_error_arrow(arrow, dataset)


def generate_input(arrow: Arrow):
    tf.reset_default_graph()
    return [gen_tensor (port_dtype(port), port_shape(port)
            for port in arrow.get_in_ports()]

# session = tf.Session()
# graph = session.graph
# with graph.as_default():
#     test_min_approx_error()
#     # show_tensorboard_graph()
#     tf.train.SummaryWriter("tensorboard_logdir", graph)

def test_min_approx_error():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(min_approx_error_arrow,
                  all_test_arrows,
                  generate_input,
                  test_name="min_approx_error",
                  ignore=ignore_inv,
                  num_iterations=10)
