from arrows import (Arrow, CompositeArrow, compose_comb_modular, compose_comb)
from arrows.port_attributes import is_param_port, is_error_port
from arrows.std_arrows import *
from arrows.config import floatX
from arrows.util.viz import show_tensorboard_graph
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors
from typing import List
import tensorflow as tf
from tensorflow import Graph, Tensor, Session
from reverseflow.train.common import *


def min_approx_error_arrow(arrow: Arrow,
                           input_data: List,
                           error_filter=is_error_port,
                           **kwargs) -> CompositeArrow:
    """
    Find parameter values of arrow which minimize approximation error of arrow(data)
    Args:
        arrow: Parametric Arrow
        input_data: List of input data for each input of arrow

    Returns:
        parametric_arrow with parameters fixed
    """
    with tf.name_scope(arrow.name):
        input_tensors = gen_input_tensors(arrow)
        output_tensors = arrow_to_graph(arrow, input_tensors)
    # show_tensorboard_graph()

    param_tensors = [t for i, t in enumerate(input_tensors) if is_param_port(arrow.in_ports()[i])]
    error_tensors = [t for i, t in enumerate(output_tensors) if error_filter(arrow.out_ports()[i])]
    assert len(param_tensors) > 0, "Must have parametric inports"
    assert len(error_tensors) > 0, "Must have error outports"
    train_tf(param_tensors, error_tensors, input_tensors, output_tensors,
               input_data, **kwargs)
