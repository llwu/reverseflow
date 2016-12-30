ffrom reverseflow.arrows.compositearrow import CompositeArrow

from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors
from reverseflow.arrows.port import ParamPort, InPort, ErrorPort
from reverseflow.arrows.arrow import Arrow
from reverseflow.config import floatX

from typing import List
import tensorflow as tf
from tensorflow import Graph, Tensor, Session

def gen_update_step(loss: Tensor) -> Tensor:
    optimizer = tf.train.MomentumOptimizer(learning_rate=options['learning_rate'],
                                           momentum=options['momentum'])
    update_step = optimizer.minimize(loss)
    return updaet_step


def accumulate_losses(tensors: List[Tensor]) -> Tensor:
    """
    Mean of list of tensors of arbitrary size
    Args:
        tensors: list of tensors

    Returns:
        mean tensor
    """
    return tf.add_n([tf.reduce_mean(t) for t in tensors]) / len(tensors)

def train_y_tf(params: List[Tensor], losses: List[Tensor]) -> Graph:
    """
    """
    loss = accumulate_losses(losses)
    update_step = gen_update_step(loss)
    train_loop(update_step)

def min_approx_error_arrow(arrow: CompositeArrow, y_data: List) -> CompositeArrow:
    """
    Find parameter values of arrow which minimize approximation error of arrow(data)
    Args:
        arrow: Parametric Arrow
        y_data: List of input data to arrow

    Returns:
        parametric_arrow with parameters fixed
    """
    graph = tf.Graph()
    input_tensors = gen_input_tensors(arrow)
    output_tensors = arrow_to_graph(arrow, input_tensors, graph,)
    params = [t for i, t in enumerate(input_tensors) if isinstance(arrow.in_ports[i], ParamPort)]
    errors = [t for i, t in enumerate(output_tensors) if isinstance(arrow.out_ports[i], ErrorPort)]
    assert len(params) > 0, "Must have parametric inports"
    assert len(error) > 0, "Must have error outports"
    train_y_tf(graph, params, errors)

def train_loop(update_step,
               sess: Session,
               num_iterations = 1000,
               summary_gap=500,
               save_every=10,
               sfx='',
               compress=False,
               save_dir="./",
               saver=None):

    for i in range(num_iterations):
        sess.run()
