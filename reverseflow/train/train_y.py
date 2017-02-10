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


def gen_update_step(loss: Tensor) -> Tensor:
    with tf.name_scope('optimization'):
        optimizer = tf.train.MomentumOptimizer(0.1,
                                               momentum=0.1)
        update_step = optimizer.minimize(loss)
        return update_step


def accumulate_losses(tensors: List[Tensor]) -> Tensor:
    """
    Mean of list of tensors of arbitrary size
    Args:
        tensors: list of tensors

    Returns:
        mean tensor
    """
    with tf.name_scope('loss'):
        return tf.add_n([tf.reduce_mean(t) for t in tensors]) / len(tensors)


def gen_batch(input_tensors, input_data):
    return dict(zip(input_tensors, input_data))


def train_loop(update_step,
               sess: Session,
               loss,
               input_tensors,
               output_tensors,
               input_data,
               num_iterations=5,
               summary_gap=500,
               save_every=10,
               sfx='',
               compress=False,
               save_dir="./",
               saver=None,
               stop_test=None,
               output_call_back=None,
               debug=False,
               **kwargs):
    """Perform training
    Args:
        update_step:
        sess: Tensorflow session
        loss: tensor to minimize
        input_tensors:
        output_tensors:
        input_data:
        num_iterations: number of iterations to run
        summary_gap:
        save_every
        sfx: String suffix to append to log data
        compress: Using numpy compression for paramter saving
        save_dir: Directory for saving logs
        saver: Tensorflow saver for saving
    """
    fetch = {}
    if debug:
        fetch['check'] = tf.add_check_numerics_ops()

    fetch['loss'] = loss
    fetch['update_step'] = update_step
    fetch['output_tensors'] = output_tensors

    # fetch = fetch + [loss, update_step] + output_tensors
    for i in range(num_iterations):
        feed_dict = gen_batch(input_tensors, input_data)
        fetch_res = sess.run(fetch, feed_dict=feed_dict)
        if output_call_back:
            output_call_back(fetch_res)
        print("Iteration: ", i, " Loss: ", fetch_res['loss'])

def train_y_tf(params: List[Tensor],
               losses: List[Tensor],
               input_tensors: List[Tensor],
               output_tensors: List[Tensor],
               input_data,
               **kwargs) -> Graph:
    """
    """
    loss = accumulate_losses(losses)
    update_step = gen_update_step(loss)
    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)
    train_loop(update_step,
               sess,
               loss,
               input_tensors,
               output_tensors,
               input_data,
               **kwargs)


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
    with tf.name_scope("%s_inv" % arrow.name):
        input_tensors = gen_input_tensors(arrow)
        output_tensors = arrow_to_graph(arrow, input_tensors)
    # show_tensorboard_graph()

    param_tensors = [t for i, t in enumerate(input_tensors) if is_param_port(arrow.in_ports()[i])]
    error_tensors = [t for i, t in enumerate(output_tensors) if error_filter(arrow.out_ports()[i])]
    assert len(param_tensors) > 0, "Must have parametric inports"
    assert len(error_tensors) > 0, "Must have error outports"
    train_y_tf(param_tensors, error_tensors, input_tensors, output_tensors,
               input_data, **kwargs)
