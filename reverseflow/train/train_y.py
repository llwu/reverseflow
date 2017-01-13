from arrows import (Arrow, CompositeArrow, compose_comb_modular, compose_comb)
from arrows import InPort
from arrows.std_arrows import *

from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors
from arrows.config import floatX

from typing import List
import tensorflow as tf
from tensorflow import Graph, Tensor, Session

def gen_update_step(loss: Tensor) -> Tensor:
    with tf.name_scope('optimization'):
        optimizer = tf.train.MomentumOptimizer(0.001,
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
               num_iterations=1000,
               summary_gap=500,
               save_every=10,
               sfx='',
               compress=False,
               save_dir="./",
               saver=None
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

    for i in range(num_iterations):
        feed_dict = gen_batch(input_tensors, input_data)
        loss_res = sess.run([loss, update_step] + output_tensors, feed_dict=feed_dict)
        print("Loss is ", loss_res)

def train_y_tf(params: List[Tensor],
               losses: List[Tensor],
               input_tensors,
               output_tensors,
               input_data,
               kwargs) -> Graph:
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

def min_approx_error_arrow(arrow: CompositeArrow, input_data: List) -> CompositeArrow:
    """
    Find parameter values of arrow which minimize approximation error of arrow(data)
    Args:
        arrow: Parametric Arrow
        input_data: List of input data for each input of arrow

    Returns:
        parametric_arrow with parameters fixed
    """
    # assert arrow.num_in_ports() == len(input_data), "Arrow has %s in_ports but only %s input data feeds" % (arrow.num_in_ports(), len(input_data))
    input_tensors = gen_input_tensors(arrow)
    output_tensors = arrow_to_graph(arrow, input_tensors)
    params = [t for i, t in enumerate(input_tensors) if isinstance(arrow.get_in_ports()[i], ParamPort)]
    errors = [t for i, t in enumerate(output_tensors) if isinstance(arrow.get_out_ports()[i], ErrorPort)]
    assert len(params) > 0, "Must have parametric inports"
    assert len(errors) > 0, "Must have error outports"
    train_y_tf(params, errors, input_tensors, output_tensors, input_data)
