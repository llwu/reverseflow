from arrows import (Arrow, CompositeArrow, compose_comb_modular, compose_comb)
from arrows.port_attributes import is_param_port, is_error_port
from arrows.std_arrows import *
from arrows.config import floatX
from arrows.util.viz import show_tensorboard_graph
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors
from typing import List, Generator, Callable
import tensorflow as tf
from tensorflow import Graph, Tensor, Session


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


def gen_fetch(sess: Session,
              loss,
              debug=False):
    update_step = gen_update_step(loss)
    init = tf.initialize_all_variables()
    sess.run(init)

    fetch = {}
    if debug:
        fetch['check'] = tf.add_check_numerics_ops()

    fetch['loss'] = loss
    fetch['update_step'] = update_step
    return fetch


def gen_update_step(loss: Tensor) -> Tensor:
    with tf.name_scope('optimization'):
        # optimizer = tf.train.MomentumOptimizer(0.001,
        #                                        momentum=0.1)
        optimizer = tf.train.AdamOptimizer(0.001)
        update_step = optimizer.minimize(loss)
        return update_step


def gen_batch(input_tensors, input_data):
    return dict(zip(input_tensors, input_data))


def train_loop(sess: Session,
               fetch,
               generators: Sequence[Generator],
               num_iterations=100000,
               output_call_back=None,
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
        # Generate input
        feed_dict = {}
        for gen in generators:
            sub_feed_dict = next(gen)
            feed_dict.update(sub_feed_dict)
        fetch_res = sess.run(fetch, feed_dict=feed_dict)
        if output_call_back:
            output_call_back(fetch_res)
        print("Iteration: ", i, " Loss: ", fetch_res['loss'])
        if "to_print" in fetch_res:
            print(fetch_res["to_print"])
