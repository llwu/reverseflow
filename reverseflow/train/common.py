from arrows import (Arrow, CompositeArrow, compose_comb_modular, compose_comb)
from arrows.port_attributes import is_param_port, is_error_port
from arrows.std_arrows import *
from arrows.config import floatX
from arrows.util.io import mk_dir
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors
from typing import List, Generator, Callable
import tensorflow as tf
from tensorflow import Graph, Tensor, Session
import os

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


def layer_width(i, o, n, p):
    """Compute the layer width for a desired number of parameters
    Args:
        i: Length of input
        o: Length of output
        p: Desired number of parameters
        n: Number of layers
    Returns:
        Size of inner layers"""
    b = i + 1 + o + n
    a = n
    c = o - p
    inner = np.sqrt(b*b - 4*a*c)
    return (-b + inner)/(2*a), (-b - inner)/(2*a)


def get_tf_num_params(arrow):
    graph = tf.Graph()
    with graph.as_default():
        input_tensors = gen_input_tensors(arrow, param_port_as_var=False)
        output_tensors = arrow_to_graph(arrow, input_tensors)
        vs = tf.global_variables()
        return sum([v.get_shape().num_elements() for v in vs])

def extract_tensors(arrow: Arrow,
                    extra_ports=[],
                    append_default=True,
                    grabs=None,
                    optional=None):
    """
    Converts an arrow into a graph and extracts tensors which correspond to
    ports with particular properties
    Args:
        arrow: arrow to convert
        grabs: dict {name: filter function}
        append_default: append to default grabs, if false then replaces it
    Returns:

    """
    optional = set() if optional is None else optional
    def_grabs   = {'input': lambda p: is_in_port(p) and not is_param_port(p),
                   'param': lambda p: is_param_port(p),
                   'error': lambda p: is_error_port(p),
                   'output': lambda p: is_out_port(p)}

    _grabs = {}
    if append_default:
        _grabs.update(def_grabs)

    _grabs.update(grabs)

    # Convert to tensorflow graph and get input, output, error, and parma_tensors
    with tf.name_scope(arrow.name):
        input_tensors = gen_input_tensors(arrow, param_port_as_var=False)
        port_grab = {p: None for p in extra_ports}
        output_tensors = arrow_to_graph(arrow, input_tensors, port_grab=port_grab)

    extra_tensors = list(port_grab.values())
    output = {}
    for grab_key, grab_func in _grabs.items():
        all_dem  = []
        for i, t in enumerate(input_tensors):
            if grab_func(arrow.in_port(i)):
                all_dem.append(t)
        for i, t in enumerate(output_tensors):
            if grab_func(arrow.out_port(i)):
                all_dem.append(t)
        # for port, t in port_grab:
        #     if grab_func(port):
        #         all_dem.append(t)
        assert grab_key in optional or len(all_dem) > 0, "%s empty" % grab_key
        if len(all_dem) > 0:
            output[grab_key] = all_dem

    output['extra'] = list(port_grab.values())
    return output
    #
    # inp_tensors = [t for i, t in enumerate(input_tensors) if not is_param_port(arrow.in_ports()[i])]
    # param_tensors = [t for i, t in enumerate(input_tensors) if is_param_port(arrow.in_ports()[i])]
    # error_tensors = [t for i, t in enumerate(output_tensors) if error_filter(arrow.out_ports()[i])]
    # assert len(param_tensors) > 0, "Must have parametric inports"
    # assert len(error_tensors) > 0, "Must have error outports"
    # return {'input': inp_tensors,
    #         'output': output_tensors,
    #         'param': param_tensors,
    #         'error': error_tensors,
    #         'extra': extra_tensors}


def prep_save(sess: Session, save: bool, sfx: str, params_file: str, load: bool):
    save_params = {}
    if save or load:
        saver = tf.train.Saver()
    if save is True:
        save_dir = mk_dir(sfx)
        save_params['save_dir'] = save_dir
        options_path = os.path.join(save_dir, "options")
        # save_dict_csv(options_path, options)
        save_params['saver'] = saver = tf.train.Saver()
    if load is True:
        saver.restore(sess, params_file)
    return save_params


# def load_train_save(sess, options, sfx, save_dir):
#     options_path = os.path.join(save_dir, "options")
#     save_dict_csv(options_path, options)
#     saver = tf.train.Saver()
#
#     if options['load'] is True:
#         saver.restore(sess, options['params_file'])
#         # adt.load(options['params_file'])
#
#     # if options['save_params'] is True:
#     #     path = os.path.join(save_dir, "final" + sfx)
#     #     # adt.save_params(path)
#
#     if options['train'] is True:
#         train(adt, pbt, sess, num_epochs=options['num_epochs'],
#               sfx=sfx, save_dir=save_dir, save_every=options['save_every'],
#               compress=options['compress'], saver=saver)
#
#     return sess


def gen_fetch(sess: Session,
              debug=False,
              **kwargs):
    init = tf.initialize_all_variables()
    sess.run(init)

    fetch = {}
    if debug:
        fetch['check'] = tf.add_check_numerics_ops()

    return fetch


def gen_update_step(loss: Tensor) -> Tensor:
    with tf.name_scope('optimization'):
        # optimizer = tf.train.MomentumOptimizer(0.001,
        #                                        momentum=0.1)
        optimizer = tf.train.AdamOptimizer(0.001)
        update_step = optimizer.minimize(loss)
        return update_step


def train_loop(sess: Session,
               loss_updates: Sequence[Tensor],
               fetch,
               generators: Sequence[Generator],
               test_generators,
               loss_ratios: Sequence[int]=None,
               test_every=100,
               num_iterations=100000,
               callbacks=[],
               **kwargs):
    """Perform training
    Args:
        sess: Tensorflow session
        loss_updates: gradient update tensors:
        test_generators: a sequence of generators. A generator should return
            a dict {tensor: value}.  The union of all the dicts is passed as
            feed_dict in the gradient steps.
        loss_ratios:
        num_iterations: number of iterations to run
        test_every: evaluate test data set test_every iterations
        num_iterations: number of iterations
        callbacks: functions to be called with result from fetch
    """
    # Default 1 for loss_ratios and normalize
    loss_ratios = [1 for i in range(len(loss_updates))] if loss_ratios is None else loss_ratios
    loss_ratios = loss_ratios / np.sum(loss_ratios)

    # Prepare dict to be passed to callbacks
    callback_dict = {}
    callback_dict.update(kwargs)
    callback_dict.update({'sess': sess})

    # Main loop
    for i in range(num_iterations):
        # Generate input
        curr_fetch = {}
        curr_fetch.update(fetch)
        curr_fetch["update_loss"] = np.random.choice(loss_updates, p=loss_ratios)
        feed_dict = {}
        for gen in generators:
            sub_feed_dict = next(gen)
            feed_dict.update(sub_feed_dict)
        # Optimizeation Step
        fetch_res = sess.run(curr_fetch, feed_dict=feed_dict)
        for cb in callbacks:
            cb(fetch_res, feed_dict, i, **callback_dict)
        print("Iteration: ", i, " Loss: ", fetch_res['loss'])
        if "to_print" in fetch_res:
            print(fetch_res["to_print"])

        # Evaluate on test data every test_every iterations
        if i % test_every == 0:
            test_feed_dict = {}
            for gen in test_generators:
                sub_feed_dict = next(gen)
                test_feed_dict.update(sub_feed_dict)
            test_feed_dict = {k: v for k, v in test_feed_dict.items() if k != "update_step"}
            test_fetch_res = sess.run(fetch, feed_dict=test_feed_dict)
            print("Test Loss", test_fetch_res['loss'])
