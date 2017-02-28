"""Reparameterization"""
from arrows.apply.propagate import propagate
from arrows.port_attributes import *
from arrows.compositearrow import CompositeArrow
from arrows.tfarrow import TfArrow
from arrows.util.generators import *
from arrows.util.io import mk_dir, save_dict_csv
from arrows.util.misc import getn, inn
from typing import Tuple, Callable
from reverseflow.train.common import *
from arrows.config import EPS
import numpy as np


def non_iden(perm):
    return [p for i, p in enumerate(perm) if i != p]


def non_iden_idx(perm):
    return [i for i, p in enumerate(perm) if i != p]


def perm_gen(batch_size, param_t, param_idx_t):
    """Generator of permutation and permutation indices"""
    while True:
        perm_data = np.arange(batch_size)
        np.random.shuffle(perm_data)
        non_iden_perm_data = non_iden(perm_data)
        perm_data_idx = np.array(non_iden_idx(perm_data))
        yield {param_t: non_iden_perm_data,
               param_idx_t: perm_data_idx}


def perm_capture(gen_data):
    return gen_data['perm_data'][0]


def perm_idx_capture(gen_data):
    return gen_data['perm_data'][1]


def reparam(comp_arrow: CompositeArrow,
            phi_shape: Tuple,
            nn_takes_input=True):
    """Reparameterize an arrow.  All parametric inputs now function of phi
    Args:
        comp_arrow: Arrow to reparameterize
        phi_shape: Shape of parameter input
    """
    reparam = CompositeArrow(name="%s_reparam" % comp_arrow.name)
    phi = reparam.add_port()
    set_port_shape(phi, phi_shape)
    make_in_port(phi)
    make_param_port(phi)
    n_in_ports = 1
    if nn_takes_input:
        n_in_ports += comp_arrow.num_in_ports() - comp_arrow.num_param_ports()
    nn = TfArrow(n_in_ports=n_in_ports, n_out_ports=comp_arrow.num_param_ports())
    reparam.add_edge(phi, nn.in_port(0))
    i = 0
    j = 1
    for port in comp_arrow.ports():
        if is_param_port(port):
            reparam.add_edge(nn.out_port(i), port)
            i += 1
        else:
            re_port = reparam.add_port()
            if is_out_port(port):
                make_out_port(re_port)
                reparam.add_edge(port, re_port)
            if is_in_port(port):
                make_in_port(re_port)
                reparam.add_edge(re_port, port)
                if nn_takes_input:
                    reparam.add_edge(re_port, nn.in_port(j))
                    j += 1
            if is_error_port(port):
                make_error_port(re_port)
            for label in get_port_labels(port):
                add_port_label(re_port, label)

    assert reparam.is_wired_correctly()
    return reparam

def pairwise_dists(t, perm, perm_idx):
    with tf.name_scope("pairwise_dists"):
        ts_shrunk = tf.gather(t, perm_idx)
        ts_permute = tf.gather(t, perm)
        diff = ts_permute - ts_shrunk + EPS
        sqrdiff = tf.square(diff) + EPS
        reduction_indices = list(range(1, sqrdiff.get_shape().ndims))
        euclids = tf.reduce_sum(sqrdiff, reduction_indices=reduction_indices) + EPS
        return euclids


def minimum_gap(euclids):
    with tf.name_scope("minimum_gap"):
        rp = tf.reduce_min(euclids)
    return rp

def mean_gap(euclids):
    with tf.name_scope("mean_gap"):
        rp = tf.reduce_mean(euclids)
    return rp


def reparam_train(arrow: Arrow,
                  extra_ports: Sequence[Port],
                  train_data: List[Generator],
                  test_data: List[Generator],
                  options=None) -> CompositeArrow:

    options = {} if options is None else options
    grabs = ({'input': lambda p: is_in_port(p) and not is_param_port(p) and not has_port_label(p, 'train_output'),
              'train_output': lambda p: has_port_label(p, 'train_output'),
              'pure_output': lambda p: is_out_port(p) and not is_error_port(p),
              'supervised_error':  lambda p: has_port_label(p, 'supervised_error'),
              'sub_arrow_error':  lambda p: has_port_label(p, 'sub_arrow_error'),
              'inv_fwd_error':  lambda p: has_port_label(p, 'inv_fwd_error')})
    # Not all arrows will have these ports
    optional = ['sub_arrow_error', 'inv_fwd_error', 'param', 'supervised_error', 'train_output']
    tensors = extract_tensors(arrow, grabs=grabs, optional=optional)

    # Make parametric inputs
    train_gen_gens = []
    test_gen_gens = []

    param_feed_gens = []
    for t in tensors['param']:
        shape = tuple(t.get_shape().as_list())
        gen = infinite_samples(np.random.rand, options['batch_size'], shape)
        param_feed_gens.append(attach(t, gen))
    train_gen_gens += param_feed_gens
    test_gen_gens += param_feed_gens

    n = len(tensors['input'])
    train_gen_gens += [attach(tensors['input'][i], train_data[i]) for i in range(n)]
    test_gen_gens += [attach(tensors['input'][i], test_data[i]) for i in range(n)]

    sound_loss = accumulate_losses(tensors['error'])



    # Generate permutation tensors
    with tf.name_scope("placeholder"):
        perm = tf.placeholder(shape=(None,), dtype='int32', name='perm')
        perm_idx = tf.placeholder(shape=(None,), dtype='int32', name='perm_idx')
    perm_feed_gen = [perm_gen(options['batch_size'], perm, perm_idx)]
    train_gen_gens += perm_feed_gen
    test_gen_gens += perm_feed_gen
    extra = tensors['output']
    butim3d = tf.concat(tensors['pure_output'], axis=1)
    extra = [butim3d]
    euclids = [pairwise_dists(t, perm, perm_idx) for t in extra]
    min_gap_losses = [minimum_gap(euclid) for euclid in euclids]
    min_gap_loss = tf.reduce_sum(min_gap_losses)
    # mean_gap_losses = [mean_gap(euclid) for euclid in euclids]
    # mean_gap_loss = tf.reduce_mean(mean_gap_losses)
    lmbda = 10.0
    min_gap_loss = min_gap_loss * lmbda
    losses = [sound_loss - min_gap_loss]
    loss_ratios = [1]
    loss_updates = [gen_update_step(loss) for loss in losses]
    # options['debug'] = True

    # All losses
    loss_dict = {}
    for loss in ['error', 'sub_arrow_error', 'inv_fwd_error', 'supervised_error']:
        if loss in tensors:
            loss_dict[loss] = accumulate_losses(tensors[loss])
    # loss_dict['mean_gap'] = mean_gap_loss
    loss_dict['min_gap_losses'] = min_gap_losses
    loss_dict['min_gap'] = min_gap_loss
    loss_dict['sound_loss'] = sound_loss

    sess = tf.Session()
    fetch = gen_fetch(sess, **options)
    fetch['input_tensors'] = tensors['input']
    fetch['param_tensors'] = tensors['param']
    fetch['output_tensors'] = tensors['output']
    fetch['loss'] = loss_dict

    if inn(options, 'save', 'sfx', 'params_file', 'load'):
        ops = prep_save(sess, *getn(options, 'save', 'sfx', 'params_file', 'load'))
        options.update(ops)

    train_loop(sess,
               loss_updates,
               fetch,
               train_gen_gens,
               test_gen_gens,
               loss_ratios=loss_ratios,
               **options)
