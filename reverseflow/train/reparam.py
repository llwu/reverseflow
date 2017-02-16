"""Reparameterization"""
from arrows.apply.propagate import propagate
from arrows.port_attributes import *
from arrows.compositearrow import CompositeArrow
from arrows.tfarrow import TfArrow
from arrows.util.generators import *
from typing import Tuple, Callable
from reverseflow.train.common import *
import numpy as np
EPS = 1e-5


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
    port_attr = propagate(comp_arrow)
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
        sqrdiff = tf.abs(diff)
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

def attach(tensor, gen):
    while True:
        res = next(gen)
        yield {tensor: res}

def gen_gens(ts, data, batch_size):
    return [attach(ts[i], infinite_batches(data[i], batch_size)) \
                 for i in range(len(data))]

def reparam_arrow(arrow: Arrow,
                  theta_ports: Sequence[Port],
                  train_data: List,
                  test_data: List,
                  batch_size,
                  error_filter=is_error_port,
                  **kwargs) -> CompositeArrow:

    # FIXME: Add broadcasting nodes
    with tf.name_scope(arrow.name):
        input_tensors = gen_input_tensors(arrow, param_port_as_var=False)
        port_grab = {p: None for p in theta_ports}
        output_tensors = arrow_to_graph(arrow, input_tensors, port_grab=port_grab)

    theta_tensors = list(port_grab.values())
    y_tensors = [t for i, t in enumerate(input_tensors) if not is_param_port(arrow.in_ports()[i])]
    param_tensors = [t for i, t in enumerate(input_tensors) if is_param_port(arrow.in_ports()[i])]
    error_tensors = [t for i, t in enumerate(output_tensors) if error_filter(arrow.out_ports()[i])]
    assert len(param_tensors) > 0, "Must have parametric inports"
    assert len(error_tensors) > 0, "Must have error outports"
    assert len(y_tensors) == len(train_data)

    # Make parametric inputs
    train_gen_gens = []
    test_gen_gens = []

    param_feed_gens = []
    for t in param_tensors:
        shape = tuple(t.get_shape().as_list())
        gen = infinite_samples(np.random.rand, batch_size, shape)
        param_feed_gens.append(attach(t, gen))
    train_gen_gens += param_feed_gens
    test_gen_gens += param_feed_gens

    train_gen_gens += gen_gens(y_tensors, train_data, batch_size)
    test_gen_gens += gen_gens(y_tensors, test_data, batch_size)

    loss2 = accumulate_losses(error_tensors)

    # Generate permutation tensors
    # with tf.name_scope("placeholder"):
    #     perm = tf.placeholder(shape=(None,), dtype='int32', name='perm')
    #     perm_idx = tf.placeholder(shape=(None,), dtype='int32', name='perm_idx')
    # perm_feed_gen = [perm_gen(batch_size, perm, perm_idx)]
    # train_gen_gens += perm_feed_gen
    # test_gen_gens += perm_feed_gen
    #
    # euclids = [pairwise_dists(t, perm, perm_idx) for t in theta_tensors]
    # min_gap_losses = [minimum_gap(euclid) for euclid in euclids]
    # min_gap_loss = tf.reduce_min(min_gap_losses)
    # mean_gap_losses = [mean_gap(euclid) for euclid in euclids]
    # mean_gap_loss = tf.reduce_mean(mean_gap_losses)
    #
    # lmbda = 10.0
    # min_gap_loss = lmbda * min_gap_loss
    # loss = loss2 / min_gap_loss
    loss = loss2
    sess = tf.Session()
    fetch = gen_fetch(sess, loss)
    fetch['input_tensors'] = input_tensors
    fetch['output_tensors'] = output_tensors
    # fetch['to_print'] = {'min_gap_loss': min_gap_loss,
    #                      'mean_gap_loss': mean_gap_loss,
    #                      'loss2': loss2}

    train_loop(sess,
               fetch,
               train_gen_gens,
               test_gen_gens,
               **kwargs)
