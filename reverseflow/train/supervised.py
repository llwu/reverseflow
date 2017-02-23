"""Supervised Training of Arrows"""
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.port_attributes import is_error_port, has_port_label, is_in_port, is_param_port
from arrows.util.misc import getn, inn
from arrows.util.generators import infinite_batches
from reverseflow.train.common import extract_tensors, prep_save, train_loop, gen_fetch
from reverseflow.train.common import accumulate_losses, gen_update_step
from arrows.util.misc import print_one_per_line

from typing import List, Generator
import tensorflow as tf

def okok(batch_size, input_data, output_data, input_tensors, output_tensors):
    """
    Generator for supervised input
    Args:
        input_data:
        output_data:
        input_tensors:
        output_tensors:
    """
    assert len(input_data) == len(input_tensors)
    assert len(output_data) == len(output_tensors)
    inp_gens = [infinite_batches(i, batch_size, shuffle=False) for i in input_data]
    out_gens = [infinite_batches(i, batch_size, shuffle=False) for i in output_data]
    while True:
        inps = [next(gen) for gen in inp_gens]
        outs = [next(gen) for gen in out_gens]
        input_feed = {input_tensors[i]: inps[i] for i in range(len(inps))}
        output_feed = {output_tensors[i]: outs[i] for i in range(len(outs))}
        both = {}
        both.update(input_feed)
        both.update(output_feed)
        yield both


def supervised_train(arrow: Arrow,
                     train_input_data: List[Generator],
                     train_output_data: List[Generator],
                     test_input_data: List[Generator],
                     test_output_data: List[Generator],
                     callbacks=None,
                     options=None) -> CompositeArrow:
    callbacks = [] if callbacks is None else callbacks
    options = {} if options is None else options
    grabs = ({'input': lambda p: is_in_port(p) and not is_param_port(p) and not has_port_label(p, 'train_output'),
              'train_output': lambda p: has_port_label(p, 'train_output'),
              'supervised_error':  lambda p: has_port_label(p, 'supervised_error'),
              'sub_arrow_error':  lambda p: has_port_label(p, 'sub_arrow_error'),
              'inv_fwd_error':  lambda p: has_port_label(p, 'inv_fwd_error')})
    # Not all arrows will have these ports
    optional = ['sub_arrow_error', 'inv_fwd_error', 'param']
    tensors = extract_tensors(arrow, grabs=grabs, optional=optional)

    train_feed_gens = [okok(options['batch_size'], train_input_data, train_output_data,
                            tensors['input'], tensors['train_output'])]

    test_feed_gens = [okok(options['batch_size'], test_input_data, test_output_data,
                          tensors['input'], tensors['train_output'])]


    # All losses
    loss_dict = {}
    for loss in ['error', 'sub_arrow_error', 'inv_fwd_error', 'supervised_error']:
        if loss in tensors:
            loss_dict[loss] = accumulate_losses(tensors[loss])

    # error to minimize
    error = options['error'] if 'error' in options else 'error'
    loss_to_min = accumulate_losses(tensors[error])
    losses = [loss_to_min]
    loss_updates = [gen_update_step(loss) for loss in losses]
    loss_ratios = [1]

    sess = tf.Session()
    fetch = gen_fetch(sess, **options)
    fetch['input_tensors'] = tensors['input']
    fetch['output_tensors'] = tensors['output']
    fetch['loss'] = loss_dict

    if inn(options, 'save', 'dirname', 'params_file', 'load'):
        ops = prep_save(sess, *getn(options, 'save', 'dirname', 'params_file', 'load'))
        options.update(ops)

    train_loop(sess,
               loss_updates,
               fetch,
               train_feed_gens,
               test_feed_gens,
               loss_ratios=loss_ratios,
               callbacks=callbacks,
               **options)

# One issue is that the input and output generators should not be completely
# independent
