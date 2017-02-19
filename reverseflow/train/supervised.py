"""Supervised Training of Arrows"""
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.port_attributes import is_error_port
from reverseflow.train.common import extract_tensors, prep_save, train_loop
from typing import List, Generator

from reverseflow.train.common import accumulate_losses, gen_update_step
import tensorflow as tf

def okOK(input_data, output_data, input_tensors, output_tensors):
    assert len(input_data) == len(input_tensors)
    assert len(output_data) == len(output_tensors)
    [attach]


def supervised_train(arrow: Arrow,
                     train_input_data: List[Generator],
                     train_output_data: List[Generator],
                     test_input_data: List[Generator],
                     test_output_data: List[Generator],
                     callbacks=[],
                     options={}) -> CompositeArrow:

    tensors = extract_tensors(arrow, error_filter=is_error_port)
    inp_tensors = tensors['input']

    # Attach each tensor to its generator
    train_gen_gens = [attach(inp_tensors[i], train_data[i]) for i in range(n)]
    test_gen_gens = [attach(inp_tensors[i], test_data[i]) for i in range(n)]

    # Accumulate error tensors into single loss term
    sound_loss = accumulate_losses(tensors['error'])
    losses = [sound_loss]
    loss_updates = [gen_update_step(loss) for loss in losses]

    sess = tf.Session()
    fetch = gen_fetch(sess, **options)
    fetch['input_tensors'] = tensors['input']
    fetch['output_tensors'] = tensors['output']
    fetch['loss'] = losses

    if inn(options, 'save', 'sfx', 'params_file', 'load'):
        ops = prep_save(sess, *getn(options, 'save', 'sfx', 'params_file', 'load'))
        options.update(ops)

    train_loop(sess,
               loss_updates,
               fetch,
               train_gen_gens,
               test_gen_gens,
               loss_ratios=loss_ratios,
               callbacks=callbacks,
               **options)
