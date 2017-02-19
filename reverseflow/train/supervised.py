"""Supervised Training of Arrows"""
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from reverseflow.train.reparam import extract_tensors
from typing import List, Generator

def okOK(input_data, output_data, input_tensors, output_tensors):
    assert len(input_data) == len(input_tensors)
    assert len(output_data) == len(output_tensors)
    [attach]


def supervised_train(arrow: Arrow,
                     train_data: List[Generator],
                     test_data: List[Generator],
                     callbacks=[],
                     error_filter=is_error_port,
                     options={}) -> CompositeArrow:

    batch_size = options['batch_size']
    tensors = extract_tensors(arrow)
    inp_tensors = tensors['input']

    # Attach each tensor to its generator
    train_gen_gens = [attach(inp_tensors[i], train_data[i]) for i in range(n)]
    test_gen_gens = [attach(inp_tensors[i], test_data[i]) for i in range(n)]

    # Accumulate error tensors into single loss term
    sound_loss = accumulate_losses(tensors['error'])
    losses = [sound_loss]
    loss_ratios = [1]

    # Do the grap that tensorflow needs done
    loss_updates = [gen_update_step(loss) for loss in losses]
    sess = tf.Session()
    fetch = gen_fetch(sess, **options)
    fetch['input_tensors'] = input_tensors
    fetch['output_tensors'] = output_tensors
    fetch['loss'] = losses
    fetch['to_print'] = {'min_gap_loss': min_gap_loss,
                         'mean_gap_loss': mean_gap_loss,
                         'sound_loss': sound_loss}


    train_loop(sess,
               loss_updates,
               fetch,
               train_gen_gens,
               test_gen_gens,
               loss_ratios=loss_ratios,
               callbacks=callbacks,
               **options)
