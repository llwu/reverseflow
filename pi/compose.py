import tensorflow as tf

## Composition of different tensorflow graphs


def accumulate_errors(pinv, within_reduce = tf.reduce_mean, inter_reduce = tf.reduce_mean):
    "Accumulate all the errors of a parametric inverse into a scalar"
    # TODO
