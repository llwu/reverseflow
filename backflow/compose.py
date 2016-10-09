import tensorflow as tf

What inputs should you be able to replace?
Any?


def compose(outputs, inputs):
    """Pipe the outputs into the inputs"""

    # Check_types are consistent
    assert len(outputs) == len(inputs), "Tried to pipe %s outputs into %s input", % len(outputs), len(inputs)
    assert all([type_compatible for])

    g = tf.Graph()
    for op in outputs:
        tf.



def accumulate_errors(pinv, within_reduce = tf.reduce_mean, inter_reduce = tf.reduce_mean):
    "Accumulate all the errors of a parametric inverse into a scalar"
    # TODO
