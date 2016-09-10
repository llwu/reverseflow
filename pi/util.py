import tensorflow as tf
import numpy as np

iden = tf.identity



def infinite_input(gen_graph, batch_size):
    generator_graph = tf.Graph()
    with generator_graph.as_default() as g:
        in_out_var = gen_graph(g, batch_size, False)
        sess = tf.Session(graph=generator_graph)
        init = tf.initialize_all_variables()

    while True:
        with generator_graph.as_default() as g:
            sess.run(init)
            output = sess.run(in_out_var['outputs'])
        yield output


def infinite_samples(sampler, shape):
    while True:
        yield sampler(*shape)


def dictionary_gen(x):
    while True:
        yield {k: next(v) for k, v in x.items()}

def tensor_type(dtype, shape, name):
    """Creates a dict for type of tensor"""
    return {'dtype': dtype, 'shape': shape, 'name': name}


def add_many_to_collection(graph, name, tensors):
    for t in tensors:
        graph.add_to_collection(name, t)


def dims_bar_batch(t):
    """Get dimensions of a tensor exluding its batch dimension (first one)"""
    return np.arange(1, t.get_shape().ndims)


def ph_or_var(dtype, shape, name, is_placeholder=False):
    if is_placeholder:
        return tf.placeholder(dtype, shape=shape, name=name)
    else:
        return tf.Variable(tf.random_uniform(shape, dtype=dtype), name=name)


def same(xs):
    """All elements in xs are the same"""
    if len(xs) == 0:
        return True
    else:
        x1 = xs[0]
        for xn in xs:
            if xn != x1:
                return False

    return True


def placeholder_like(tensor, name, shape=None, dtype=None):
    """Create a placeholder like tensor with its name"""
    if shape is None:
        shape = tensor.get_shape()
    if dtype is None:
        dtype = tensor.dtype
    return tf.placeholder(dtype, shape, name=name)


def smthg_like(x, smthg):
    """
    Like ones_like but not one smthg
    """
    return tf.fill(x.get_shape(), smthg)


def is_constant(tensor):
    """Determine whether a tensor is constant"""
    print("WTD", tensor)
    sess = tf.Session(graph=tensor.graph)
    try:
        tensor.eval(session=sess)
    except tf.errors.InvalidArgumentError as e:
        # print(type(e), e)
        return False
    return True
