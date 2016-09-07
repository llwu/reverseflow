import tensorflow as tf
import numpy as np

def dims_bar_batch(t):
    """Get the dimensions of a tensor exluding its batch dimension (first one)"""
    return np.arange(1, t.get_shape().ndims)

def ph_or_var(dtype, shape, name, is_placeholder=False):
    if is_placeholder:
        return tf.placeholder(dtype, shape=shape, name=name)
    else:
        return tf.Variable(tf.random_uniform(shape,dtype=dtype), name=name)

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

iden = tf.identity

def placeholder_like(tensor, name, shape=None, dtype=None):
    if shape is None: shape = tensor.get_shape()
    if dtype is None: dtype = tensor.dtype
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
