"""Tensroflow specific utilities"""
import tensorflow as tf

def tf_eval(f, *args, **kwargs):
    """Eval a tensor"""
    sess = tf.InteractiveSession()
    ret = f(*args, **kwargs).eval(session=sess)
    sess.close()
    return ret
