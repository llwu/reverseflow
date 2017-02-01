"""Tensroflow specific utilities"""
import tensorflow as tf

def tf_eval(f, *args):
    """Eval a tensor"""
    sess = tf.InteractiveSession()
    ret = f(*args).eval(session=sess)
    sess.close()
    return ret
