import tensorflow as tf
import numpy as np

def gen_y(graph, out_tensors):
    """For a function f:X->Y, generate a dataset of valid elemets y"""
    sess = tf.Session(graph=graph)
    initializer = tf.initialize_all_variables()
    sess.run(initializer)
    outputs = sess.run(out_tensors)
    sess.close()
    return outputs
