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

def xy(x, y, d = tf.abs):
    d(x - y)

def evaluate(fwd_model, y_batch, sess, max_iterations=10000):
    """
    Solve inverse problem by search over inputs
    Given a function f, and y_batch, find x_batch s.t. f(x_batch) = y
    """
    losses = []
    # Create a variable to store y' = f(x') for each output
    for out_tensor in out_tensors:
        tf.Variable(out_tensor)
        losses.append(out_tensor)

    loss = ...
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(max_iterations=10000):
        output = sess.run({"t":train_step,"loss":loss}, feed_dict=input_feed)
        print(output)
