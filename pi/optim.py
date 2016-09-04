import tensorflow as tf
import numpy as np

def accumulate_mean_error(errors):
    return tf.add_n(errors)/len(errors)

def minimize_error(inv_g, input_feed, sess):
    # params = inv_g.get_collection("params")
    errors = inv_g.get_collection("errors")
    loss = accumulate_mean_error(errors)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    print("loss", loss)

    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(10000):
        output = sess.run({"t":train_step,"loss":loss}, feed_dict=input_feed)
        print(output)


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

def evaluate(y_batch, out_tensors, sess, max_iterations=10000):
    """
    Solve inverse problem by search over inputs
    Given a function f, and y_batch, find x_batch s.t. f(x_batch) = y
    """
    losses = []
    variables = []
    # Create a variable to store y' = f(x') for each output
    for i, out_tensor in enumerate(out_tensors):
        assert out_tensors[i].get_shape() == y_batch[i].shape
        var = tf.Variable(y_batch[i], trainable=False)
        variables.append(var)
        # loss = tf.nn.l2_loss(out_tensor - var)
        loss = tf.reduce_sum(tf.abs(out_tensor - var))
        losses.append(out_tensor)

    loss = accumulate_mean_error(losses)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(max_iterations):
        output = sess.run({"t":train_step,"loss":loss})
        print(output)
