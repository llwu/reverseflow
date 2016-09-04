import tensorflow as tf
import numpy as np

## Comparisons
## 1. Search other x in X to find x* = argmin(|f(x), y|)
## 2. Search other theta to find theta* and corresponding x* such that error node = zero
## 3. Serching over parameters for neural network to find h:Y->X such that argmin(|f(h(y)), y|)
## 4. Search over


def accumulate_mean_error(errors):
    return tf.add_n(errors)/len(errors)

def evaluate_loss(loss):
    """Compute |f(x), y|"""

    sess.run(loss)

def gen_y(graph, out_tensors):
    """For a function f:X->Y, generate a dataset of valid elemets y"""
    sess = tf.Session(graph=graph)
    initializer = tf.initialize_all_variables()
    sess.run(initializer)
    outputs = sess.run(out_tensors)
    sess.close()
    return outputs


def minimize_error(loss, inv_g, y_batch, sess, max_iterations=10000):
    errors = inv_g.get_collection("errors")
    node_loss = accumulate_mean_error(errors)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(node_loss)
    init = tf.initialize_all_variables()

    sess.run(init)
    node_loss_data = []
    feed = {"train_step":train_step, "node_loss":node_loss}
    for i in range(max_iterations):
        output = sess.run(feed, feed_dict=y_batch)
        node_loss_data.append(output['node_loss'])
        print("i",i, output)
    return node_loss_data

def gen_loss_model(in_out, y_batch, sess):
    losses = []
    variables = []
    # Create a variable to store y' = f(x') for each output
    out_tensors = in_out["outputs"]
    for i, out_tensor in enumerate(out_tensors):
        assert out_tensors[i].get_shape() == y_batch[i].shape
        var = tf.Variable(y_batch[i], trainable=False)
        variables.append(var)
        # loss = tf.nn.l2_loss(out_tensor - var)
        loss = tf.reduce_sum(tf.abs(out_tensor - var))
        losses.append(loss)

    loss = accumulate_mean_error(losses)
    return loss


def evaluate(loss, in_out, sess, max_iterations=10000):
    """
    Solve inverse problem by search over inputs
    Given a function f, and y_batch, find x_batch s.t. f(x_batch) = y
    """
    print("LOSS", loss)
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
    init = tf.initialize_all_variables()
    sess.run(init)
    feed = {"train_step":train_step, "loss":loss}
    feed.update(in_out)
    loss_data = []
    for i in range(max_iterations):
        output = sess.run(feed)
        loss_data.append(output['loss'])
        print("i",i, output)
    return loss_data
