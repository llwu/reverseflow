import tensorflow as tf
import numpy as np

## Comparisons
## 1. Search other x in X to find x* = argmin(|f(x), y|)
## 2. Search other theta to find theta* and corresponding x* such that error node = zero
## 3. Serching over parameters for neural network to find h:Y->X such that argmin(|f(h(y)), y|)
## 4. Search over

# 1. Need to repeat it over many data points
# 2. Need stopping criteria
# 3. Get loss from other methods
# 3. Get out dict
# Need to collec the time

# So there's a pretty high degree of variance between runs.
# depending on initial point and on learning_rate
# need to run more than one example.
# 

def accumulate_mean_error(errors):
    return tf.add_n(errors)/len(errors)

def evaluate_loss(loss, output, variables, sess):
    """Compute |f(x) - y|"""
    feed = {variable: output[var_name] for var_name, variable in variables.items()}
    return sess.run(loss, feed_dict=feed)

def gen_y(graph, out_tensors):
    """For a function f:X->Y, generate a dataset of valid elemets y"""
    sess = tf.Session(graph=graph)
    initializer = tf.initialize_all_variables()
    sess.run(initializer)
    outputs = sess.run(out_tensors)
    sess.close()
    return outputs


def minimize_error(loss, inv_g, inputs, out_map, y_batch, variables, sess,
                   max_iterations=10000):
    errors = inv_g.get_collection("errors")
    node_loss = accumulate_mean_error(errors)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(node_loss)
    init = tf.initialize_all_variables()

    sess.run(init)
    node_loss_data = []
    std_losses = []
    fetch = {"train_step":train_step, "node_loss":node_loss}
    fetch.update(out_map)
    print("Fetchy", fetch)
    input_feed = {inputs[k]:y_batch[k] for k in y_batch.keys()}
    print("FEEDY", input_feed)
    for i in range(max_iterations):
        output = sess.run(fetch, feed_dict=input_feed)
        std_loss = evaluate_loss(loss, output, variables, sess)
        std_losses.append(std_loss)
        node_loss_data.append(output['node_loss'])
        print("i",i, output['node_loss'], std_loss)
    return node_loss_data, std_losses

def gen_loss_model(in_out, y_batch, sess):
    losses = []
    variables = {}
    # Create a variable to store y' = f(x') for each output
    out_tensors = in_out["outputs"]
    for out_name, out_tensor in out_tensors.items():
        assert out_tensor.get_shape() == y_batch[out_name].shape
        var = tf.Variable(y_batch[out_name], trainable=False)
        variables[out_name] = var
        loss = tf.reduce_sum(tf.abs(out_tensor - var))
        losses.append(loss)

    loss = accumulate_mean_error(losses)
    return loss, variables


def evaluate(loss, in_out, sess, max_iterations=10000):
    """
    Solve inverse problem by search over inputs
    Given a function f, and y_batch, find x_batch s.t. f(x_batch) = y
    """
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()
    sess.run(init)
    fetch = {"train_step":train_step, "loss":loss}
    # fetch.update(in_out)
    print("FEEDLY", fetch)
    loss_data = []
    for i in range(max_iterations):
        output = sess.run(fetch)
        loss_data.append(output['loss'])
        print("i",i, output)
    return loss_data
