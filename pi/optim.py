import tensorflow as tf
import numpy as np
import time
import collections
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

def evaluate_loss(loss, output, variables, y_batch, target_outputs, sess):
    """Compute |f(x) - y|"""
    target_feed = {target_outputs[k]:y_batch[k] for k in y_batch.keys()}
    feed = {variable: output[var_name] for var_name, variable in variables.items()}
    feed.update(target_feed)
    return sess.run(loss, feed_dict=feed)

def gen_y(out_tensors):
    """For a function f:X->Y, generate a dataset of valid elemets y"""
    graph = list(out_tensors.values())[0].graph
    sess = tf.Session(graph=graph)
    initializer = tf.initialize_all_variables()
    sess.run(initializer)
    outputs = sess.run(out_tensors)
    sess.close()
    return outputs

def gen_loss_model(in_out, sess):
    """
    elementwise_loss_per_batch_per_output : {x:[1,2,3],y:[[1,2,3],[4,5,6]]}
    loss_per_batch_per_output: {x:[1,2,3,4,..,n_batch],y:[1,2,3,4,..n]}
    loss_per_batch:
    """
    # Create a placeholder to store y' = f(x') for each output
    out_tensors = in_out["outputs"]
    target_outputs = {k: tf.placeholder(v.dtype, shape=v.get_shape(),
                      name="%s_target" % k) for k, v in out_tensors.items()}
    absdiffs = {k: tf.abs(out_tensors[k] - target_outputs[k])
                for k in out_tensors.keys()}
    mean_loss_per_batch_per_op = {k: tf.reduce_mean(v,
                                  reduction_indices=np.arange(1,v.get_shape().ndims)) for k,v in absdiffs.items()}
    mean_loss_per_batch = tf.add_n(list(mean_loss_per_batch_per_op.values())) / len(absdiffs)
    loss = tf.reduce_mean(mean_loss_per_batch)
    return loss, absdiffs, mean_loss_per_batch_per_op, mean_loss_per_batch, target_outputs


def min_param_error(loss, inv_g, inputs, out_map, y_batch, variables,
                    target_outputs, sess, max_iterations=None,
                    max_time=10.0):
    """
    Solve inverse problem by searching over parameter values which minimize
    error range error.
    """
    assert (max_iterations is None) != (max_time is None), "set stop time xor max-iterations"

    # Generate loss_op
    errors = inv_g.get_collection("errors")
    domain_loss = accumulate_mean_error(errors)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(domain_loss)
    init = tf.initialize_all_variables()

    sess.run(init)
    fetch = {"train_step":train_step, "domain_loss":domain_loss}
    fetch.update(out_map)
    input_feed = {inputs[k]:y_batch[k] for k in y_batch.keys()}

    domain_loss_data = []
    std_loss_data = []

    total_time = previous_time = 0.0
    domain_loss_hist = collections.OrderedDict()
    std_loss_hist = collections.OrderedDict()

    curr_time_slice = 0
    domain_loss_hist[curr_time_slice] = np.array([])
    std_loss_hist[curr_time_slice] = np.array([])

    i = 0
    while True:
        if max_iterations is not None and i > max_iterations:
            break
        elif max_time is not None and total_time > max_time:
            break
        i = i + 1
        start_time = time.clock()
        output = sess.run(fetch, feed_dict=input_feed)
        end_time = time.clock()
        elapsed = end_time - start_time
        total_time = elapsed + total_time

        std_loss = evaluate_loss(loss, output, variables, y_batch, target_outputs, sess)
        std_loss_data.append(std_loss)
        domain_loss_data.append(output['domain_loss'])

        if output['loss'] < 0.3:
            # Generate new elements of y and reinitialize x
            y_batch = gen_y(in_out["outputs"])
            target_feed = {target_outputs[k]: y_batch[k] for k in y_batch.keys()}
            loss_hist[curr_time_slice] = np.concatenate([loss_hist[curr_time_slice], output['batch_loss']])

        if total_time - previous_time > time_grain:
            previous_time = total_time
            loss_hist[curr_time_slice] = np.concatenate([loss_hist[curr_time_slice], output['batch_loss']])
            curr_time_slice += 1
            loss_hist[curr_time_slice] = []

        print("i", i, output['domain_loss'], std_loss)
    return domain_loss_data, std_loss_data


def min_fx_y(loss_op, mean_loss_per_batch_op, in_out, target_outputs, y_batch,
             sess, max_iterations=None, max_time=10.0, time_grain=1):
    """
    Solve inverse problem by search over inputs
    Given a function f, and y_batch, find x_batch s.t. f(x_batch) = y
    """
    assert (max_iterations is None) != (max_time is None), "set stop time xor max-iterations"

    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss_op)
    init = tf.initialize_all_variables()
    sess.run(init)
    fetch = {"train_step": train_step, "loss": loss_op,
             "batch_loss": mean_loss_per_batch_op}
    target_feed = {target_outputs[k]: y_batch[k] for k in y_batch.keys()}
    loss_data = []
    total_time = previous_time = 0.0
    loss_hist = collections.OrderedDict()
    curr_time_slice = 0
    loss_hist[curr_time_slice] = np.array([])

    i = 0
    while True:
        if max_iterations is not None and i > max_iterations:
            break
        elif max_time is not None and total_time > max_time:
            break

        i = i + 1
        # Timing
        start_time = time.clock()
        output = sess.run(fetch, feed_dict=target_feed)
        end_time = time.clock()
        elapsed = end_time - start_time
        total_time = elapsed + total_time

        loss_data.append(output['loss'])
        print("i", i, output)
        # Start a new batch
        if output['loss'] < 0.3:
            # Generate new elements of y and reinitialize x
            y_batch = gen_y(in_out["outputs"])
            target_feed = {target_outputs[k]: y_batch[k] for k in y_batch.keys()}
            loss_hist[curr_time_slice] = np.concatenate([loss_hist[curr_time_slice], output['batch_loss']])

        if total_time - previous_time > time_grain:
            previous_time = total_time
            loss_hist[curr_time_slice] = np.concatenate([loss_hist[curr_time_slice], output['batch_loss']])
            curr_time_slice += 1
            loss_hist[curr_time_slice] = []


    return loss_data, loss_hist
