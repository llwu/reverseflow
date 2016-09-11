import tensorflow as tf
import numpy as np
import time
import collections
from pi.util import *
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

def evaluate_loss(loss_op, mean_loss_per_batch_op, output, variables, y_batch, target_outputs, sess):
    """Compute |f(x) - y|"""
    fetch = {"loss": loss_op, "batch_loss": mean_loss_per_batch_op}
    target_feed = {target_outputs[k]: y_batch[k] for k in y_batch.keys()}
    feed = {variable: output[var_name] for var_name, variable in variables.items()}
    feed.update(target_feed)
    return sess.run(fetch, feed_dict=feed)


def gen_loss_evaluator(loss_op, mean_loss_per_batch_op, target_outputs, variables, sess):
    """
    Generates a loss closure which evaluates Compute |f(x) - y|
    """
    fetch = {"loss": loss_op, "batch_loss": mean_loss_per_batch_op}

    def check_loss(inv_output_values, inv_input_values):
        feed = {variable: inv_output_values[var_name] for var_name, variable in variables.items()}
        target_feed = {target_outputs[k]: inv_input_values[k] for k in inv_input_values.keys()}
        feed.update(target_feed)
        return sess.run(fetch, feed_dict=feed)
    return check_loss

def gen_y(out_tensors):
    """For a function f:X->Y, generate a dataset of valid elemets y"""
    graph = list(out_tensors.values())[0].graph
    sess = tf.Session(graph=graph)
    initializer = tf.initialize_all_variables()
    sess.run(initializer)
    outputs = sess.run(out_tensors)
    sess.close()
    return outputs

def multi_io_loss(a_tensors, b_tensors):
    """
    For a vector
    a_tensors : {name: tf.Tensor}
    b_tensors : {name: tf.Tensor}
    """
    # assert same_kinda_tensors(a_tensors, b_tensors), "mismatch in tensor shapes"
    absdiffs = {k: tf.abs(a_tensors[k] - b_tensors[k])
                for k in a_tensors.keys()}
    mean_loss_per_batch_per_op = {k: tf.reduce_mean(v, reduction_indices=dims_bar_batch(v)) for k,v in absdiffs.items()}
    mean_loss_per_batch = tf.add_n(list(mean_loss_per_batch_per_op.values())) / len(absdiffs)
    loss = tf.reduce_mean(mean_loss_per_batch)
    return loss, absdiffs, mean_loss_per_batch_per_op, mean_loss_per_batch


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
    return multi_io_loss(out_tensors, target_outputs) + (target_outputs,)

def timed_run(sess, **kwargs):
    start_time = time.clock()
    output = sess.run(**kwargs)
    end_time = time.clock()
    elapsed = end_time - start_time
    return elapsed, output

def nnet(fwd_f, fwd_inputs, fwd_outputs, inv_inp_gen, nnet_template,
         sess, max_iterations=None, max_time=1.0, time_grain=1.0,
         **template_kwargs):
    """
    Train a neural network f to map y to x such that f(x) = y.
    loss = |f((f-1(y))) - y|
    fwd_inputs : (tf.Tensor) - placeholders for forward function inputs (used for shape only)
    fwd_outputs : (tf.Tensor) - placeholders for forward function outputs (used for shape only)
    template :: f - a function that builds a neural network given in/out types
                returns network inputs and outputs
    """
    # wwhats next
    inv_inputs = {k: tf.placeholder(v.dtype, shape=v.get_shape()) for k, v  in fwd_outputs.items()}
    nnet_output_shapes = {k: fwd_inp.get_shape().as_list() for k, fwd_inp in fwd_inputs.items()}
    nnet_outputs, nnet_params = nnet_template(inv_inputs, nnet_output_shapes, **template_kwargs)

    ## Take outputs of neural network and apply to input of function
    fwd_outputs = fwd_f(nnet_outputs, **template_kwargs)
    loss_op, absdiffs, mean_loss_per_batch_per_op, mean_loss_per_batch_op = multi_io_loss(fwd_outputs, inv_inputs)

    # Training
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_op)
    sess.run(tf.initialize_all_variables())

    total_time = previous_time = 0.0
    std_loss_hist = collections.OrderedDict()

    curr_time_slice = 0
    std_loss_hist[curr_time_slice] = np.array([])

    ## Generate y data
    fetch = {"loss": loss_op, "batch_loss": mean_loss_per_batch_op,
             "train_step": train_step}
    i = 0
    while True:
        if max_iterations is not None and i > max_iterations:
            break
        elif max_time is not None and total_time > max_time:
            break
        i = i + 1

        inv_inp_batch = next(inv_inp_gen)
        input_feed = {inv_inputs[k]: inv_inp_batch[k] for k in inv_inp_batch.keys()}

        ## Training Step
        elapsed, output = timed_run(sess, fetches=fetch, feed_dict=input_feed)
        total_time = elapsed + total_time
        std_loss = output

        std_loss_hist[curr_time_slice] = np.concatenate([std_loss_hist[curr_time_slice], std_loss['batch_loss']])

        if total_time - previous_time > time_grain:
            previous_time = total_time
            curr_time_slice += 1
            std_loss_hist[curr_time_slice] = []
        print("nnet-loss", std_loss['loss'])

    return std_loss_hist, total_time


def enhanced_pi(inv_g, inv_inputs, inv_inp_gen, shrunk_params, shrunk_param_gen,
                inv_outputs, check_loss, sess, max_iterations=None, max_time=10.0, time_grain=1.0):
    """
    Train a neural network enhanced parametric inverse
    inv_inp : {name: tf.Tesor}
    inv_inp_gen : coroutine -> {inv_inp_name: np.array}
    """
    ## Feed in a sample of inputs from the cross product y X parameer inputs
    ## minimize node error
    errors = inv_g.get_collection("errors")
    batch_domain_loss = accumulate_mean_error(errors)
    domain_loss = tf.reduce_mean(batch_domain_loss)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(domain_loss)
    init = tf.initialize_all_variables()
    sess.run(init)

    fetches = {"train_step": train_step, "domain_loss": domain_loss,
               "batch_domain_loss": batch_domain_loss}
    fetches.update(inv_outputs)

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

        ## Generate data
        inv_inp_batch = next(inv_inp_gen)
        input_feed = {inv_inputs[k]: inv_inp_batch[k] for k in inv_inp_batch.keys()}
        param_batch = next(shrunk_param_gen)
        param_feed = {shrunk_params[k]: param_batch[k] for k in param_batch.keys()}
        input_feed.update(param_feed)

        ## Training Step
        elapsed, output = timed_run(sess, fetches=fetches, feed_dict=input_feed)
        total_time = elapsed + total_time

        inv_outputs_values = {k: output[k] for k in inv_outputs.keys()}
        std_loss = check_loss(inv_outputs_values, inv_inp_batch)

        domain_loss_hist[curr_time_slice] = np.concatenate([domain_loss_hist[curr_time_slice], output['batch_domain_loss']])
        std_loss_hist[curr_time_slice] = np.concatenate([std_loss_hist[curr_time_slice], std_loss['batch_loss']])

        if total_time - previous_time > time_grain:
            previous_time = total_time
            curr_time_slice += 1
            domain_loss_hist[curr_time_slice] = []
            std_loss_hist[curr_time_slice] = []
        print("domain", output["domain_loss"], "std:", std_loss["loss"])

    return domain_loss_hist, std_loss_hist, total_time


def min_param_error(inv_g, inv_inputs, inv_inp_gen, inv_outputs, check_loss,
                    sess, max_iterations=None, max_time=10.0, time_grain=1.0):
    """
    Train a neural network enhanced parametric inverse
    inv_inp : {name: tf.Tesor}
    inv_inp_gen : coroutine -> {inv_inp_name: np.array}
    """
    ## Feed in a sample of inputs from the cross product y X parameer inputs
    ## minimize node error
    errors = inv_g.get_collection("errors")
    batch_domain_loss = accumulate_mean_error(errors)
    domain_loss = tf.reduce_mean(batch_domain_loss)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(domain_loss)
    init = tf.initialize_all_variables()
    sess.run(init)

    fetches = {"train_step": train_step, "domain_loss": domain_loss,
               "batch_domain_loss": batch_domain_loss}
    fetches.update(inv_outputs)

    total_time = previous_time = 0.0
    domain_loss_hist = collections.OrderedDict()
    std_loss_hist = collections.OrderedDict()

    curr_time_slice = 0
    domain_loss_hist[curr_time_slice] = np.array([])
    std_loss_hist[curr_time_slice] = np.array([])
    ## Generate data
    inv_inp_batch = next(inv_inp_gen)
    input_feed = {inv_inputs[k]: inv_inp_batch[k] for k in inv_inp_batch.keys()}
    i = 0
    while True:
        if max_iterations is not None and i > max_iterations:
            break
        elif max_time is not None and total_time > max_time:
            break
        i = i + 1

        ## Training Step
        elapsed, output = timed_run(sess, fetches=fetches, feed_dict=input_feed)
        total_time = elapsed + total_time

        inv_outputs_values = {k: output[k] for k in inv_outputs.keys()}
        std_loss = check_loss(inv_outputs_values, inv_inp_batch)

        if std_loss["loss"] < 0.3:
            # Generate new elements of y and reinitialize x
            inv_inp_batch = next(inv_inp_gen)
            input_feed = {inv_inputs[k]: inv_inp_batch[k] for k in inv_inp_batch.keys()}
            domain_loss_hist[curr_time_slice] = np.concatenate([domain_loss_hist[curr_time_slice], output['batch_domain_loss']])
            std_loss_hist[curr_time_slice] = np.concatenate([std_loss_hist[curr_time_slice], std_loss['batch_loss']])

        if total_time - previous_time > time_grain:
            previous_time = total_time
            domain_loss_hist[curr_time_slice] = np.concatenate([domain_loss_hist[curr_time_slice], output['batch_domain_loss']])
            std_loss_hist[curr_time_slice] = np.concatenate([std_loss_hist[curr_time_slice], std_loss['batch_loss']])
            curr_time_slice += 1
            domain_loss_hist[curr_time_slice] = []
            std_loss_hist[curr_time_slice] = []
        print("domain", output["domain_loss"], "std:", std_loss["loss"])

    return domain_loss_hist, std_loss_hist, total_time

def update_batch_loss_mins(batch_loss_mins, batch_loss):
    if batch_loss_mins is None:
        return batch_loss
    else:
        return np.minimum(batch_loss, batch_loss_mins)

def selective_update(new_values, old_values, do_batch_refresh):
    return np.where(do_batch_refresh, new_values, old_values)

def infty_like(a):
    """
    Return an array of infinities with shape of a
    """
    b = np.empty(a.shape)
    b[:] = np.infty
    return b

def expand_dims_like(a,b):
    """
    Expand the dimensions of a with ones to be like b
    """
    assert b.ndim > a.ndim
    extra_dim_ones = [1 for i in range(b.ndim-1)]
    return a.reshape([-1]+extra_dim_ones)

def min_fx_y(loss_op, batch_loss_op, inv_inputs, inv_inp_gen, sess, max_iterations=None,
             max_time=10.0, time_grain=1.0):
    """
    Solve inverse problem by search over inputs
    Given a function f, and y_batch, find x_batch s.t. f(x_batch) = y
    """
    assert (max_iterations is None) != (max_time is None), "set stop time xor max-iterations"

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_op)
    init = tf.initialize_all_variables()
    sess.run(init)

    fetches = {"train_step": train_step, "loss": loss_op,
               "batch_loss": batch_loss_op}

    total_time = previous_time = 0.0
    std_loss_hist = collections.OrderedDict()

    curr_time_slice = 0
    std_loss_hist[curr_time_slice] = np.array([])
    ## Generate data
    inv_inp_batch = next(inv_inp_gen)
    input_feed = {inv_inputs[k]: inv_inp_batch[k] for k in inv_inp_batch.keys()}

    batch_loss_mins = None # ew
    i = 0
    while True:
        if max_iterations is not None and i > max_iterations:
            break
        elif max_time is not None and total_time > max_time:
            break
        i = i + 1

        ## Training Step
        elapsed, output = timed_run(sess, fetches=fetches, feed_dict=input_feed)
        total_time = elapsed + total_time
        batch_loss = output['batch_loss']

        # Is the (elemement wise) loss zero?
        eps = 1e-5
        batch_loss_zero = batch_loss < eps

        new_batch_loss_mins = update_batch_loss_mins(batch_loss_mins, batch_loss)
        if batch_loss_mins is None:
            batch_loss_mins_change = infty_like(batch_loss)
        else:
            batch_loss_mins_change = batch_loss_mins - new_batch_loss_mins
            # relative_change = batch_loss_mins_change / batch_loss_mins
        min_min_loss = 0.001
        batch_loss_converged = batch_loss_mins_change < min_min_loss
        do_batch_refresh = np.logical_or(batch_loss_zero, batch_loss_converged)
        batch_loss_mins = new_batch_loss_mins

        # print(batch_loss_mins, batch_loss_mins_change, batch_loss_converged)
        if np.any(batch_loss_zero):
            # Generate new elements of y and reinitialize x
            new_inv_inp_batch = next(inv_inp_gen)
            input_feed = {inv_inputs[k]: np.where(expand_dims_like(do_batch_refresh, inv_inp_batch[k]),
                                                  new_inv_inp_batch[k],
                                                  input_feed[inv_inputs[k]])
                          for k in inv_inp_batch.keys()}
            # update minimums
            batch_loss_mins = np.where(do_batch_refresh,
                                       infty_like(batch_loss_mins),
                                       batch_loss_mins)
            # FOllowing line is wrong because it lumps everything on
            good_indices = batch_loss_zero.nonzero()[0]
            a = std_loss_hist[curr_time_slice]
            b = batch_loss.take(good_indices)
            std_loss_hist[curr_time_slice] = np.concatenate([a,b])

        if total_time - previous_time > time_grain:
            previous_time = total_time
            std_loss_hist[curr_time_slice] = np.concatenate([std_loss_hist[curr_time_slice], output['batch_loss']])
            curr_time_slice += 1
            std_loss_hist[curr_time_slice] = []
        print("std:", output["loss"])

    return std_loss_hist, total_time
