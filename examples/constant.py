import pi
from pi import invert
from pi import analysis
import tensorflow as tf
from tensorflow import float32
import numpy as np
from pi.optim import min_param_error, min_fx_y, gen_y, gen_loss_model, nnet
from pi.optim import enhanced_pi
from pi.util import *
import pi.templates.res_net


def tensor_rand(tensors):
    return {t: np.random.rand(*t.get_shape().as_list()) for t in tensors}


def fwd_f(inputs):
    x, y = inputs['x'], inputs['y']
    # a = ((x * 2)*x - (4 * y)) + 5 + x
    # b = (a + 2*a)+4
    # z = a + b
    z = x * y + x
    outputs = {"z": z}
    return outputs


def gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        x = ph_or_var(float32, name="x", shape=(batch_size, 1),
                      is_placeholder=is_placeholder)
        y = ph_or_var(float32, name="y", shape=(batch_size, 1),
                      is_placeholder=is_placeholder)
        inputs = {"x": x, "y": y}
        outputs = fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}

n_iters = 1000
batch_size = 128
max_time = 50

# Default graph and session
g = tf.get_default_graph()
sess = tf.Session(graph=g)

data_generator = False
nnet_approx = False
nnet_enhanced_pi = True
pointwise_pi = False

## create data generator


def infinite_input(batch_size):
    generator_graph = tf.Graph()
    with generator_graph.as_default() as g:
        in_out_var = gen_graph(g, batch_size, False)

    while True:
        with generator_graph.as_default() as g:
            output = gen_y(in_out_var["outputs"])
        yield output


def infinite_samples(sampler, shape):
    while True:
        yield sampler(*shape)

def dictionary_gen(x):
    while True:
        yield {k: next(v) for k, v in x.items()}

inv_inp_gen = infinite_input(batch_size)

if data_generator:
    ## Generate graph for generating output data
    in_out_var = gen_graph(g, batch_size, False)
    print(in_out_var)
    y_batch = gen_y(in_out_var["outputs"])


if nnet_approx:
    ## Neural Network Construction
    nnet(fwd_f, in_out_var['inputs'], in_out_var['outputs'],
         pi.templates.res_net.res_net_template_dict, y_batch, sess)

if nnet_enhanced_pi:
    ## Inverse Graph
    with g.name_scope('nnet_enhanced_pi'):
        in_out_ph = gen_graph(g, batch_size, True)
        x, y = in_out_ph['inputs']['x'], in_out_ph['inputs']['y']
        z = in_out_ph['outputs']['z']
        min_param_size = 5
        shrunk_params = {'theta': tf.placeholder(dtype=tf.float32, shape=(batch_size, min_param_size), name="shrunk_param")}
        inv_g, inv_inputs, inv_outputs_map = pi.invert.invert((z,),
                                                              shrunk_params=shrunk_params)
        inv_inp_map = dict(zip(['z'], inv_inputs))

        generators = {k: infinite_samples(np.random.rand, v.get_shape().as_list()) for k,v in shrunk_params.items()}
        shrunk_param_gen = dictionary_gen(generators)
        enhanced_pi(inv_g, inv_inp_map, inv_inp_gen, shrunk_params, shrunk_param_gen,
                    inv_outputs_map, sess, max_iterations=10000)
        writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)

assert False

# loss_op, absdiffs, mean_loss_per_batch_op, mean_loss_per_batch_op, target_outputs = gen_loss_model(in_out_var, sess)
# loss_data, loss_hist, total_time = min_fx_y(loss_op, mean_loss_per_batch_op, in_out_var, target_outputs, y_batch, sess,
#                                 max_time=max_time)

if pointwise_pi:
    ## Inverse Graph
    in_out_ph = gen_graph(g, batch_size, True)
    x, y = in_out_ph['inputs']['x'], in_out_ph['inputs']['y']
    z = in_out_ph['outputs']['z']
    inv_g, inv_inputs, inv_outputs_map = pi.invert.invert((z,))

    inv_outputs_map_canonical = {k:inv_outputs_map[v.name] for k,v in in_out_ph['inputs'].items()}
    inv_inp_map = dict(zip(['z'], inv_inputs))
    std_loss_hist, domain_loss_hist, total_time_p    = min_param_error(loss_op, mean_loss_per_batch_op, inv_g, inv_inp_map,
                                                    inv_outputs_map_canonical,
                                                    y_batch, in_out_var['inputs'],
                                                    target_outputs,
                                                    sess,
                                                    max_time=max_time)
    # writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)
