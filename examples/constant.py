import pi
from pi import invert
from pi import analysis
import tensorflow as tf
from tensorflow import float32
import numpy as np
from pi.optim import min_param_error, min_fx_y, gen_y, gen_loss_model
from pi.util import *

def tensor_rand(tensors):
    return {t:np.random.rand(*t.get_shape().as_list()) for t in tensors}

def gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        x = ph_or_var(float32, name="x", shape = (batch_size,1), is_placeholder=is_placeholder)
        y = ph_or_var(float32, name="y", shape = (batch_size,1), is_placeholder=is_placeholder)
        a = ((x * 2)*x - (4 * y)) + 5 + x
        b = (a + 2*a)+4
        z = a + b
        inputs = {"x":x, "y":y}
        outputs = {"z":z}

    return {"inputs":inputs, "outputs":outputs}

n_iters = 1000
batch_size = 128
max_time = 50

# Default graph and session
g = tf.get_default_graph()
sess = tf.Session(graph=g)

in_out_var = gen_graph(g, batch_size, False)
y_batch = gen_y(in_out_var["outputs"])

loss_op, absdiffs, mean_loss_per_batch_op, mean_loss_per_batch_op, target_outputs = gen_loss_model(in_out_var, sess)
loss_data, loss_hist, total_time = min_fx_y(loss_op, mean_loss_per_batch_op, in_out_var, target_outputs, y_batch, sess,
                                max_time=max_time)

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
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt

plt.plot(np.arange(len(loss_data)), loss_data)
# plt.plot(np.arange(n_iters), std_loss_data, 'bs', np.arange(n_iters), node_loss_data, 'g^')

# for k, v in loss_hist.items():

plt.show()
