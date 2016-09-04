import pi
from pi import invert
import tensorflow as tf
from tensorflow import float32
import numpy as np
from pi.optim import minimize_error, evaluate, gen_y
from pi.util import *

def tensor_rand(tensors):
    return {t:np.random.rand(*t.get_shape().as_list()) for t in tensors}

def gen_graph(g, is_placeholder):
    with g.name_scope("fwd_g"):
        x = ph_or_var(float32, name="x", shape = (), is_placeholder=is_placeholder)
        y = ph_or_var(float32, name="y", shape = (), is_placeholder=is_placeholder)
        z = ((x * 2) - (4 * y)) + 5 + x

    return {"inputs":(x,y), "outputs":(z,)}


g = tf.get_default_graph()
in_out = gen_graph(g, False)
y_batch = gen_y(g, in_out["outputs"])
sess = tf.Session(graph=g)
evaluate(y_batch, in_out["outputs"], sess)
# inv_g, inputs, out_map = pi.invert.invert((z,))
# params = inv_g.get_collection("params")
# errors = inv_g.get_collection("errors")
#
# writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)
#
#
# sess = tf.Session(graph=inv_g)
#
# input_feed = tensor_rand(inputs)
# minimize_error(inv_g, input_feed, sess)
# output = sess.run(feed_dict=input_feed, fetches=out_map)
#
# yy = output['fwd_g/y']
# xx = output['fwd_g/x']
# ((xx * 2) - (4 * yy)) + 5 + xx
