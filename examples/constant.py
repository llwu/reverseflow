import pi
from pi import invert
import tensorflow as tf
from tensorflow import float32
import numpy as np
from pi.optim import minimize_error

def tensor_rand(tensors):
    return {t:np.random.rand(*t.get_shape().as_list()) for t in tensors}

g = tf.get_default_graph()
with g.name_scope("fwd_g"):
    x = tf.placeholder(float32, name="x", shape = ())
    y = tf.placeholder(float32, name="y", shape = ())
    z = ((x * 2) - (4 * y)) + 5 + x

inv_g, inputs, out_map = pi.invert.invert((z,))
params = inv_g.get_collection("params")
errors = inv_g.get_collection("errors")

writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)
sess = tf.Session(graph=inv_g)

input_feed = tensor_rand(inputs)
minimize_error(inv_g, input_feed, sess)
output = sess.run(feed_dict=input_feed, fetches=out_map)

yy = output['fwd_g/y']
xx = output['fwd_g/x']
((xx * 2) - (4 * yy)) + 5 + xx
