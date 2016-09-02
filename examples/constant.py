import pi
from pi import invert
import tensorflow as tf
from tensorflow import float32

g = tf.get_default_graph()
with g.name_scope("fwd_g"):
    tf.reset_default_graph()
    g = tf.get_default_graph()

    x = tf.placeholder(float32, name="x", shape = ())
    y = tf.placeholder(float32, name="y", shape = ())

    z = ((x * 2) - (4 * y)) + 5 + x

inv_g, inputs, out_map = pi.invert.invert((z,))
params = inv_g.get_collection("params")
errors = inv_g.get_collection("errors")

writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)
sess = tf.Session(graph=inv_g)

# feed_dict = {inputs[0] : 10.0, params[0]: 1.0}
# output = sess.run(feed_dict=feed_dict, fetches=out_map)
