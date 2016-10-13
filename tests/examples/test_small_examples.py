from reverseflow.invert import invert
import tensorflow as tf
from tensorflow import float32

# f(x,y) = x * y + x
tf.reset_default_graph()
g = tf.get_default_graph()

with g.name_scope("fwd_g"):
    x = tf.placeholder(float32, name="x", shape=())
    y = tf.placeholder(float32, name="y", shape=())

z = x * y + x
g = tf.get_default_graph()
inv_g, inputs, out_map = invert(({'z': z}))

tf.reset_default_graph()
g = tf.get_default_graph()

with g.name_scope("fwd_g"):
    x = tf.placeholder(float32, name="x", shape=())
    y = tf.placeholder(float32, name="y", shape=())
    z = tf.placeholder(float32, name="z", shape=())

    e = x*y
    f = x-y
    o1 = (e + 2*f)+(3*z)
    o2 = e + f

inv_g, inputs, out_map, = invert({'o1': o1, "o2": o2})
# print(out_map)
#
# writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)
# sess = tf.Session(graph=inv_g)
# feed_dict = {inputs[0] : 10.0, params[0]: 1.0}
# output = sess.run(feed_dict=feed_dict, fetches=out_map)
