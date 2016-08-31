import pi
from pi import invert
import tensorflow as tf
from tensorflow import float32

## f(x,y) = x * y + x
tf.reset_default_graph()
g = tf.get_default_graph()

x = tf.placeholder(float32, name="x")
y = tf.placeholder(float32, name="y")

z = x * y + x
g = tf.get_default_graph()
inv_g, inputs, out_map, params = pi.invert.invert(g, (z,))
print(out_map)

writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)
sess = tf.Session(graph=inv_g)
feed_dict = {inputs[0] : 10.0, params[0]: 1.0, params[1]: 1.0}
output = sess.run(feed_dict=feed_dict, fetches=out_map)
print(output)
