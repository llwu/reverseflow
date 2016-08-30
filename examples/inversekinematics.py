from pi import invert
import pi
import tensorflow as tf
from tensorflow import float32

## 2d linkage bot
tf.reset_default_graph()
g = tf.get_default_graph()

l1 = tf.constant(1.0)
l2 = tf.constant(1.0)
phi1 = tf.placeholder(float32, name="phi1")
phi2 = tf.placeholder(float32, name="phi2")

x = l1 * tf.cos(phi1) + l2 * tf.cos(phi1+phi2)
y = l1 * tf.sin(phi1) + l2 * tf.sin(phi1+phi2)

g = tf.get_default_graph()
inv_g = pi.invert.invert(g, (x, y))
