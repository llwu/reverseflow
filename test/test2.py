import tensorflow as tf
from invert import invert
from inverses import default_inverses

 

tf.reset_default_graph()
sess = tf.Session()
x = tf.placeholder(float32, name='x')
y = tf.placeholder(float32, name='y')
z = x*y + x
g = z.graph
inv_g = invert(g, (z,))
