## Parametric Inversion Test
import pi as pi
import tensorflow as tf

## f(x) = sin(xos(x))
x = tf.placeholder('float32')
a = tf.cos(x)
b = tf.sin(a)
c = tf.square(b)

## Invert a graph g
def invert(g):
    ...

    
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g

def func(x, y):
    return 2*x + ys


import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

x = tf.placeholder(float32)
y = tf.abs(x)

'Abs' :

def invert(graph):
    ddd

y =
