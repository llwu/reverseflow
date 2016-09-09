## Parametric Inversion Test
import pi as pi
import tensorflow as tf
from tensorflow import float32

## f(x) = sin(xos(x))
x = tf.placeholder('float32')
a = tf.cos(x)
b = tf.sin(a)
c = tf.square(b)

## Invert a graph g
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g

def func(x, y):
    return 2*x + ys


import tensorflow as tf
import numpy as np

# Either, create new OP type for parametric inverses.
# OR just add them to the graph.

from queues import PriorityQueue

"g :: tf.Graph the graph to invert"
def invert(g, out_tensors):
    inv_g = tf.Graph()

    ops = PriorityQueue()

    for op in g.get_operations():
        outs = op.outputs
        nouts = len(outs)

        # if any of the outputs of this op is an output of the whole function
        for out in out_tensors:
            for local_out in outs:
                if local_out == out:
                    nouts = nouts - 1

        ops.put(nouts, op)

    while True:
        priority, op = ops.pop()
        assert(priority == 0, "Tried to invert op before inverting its dependents")

    ## foreach output output tensor, set the edge›
    for out in out_tensors:


    # For each output of g (out_tensors), create an input (placehodler) in inv_g
    with g.as_default():
        for out in out_tensors:
            tf.placeholder(dtype=out.dtype)

    return inv_g

    # inv_g = tf.Graph()
    # # For every operation in original graph, add it's inverse in new graph
    # for op in g.get_operations():
    #     inv_g.create_op(op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_shapes=True, compute_device=True

tf.reset_default_graph()
sess = tf.Session()
x = tf.placeholder(float32, name='x')
y = tf.placeholder(float32, name='y')
z = x*y + x
zxf = z.graph
inv_g = invert(f, (z,))

## Alapplyrithm
# Construct flag-map, P
# Add input placeholders for each output of g
#

# ex 1 f(x) = cos(sin(x)) + x
# x = tf.placeholder('float32')
# y = tf.sin(x)
# z = tf.cos(y)
# f = z + x
#
# inv_g = invert(f, (z,))
#
# ex f(x,y) = x * y + 1


sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', sess.graph)


z = tf.cos(y)
f = z + x

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
