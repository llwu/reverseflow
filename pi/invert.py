import tensorflow as tf
from tensorflow import float32
from pqdict import pqdict
import numpy as np

from pi import inverses

def doshit(g, op, inputs, inverses):
    """g :: tf.Graph - graph to add to
       op :: tf.Op - op to invert
       inputs :: [tf.Tensor] - inputs to inv_op
       inverses :: {tf.}
     """
     inv_op = inverses[op.type]
     inv_op.go(g, inputs)


def invert(g, out_tensors):
    """g :: tf.Graph the graph to invert"""
    inv_g = tf.Graph()
    op_nouts = {}
    # Map between tensors from g to inv_g
    tensor_map = {}

    for op in g.get_operations():
        outs = op.outputs
        nouts = len(outs)
        op_nouts[op] = nouts
    ops = pqdict(op_nouts)

    for out_tensor in out_tensors:
        op = out_tensor.op
        ops[op] = ops[op] - 1

        with inv_g.as_default():
            inv_inp_tensor = tf.placeholder(dtype=out_tensor.dtype)
            tensor_map[out_tensor] = inv_inp_tensor

    while True:
        op, priority = ops.popitem()
        print(priority, op)
        assert priority == 0, "Tried to invert op before inverting its dependents"

        # Inputs to the inverse function are outputs from forward function
        inputs = [tensor_map[out] for out in op.outputs]
        print("inputs", inputs)

        inputs(inv_g, inputs)

    # Now what's next?


    ## foreach output output tensor, set the edge›
    # for out in out_tensors:

    # For each output of g (out_tensors), create an input (placehodler) in inv_g
    # with g.as_default():
    #     for out in out_tensors:
    #         tf.placeholder(dtype=out.dtype)
    #
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
g = z.graph
inv_g = invert(g, (z,))

# sess = tf.Session()
# # Create a summary writer, add the 'graph' to the event file.
# writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', sess.graph)
#
#
# z = tf.cos(y)
# f = z + x
#
# # Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data * 0.1 + 0.3
#
# # Try to find values for W and b that compute y_data = W * x_data + b
# # (We know that W should be 0.1 and b 0.3, but TensorFlow will
# # figure that out for us.)
# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
# y = W * x_data + b
#
# x = tf.placeholder(float32)
# y = tf.abs(x)
#
# 'Abs' :
#
# def invert(graph):
#     ddd
#
# y =
