import pi
from pi import invert
import tensorflow as tf
from tensorflow import float32
import numpy as np
from bf.optim import minimize_error


def tensor_rand(tensors):
    return {t:np.random.rand(*t.get_shape().as_list()) for t in tensors}

## This is the problem of computing the inverse kinematics of a robot

## nao: https://www.cs.umd.edu/~nkofinas/Projects/KofinasThesis.pdf
## Bio: file:///home/zenna/Downloads/65149.pdf
## Standard Manipulator: http://cdn.intechweb.org/pdfs/379.pdf
## https://upcommons.upc.edu/bitstream/handle/2099.1/24573/A-Denavit%20Hartenberg.pdf?sequence=2&isAllowed=y
## Ref: http://s3.amazonaws.com/academia.edu.documents/30756918/10.1.1.60.8175.pdf?AWSAccessKeyId=AKIAJ56TQJRTWSMTNPEA&Expires=1472865779&Signature=1o70EkdUm484Apxh69vX%2F6m3BZQ%3D&response-content-disposition=inline%3B%20filename%3DLearning_inverse_kinematics.pdf


sin = tf.sin
cos = tf.cos

g = tf.get_default_graph()
with g.name_scope("fwd_g"):
    phi1 = tf.placeholder(float32, name="phi1", shape = ())
    phi2 = tf.placeholder(float32, name="phi2", shape = ())
    phi3 = tf.placeholder(float32, name="phi3", shape = ())
    phi4 = tf.placeholder(float32, name="phi4", shape = ())
    phi5 = tf.placeholder(float32, name="phi5", shape = ())
    phi6 = tf.placeholder(float32, name="phi6", shape = ())
    r11 = -sin(phi6)*(cos(phi4)*sin(phi1)+cos(phi1)*cos(phi2)*sin(phi4)) - cos(phi6)*(cos(phi5)*(sin(phi1)*sin(phi4)-cos(phi1)*cos(phi2)*cos(phi4))+cos(phi1)*sin(phi2)*sin(phi5))

inverses = bf.defaults.default_inverses
inverses['Sin'] = bf.inv_ops.inv_math_ops.injsin
inverses['Cos'] = bf.inv_ops.inv_math_ops.injcos

print(inverses  )
inv_g, inputs, out_map = bf.invert.invert((r11,), inverses=inverses)
params = inv_g.get_collection("params")
errors = inv_g.get_collection("errors")

writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', inv_g)
sess = tf.Session(graph=inv_g)
# input_feed = tensor_rand(inputs)
# minimize_error(inv_g, input_feed, sess)
# output = sess.run(feed_dict=input_feed, fetches=out_map)
#
# yy = output['fwd_g/y']
# xx = output['fwd_g/x']
# ((xx * 2) - (4 * yy)) + 5 + xx
