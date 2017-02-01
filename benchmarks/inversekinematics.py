import sys
import getopt
import tensorflow as tf
import numpy as np
from reverseflow.util.tf import *
from arrows.util.viz import show_tensorboard_graph
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow

c = tf.cos
s = tf.sin


def ik_fwd_f(inputs):
    phi1 = inputs[0]
    phi2 = inputs[1]
    phi4 = inputs[2]
    phi5 = inputs[3]
    phi6 = inputs[4]
    d2 = inputs[5]
    d3 = inputs[6]
    h1 = inputs[7]
    r11 = -s(phi6)*(c(phi4)*s(phi1) + c(phi1)*c(phi2)*s(phi4) ) - c(phi6)*(c(phi5)*(s(phi1)*s(phi4) - c(phi1)*c(phi2)*c(phi4) ) + c(phi1)*s(phi2)*s(phi5) )
    r12 = s(phi6)*(c(phi5)*(s(phi1)*s(phi4) - c(phi1)*c(phi2)*c(phi4) ) + c(phi1)*s(phi2)*s(phi5) ) - c(phi6)*(c(phi4)*s(phi1) + c(phi1)*c(phi2)*s(phi4) )
    r13 = s(phi5)*(s(phi1)*s(phi4) - c(phi1)*c(phi2)*c(phi4) ) - c(phi1)*c(phi5)*s(phi2)
    r21 = s(phi6)*(c(phi1)*c(phi4) - c(phi2)*s(phi1)*s(phi4) ) + c(phi6)*(c(phi5)*(c(phi1)*s(phi4) + c(phi2)*c(phi4)*s(phi1) ) - s(phi1)*s(phi2)*s(phi5) )
    r22 = c(phi6)*(c(phi1)*c(phi4) - c(phi2)*s(phi1)*s(phi4) ) - s(phi6)*(c(phi5)*(c(phi1)*s(phi4) + c(phi2)*c(phi4)*s(phi1) ) - s(phi1)*s(phi2)*s(phi5) )
    r23 = -s(phi5)*(c(phi1)*s(phi4) + c(phi2)*c(phi4)*s(phi1) ) - c(phi5)*s(phi1)*s(phi2)
    r31 = c(phi6)*(c(phi2)*s(phi5) + c(phi4)*c(phi5)*s(phi2) ) - s(phi2)*s(phi4)*s(phi6)
    r32 = -s(phi6)*(c(phi2)*s(phi5) + c(phi4)*c(phi5)*s(phi2) ) - c(phi6)*s(phi2)*s(phi4)
    r33 = c(phi2)*c(phi5) - c(phi4)*s(phi2)*s(phi5)
    px = d2*s(phi1) - d3*c(phi1)*s(phi2)
    py = -d2*c(phi1) - d3*s(phi1)*s(phi2)
    pz = h1 + d3*c(phi2)
    outputs = [px, py, pz]
    outputs.extend([r11,
                    r12,
                    r13,
                    r21,
                    r22,
                    r23,
                    r31,
                    r32,
                    r33])
    return outputs


def ik_gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        inputs = [0]*8
        inputs[0] = tf.placeholder(tf.float32, name="phi1", shape=())
        inputs[1] = tf.placeholder(tf.float32, name="phi2", shape=())
        # inputs['phi3'] = placeholder(tf.float32, name="phi3", shape=(batch_size, 1))
        inputs[2] = tf.placeholder(tf.float32, name="phi4", shape=())
        inputs[3] = tf.placeholder(tf.float32, name="phi5", shape=())
        inputs[4] = tf.placeholder(tf.float32, name="phi6", shape=())

        inputs[5] = tf.placeholder(tf.float32, name="d2", shape=())
        inputs[6] = tf.placeholder(tf.float32, name="d3", shape=())
        inputs[7] = tf.placeholder(tf.float32, name="h1", shape=())

        outputs = ik_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}

def test_ik():
    with tf.name_scope("ik_stanford_manipulator"):
        in_out = ik_gen_graph(tf.Graph(), 1, is_placeholder)
        input_values = [30, 45, 90, 0, 60, 1, 2, 1]
        with tf.Session() as sess:
            output_values = sess.run(in_out["outputs"], feed_dict={in_out["inputs"][i]: input_values[i] for i in range(len(input_values))})
        print(output_values)
    arrow = graph_to_arrow(in_out["outputs"], name="ik_stanford")
    show_tensorboard_graph()
    tf.reset_default_graph()
    inverse = invert(arrow)
    min_approx_error_arrow(inverse, output_values)

test_ik()
