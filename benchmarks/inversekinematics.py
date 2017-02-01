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
    phi1 = inputs['phi1']
    phi2 = inputs['phi2']
    phi4 = inputs['phi4']
    phi5 = inputs['phi5']
    phi6 = inputs['phi6']
    d2 = inputs['d2']
    d3 = inputs['d3']
    h1 = inputs['h1']
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
    outputs = {'px': px, 'py': py,'pz': pz}
    outputs.update({'r11': r11,
                    'r12': r12,
                    'r13': r13,
                    'r21': r21,
                    'r22': r22,
                    'r23': r23,
                    'r31': r31,
                    'r32': r32,
                    'r33': r33})
    return outputs


def ik_gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        inputs = {}
        inputs['phi1'] = tf.placeholder(tf.float32, name="phi1", shape=(batch_size, 1))
        inputs['phi2'] = tf.placeholder(tf.float32, name="phi2", shape=(batch_size, 1))
        # inputs['phi3'] = placeholder(tf.float32, name="phi3", shape=(batch_size, 1))
        inputs['phi4'] = tf.placeholder(tf.float32, name="phi4", shape=(batch_size, 1))
        inputs['phi5'] = tf.placeholder(tf.float32, name="phi5", shape=(batch_size, 1))
        inputs['phi6'] = tf.placeholder(tf.float32, name="phi6", shape=(batch_size, 1))

        inputs['d2'] = tf.placeholder(tf.float32, name="d2", shape=(batch_size, 1))
        inputs['d3'] = tf.placeholder(tf.float32, name="d3", shape=(batch_size, 1))
        inputs['h1'] = tf.placeholder(tf.float32, name="h1", shape=(batch_size, 1))

        outputs = ik_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}

def test_ik():
    with tf.name_scope("ik_stanford_manipulator"):
        in_out = ik_gen_graph(tf.Graph(), 1, is_placeholder)
        outputs = list(in_out["outputs"].values())
    arrow = graph_to_arrow(outputs, name="ik_stanford")
    show_tensorboard_graph()
    tf.reset_default_graph()
    inverse = invert(arrow)
    output_values = [np.random.randn() for _ in range(len(outputs))]
    min_approx_error_arrow(inverse, output_values)

test_ik()
