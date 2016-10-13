from rf.compare import compare
import sys
import getopt
import tensorflow as tf
import numpy as np
from rf.util import *
from rf.templates.res_net import res_net_template_dict

## nao: https://www.cs.umd.edu/~nkofinas/Projects/KofinasThesis.pdf
## Bio: file:///home/zenna/Downloads/65149.pdf
## Standard Manipulator: http://cdn.intechweb.org/pdfs/379.pdf
## https://upcommons.upc.edu/bitstream/handle/2099.1/24573/A-Denavit%20Hartenberg.pdf?sequence=2&isAllowed=y
## Ref: http://s3.amazonaws.com/academia.edu.documents/30756918/10.1.1.60.8175.pdf?AWSAccessKeyId=AKIAJ56TQJRTWSMTNPEA&Expires=1472865779&Signature=1o70EkdUm484Apxh69vX%2F6m3BZQ%3D&response-content-disposition=inline%3B%20filename%3DLearning_inverse_kinematics.pdf

## Constant
c = tf.cos
s = tf.sin
def ik_fwd_f(inputs):
    phi1, phi2, phi4, phi5, phi6 = inputs['phi1'], inputs['phi2'], inputs['phi4'], inputs['phi5'], inputs['phi6']
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
    outputs = {'px':px, 'py':py,'pz':pz}
    outputs.update({'r11':r11,
                    'r12':r12,
                    'r13':r13,
                    'r21':r21,
                    'r22':r22,
                    'r23':r23,
                    'r31':r31,
                    'r32':r32,
                    'r33':r33})
    return outputs

def ik_gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        inputs = {}
        inputs['phi1'] = ph_or_var(tf.float32, name="phi1", shape = (batch_size, 1), is_placeholder=is_placeholder)
        inputs['phi2'] = ph_or_var(tf.float32, name="phi2", shape = (batch_size, 1), is_placeholder=is_placeholder)
        # inputs['phi3'] = ph_or_var(tf.float32, name="phi3", shape = (batch_size, 1), is_placeholder=is_placeholder)
        inputs['phi4'] = ph_or_var(tf.float32, name="phi4", shape = (batch_size, 1), is_placeholder=is_placeholder)
        inputs['phi5'] = ph_or_var(tf.float32, name="phi5", shape = (batch_size, 1), is_placeholder=is_placeholder)
        inputs['phi6'] = ph_or_var(tf.float32, name="phi6", shape = (batch_size, 1), is_placeholder=is_placeholder)

        inputs['d2'] = ph_or_var(tf.float32, name="d2", shape = (batch_size, 1), is_placeholder=is_placeholder)
        inputs['d3'] = ph_or_var(tf.float32, name="d3", shape = (batch_size, 1), is_placeholder=is_placeholder)
        inputs['h1'] = ph_or_var(tf.float32, name="h1", shape = (batch_size, 1), is_placeholder=is_placeholder)

        outputs = ik_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}


def main(argv):
    options = {'batch_size': 512, 'max_time': 100.0,
               'logdir': '/home/zenna/repos/inverse/log',
               'template': template_dict,
               'nnet_enhanced_pi': False,
               'pointwise_pi': False,
               'min_fx_y': False,
               'nnet': False,
               'min_fx_param': False,
               'rightinv_pi_fx': True,
               'nruns': 2}
    gen_graph = ik_gen_graph
    fwd_f = ik_fwd_f
    min_param_size = 1
    param_types = {'theta': tensor_type(dtype=tf.float32,
                   shape=(options['batch_size'], min_param_size),
                   name="shrunk_param")}

    param_gen = {k: infinite_samples(np.random.rand, v['shape'])
                  for k, v in param_types.items()}
    shrunk_param_gen = dictionary_gen(param_gen)
    return compare(gen_graph, ik_fwd_f, param_types, shrunk_param_gen, options)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    global runs
    runs = main(sys.argv)
    import pi
    rf.analysis.plot(runs, 30.0)


## This is the problem of computing the inverse kinematics of a robot
