from pi.compare import compare
import sys
import getopt
import tensorflow as tf
import numpy as np
from pi.util import *
from pi.templates.res_net import res_net_template_dict
from pi.generator import gen_graph, maybe_stop, apply_elem_op
import random


## Issues
## How to do rand_fwd_f
## Handle bath sizee
## Handle disconnected graphs

def name_tensors(tensors, prefix):
    return {"%s_%s" % (prefix, i): t for i, t in enumerate(tensors)}

def rand_fwd_f(inputs):
    random.seed(0)
    np.random.seed(0)
    x, y = inputs['x'], inputs['y']
    g = x.graph
    gen_graph(g, [maybe_stop, apply_elem_op])
    outputs = list(filter(is_output, all_tensors_namescope(g, 'fwd_g')))
    named_outputs = name_tensors(outputs, prefix="output")
    return named_outputs


def rand_gen_graph(g, batch_size, is_placeholder):
    np.random.seed(0)
    with g.name_scope("fwd_g"):
        x = ph_or_var(tf.float32, name="x", shape=(batch_size, 1),
                      is_placeholder=is_placeholder)
        y = ph_or_var(tf.float32, name="y", shape=(batch_size, 1),
                      is_placeholder=is_placeholder)
        inputs = {"x": x, "y": y}
        outputs = rand_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}

def main(argv):
    global stats
    options = {'batch_size': 512, 'max_time': 5.0,
               'logdir': '/home/zenna/repos/inverse/log',
               'template': res_net_template_dict,
               'nnet_enhanced_pi': True,
               'pointwise_pi': True,
               'min_fx_y': True,
               'nnet': True}
    gen_graph = rand_gen_graph
    fwd_f = rand_fwd_f
    min_param_size = 1
    param_types = {'theta': tensor_type(dtype=tf.float32,
                   shape=(options['batch_size'], min_param_size),
                   name="shrunk_param")}

    param_gen = {k: infinite_samples(np.random.rand, v['shape'])
                  for k, v in param_types.items()}
    np.random.seed(0)
    shrunk_param_gen = dictionary_gen(param_gen)
    np.random.seed(0)
    stats = compare(gen_graph, rand_fwd_f, param_types, shrunk_param_gen,
                    options)
    return stats

if __name__ == "__main__":
    global x
    x = main(sys.argv)
