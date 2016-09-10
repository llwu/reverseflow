from pi.compare import compare
import sys
import getopt
import tensorflow as tf
import numpy as np
from pi.util import *

## Constant
def constant_fwd_f(inputs):
    x, y = inputs['x'], inputs['y']
    z = x * y + x
    outputs = {"z": z}
    return outputs

def constant_gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        x = ph_or_var(tf.float32, name="x", shape=(batch_size, 1),
                      is_placeholder=is_placeholder)
        y = ph_or_var(tf.float32, name="y", shape=(batch_size, 1),
                      is_placeholder=is_placeholder)
        inputs = {"x": x, "y": y}
        outputs = constant_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}


def main(argv):
    options = {'batch_size': 512, 'max_time': 30.0,
               'logdir': '/home/zenna/repos/inverse/log',
               'nnet_enhanced_pi':True}
    g = tf.get_default_graph()
    gen_graph = constant_gen_graph
    fwd_f = constant_fwd_f
    min_param_size = 1
    param_types = {'theta': tensor_type(dtype=tf.float32,
                   shape=(options['batch_size'], min_param_size),
                   name="shrunk_param")}

    param_gen = {k: infinite_samples(np.random.rand, v['shape'])
                  for k, v in param_types.items()}
    shrunk_param_gen = dictionary_gen(param_gen)
    compare(g, gen_graph, constant_fwd_f, param_types, shrunk_param_gen, options)

if __name__ == "__main__":
    main(sys.argv)
