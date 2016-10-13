from bf.compare import compare
import sys
import getopt
import tensorflow as tf
import numpy as np
<<<<<<< HEAD
from bf.util import *
from bf.templates.res_net import res_net_template_dict
=======
from pi.util import *
from tensortemplates.res_net import template_dict
>>>>>>> 1f0dffa638da4f42bf7f50b4dafe542d1e51f226

## Constant
def constant_fwd_f(inputs):
    x, y = inputs['x'], inputs['y']
    a = x + 2 * x + 4 * y
    b = a + a
    z = b * 10*x
    outputs = {"z": z}
    return outputs

def constant_gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        x = ph_or_var(tf.float32, name="x", shape=(batch_size, 128),
                      is_placeholder=is_placeholder)
        y = ph_or_var(tf.float32, name="y", shape=(batch_size, 128),
                      is_placeholder=is_placeholder)
        inputs = {"x": x, "y": y}
        outputs = constant_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}


def main(argv):
    options = {'batch_size': 512, 'max_time': 200.0,
               'logdir': '/home/zenna/repos/inverse/log',
               'template': template_dict,
               'nnet_enhanced_pi': False,
               'pointwise_pi': False,
               'min_fx_y': False,
               'nnet': True,
               'min_fx_param': False,
               'rightinv_pi_fx': False,
               'nruns': 2}
    gen_graph = constant_gen_graph
    fwd_f = constant_fwd_f
    min_param_size = 1
    param_types = {'theta': tensor_type(dtype=tf.float32,
                   shape=(options['batch_size'], min_param_size),
                   name="shrunk_param")}

    param_gen = {k: infinite_samples(np.random.rand, v['shape'])
                  for k, v in param_types.items()}
    shrunk_param_gen = dictionary_gen(param_gen)
    return compare(gen_graph, constant_fwd_f, param_types, shrunk_param_gen, options)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    global runs
    runs = main(sys.argv)
    import pi
    bf.analysis.plot(runs, 30.0)
