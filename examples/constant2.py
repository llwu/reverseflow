from pi.compare import compare
import sys
import getopt
import tensorflow as tf
import numpy as np
from pi.util import *
from pi.templates.res_net import res_net_template_dict

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
    options = {'batch_size': 512, 'max_time': 600.0,
               'logdir': '/home/zenna/repos/inverse/log',
               'template': res_net_template_dict,
               'nnet_enhanced_pi': True,
               'pointwise_pi': True,
               'min_fx_y': True,
               'nnet': True}
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
    global x
    x = main(sys.argv)

std_loss_hists, domain_loss_hists, total_times = x

import matplotlib.pyplot as plt
import pi
for k, v in std_loss_hists.items():
    print(k)
    pi.analysis.profile2d(v, total_times[k], max_error=1.0)
    plt.title('std_loss %s' % k)
    plt.figure()

for k, v in domain_loss_hists.items():
    print(k)
    pi.analysis.profile2d(v, total_times[k])
    plt.title('domain_loss %s' % k)
    plt.figure()
