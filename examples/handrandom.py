from pi.compare import compare
import sys
import getopt
import tensorflow as tf
import numpy as np
from pi.util import *
from tensortemplates.res_net import template_dict



## Constant
def constant_fwd_f(inputs, seed=0):
    print("Seed is", seed)
    np.random.seed(seed)
    x, y = inputs['x'], inputs['y']
    ops = [tf.add, tf.mul, tf.sub]
    inp_tensors = [x, y]
    tensors = []
    nsteps = 10
    for i in range(nsteps):
        op = np.random.choice(ops)
        a, b = np.random.choice(inp_tensors + tensors, (2,))
        output = op(a, b)
        tensors.append(output)

    output_tensors = list(filter(lambda t:len(t.consumers()) == 0, tensors))
    outputs = {"output_%s" % i: v for i, v in enumerate(output_tensors)}
    writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', x.graph)
    print(summary(x.graph))
    # assert False
    return outputs

def constant_gen_graph(g, batch_size, is_placeholder, seed=0):
    with g.name_scope("fwd_g"):
        x = ph_or_var(tf.float32, name="x", shape=(batch_size, 128),
                      is_placeholder=is_placeholder)
        y = ph_or_var(tf.float32, name="y", shape=(batch_size, 128),
                      is_placeholder=is_placeholder)
        inputs = {"x": x, "y": y}
        outputs = constant_fwd_f(inputs, seed=seed)
        return {"inputs": inputs, "outputs": outputs}


def main(argv):
    options = {'batch_size': 512, 'max_time': 10.0,
               'logdir': '/home/zenna/repos/inverse/log',
               'template': template_dict,
               'nnet_enhanced_pi': False,
               'pointwise_pi': False,
               'min_fx_y': False,
               'min_fx_param': False,
               'nnet': True,
               'rightinv_pi_fx': True,
               'rightinv_pi_domain': True,
               'nruns': 50}
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
    pi.analysis.plot(runs, 30.0)
