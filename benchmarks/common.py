"""Functions common for examples"""
import sys
from arrows.util.io import *
from linkage_kinematics import robo_tensorflow
from stanford_kinematics import stanford_tensorflow
from arrows.util.misc import rand_string, getn
from metrics.generalization import test_everything
from reverseflow.train.common import layer_width
from reverseflow.train.reparam import *
from reverseflow.train.unparam import unparam
from reverseflow.train.loss import inv_fwd_loss_arrow, supervised_loss_arrow
from reverseflow.train.supervised import supervised_train
from reverseflow.train.callbacks import save_callback, save_options, save_every_n, save_everything_last

import tensortemplates.res_net as res_net
import tensortemplates.conv_res_net as conv_res_net
import numpy as np
import tensorflow as tf

def stringy_dict(d):
    out = ""
    for (key, val) in d.items():
        if val is not None and val is not '':
            out = out + "%s_%s__" % (str(key), str(val))
    return out

def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


template_module = {'res_net': res_net, 'conv_res_net': conv_res_net}

def boolify(x):
    if x in ['0', 0, False, 'False', 'false']:
        return False
    elif x in ['1', 1, True, 'True', 'true']:
        return True
    else:
        assert False, "couldn't convert to bool"


def default_kwargs():
    """Default kwargs"""
    options = {}
    options['learning_rate'] = (float, 0.1)
    options['update'] = (str, 'momentum')
    options['params_file'] = (str, 28)
    options['momentum'] = (float, 0.9)
    options['description'] = (str, "")
    options['batch_size'] = (int, 128)
    options['save_every'] = (int, 100)
    options['compress'] = (boolify, 0,)
    options['num_iterations'] = (int, 1000)
    options['save'] = (boolify, True)
    options['template'] = (str, 'res_net')
    options['train'] = (boolify, True)
    return options


def handle_options(name, argv):
    """Parse options from the command line and populate with defaults"""
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1, type='string')
    (poptions, args) = parser.parse_args(argv)
    # Get default options
    options = default_kwargs()
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template

    # Get template specific options
    template_kwargs = template_module[options['template']].kwargs()
    options.update(template_kwargs)
    options['name'] = (str, name)
    options = handle_args(argv, options)
    options['template'] = template_module[options['template']].template
    return options

def gen_arrow(batch_size, model_tensorflow, options):
    angles, outputs = getn(model_tensorflow(**options), 'inputs', 'outputs')
    arrow = graph_to_arrow(outputs,
                           input_tensors=angles,
                           name="robot_fwd_kinematics")
    return arrow

def rand_input(batch_size, n_angles, n_lengths):
    input_data = []
    for _ in range(n_angles):
        input_data.append(np.random.rand(batch_size, 1) * 90)
    for _ in range(n_lengths):
        input_data.append(np.random.rand(batch_size, 1))
    return input_data

def gen_rand_data(batch_size, model_tensorflow, options):
    """Generate data for training"""
    graph = tf.Graph()
    n_links, n_angles, n_lengths = getn(options, 'n_links', 'n_angles', 'n_lengths')
    final_out_data = []
    final_in_data = []
    with graph.as_default():
        sess = tf.Session()
        for i in range(10):
            inputs, outputs = getn(model_tensorflow(batch_size, n_links), 'inputs', 'outputs')
            input_data = rand_input(batch_size, n_angles, n_lengths)
            output_data = sess.run(outputs, feed_dict=dict(zip(inputs, input_data)))
            final_out_data.append(output_data)
            final_in_data.append(input_data)
        sess.close()

    noutputs = len(final_out_data[0])
    all_all_out_data = []
    for j in range(noutputs):
        all_data = [final_out_data[i][j] for i in range(10)]
        res = np.concatenate(all_data)
        all_all_out_data.append(res)

    ninputs = len(final_in_data[0])
    all_all_in_data = []
    for j in range(ninputs):
        all_data = [final_in_data[i][j] for i in range(10)]
        res = np.concatenate(all_data)
        all_all_in_data.append(res)

    return {'inputs': all_all_in_data, 'outputs': all_all_out_data}

def pi_supervised(options):
    """Neural network enhanced Parametric inverse! to do supervised learning"""
    tf.reset_default_graph()
    batch_size = options['batch_size']
    model_tensorflow = options['model']
    gen_data = options['gen_data']

    arrow = gen_arrow(batch_size, model_tensorflow, options)
    inv_arrow = inv_fwd_loss_arrow(arrow)
    right_inv = unparam(inv_arrow)
    sup_right_inv = supervised_loss_arrow(right_inv)
    # Get training and test_data
    train_data = gen_data(batch_size, model_tensorflow, options)
    test_data = gen_data(batch_size, model_tensorflow, options)

    # Have to switch input from output because data is from fwd model
    train_input_data = train_data['outputs']
    train_output_data = train_data['inputs']
    test_input_data = test_data['outputs']
    test_output_data = test_data['inputs']
    num_params = get_tf_num_params(right_inv)
    print("Number of params", num_params)
    supervised_train(sup_right_inv,
                     train_input_data,
                     train_output_data,
                     test_input_data,
                     test_output_data,
                     callbacks=[save_every_n, save_everything_last, save_options],
                     options=options)


def nn_supervised(options):
    """Plain neural network to do supervised learning"""
    tf.reset_default_graph()
    n_inputs = options['n_inputs']
    n_outputs = options['n_outputs']
    batch_size = options['batch_size']
    model_tensorflow = options['model']
    gen_data = options['gen_data']
    # Get training and test_data
    train_data = gen_data(batch_size, model_tensorflow, options)
    test_data = gen_data(batch_size, model_tensorflow, options)

    # Have to switch input from output because data is from fwd model
    train_input_data = train_data['outputs']
    train_output_data = train_data['inputs']
    test_input_data = test_data['outputs']
    test_output_data = test_data['inputs']

    template = res_net.template
    n_layers = 2
    l = round(max(*layer_width(2, n_inputs, n_layers, 630))) * 2
    tp_options = {'layer_width': l,
                  'num_layers': 2,
                  'nblocks': 1,
                  'block_size': 1,
                  'reuse': False}

    tf_arrow = TfArrow(n_outputs, n_inputs, template=template, options=tp_options)
    for port in tf_arrow.ports():
        set_port_shape(port, (None, 1))
    sup_tf_arrow = supervised_loss_arrow(tf_arrow)
    num_params = get_tf_num_params(sup_tf_arrow)
    print("NNet Number of params", num_params)
    supervised_train(sup_tf_arrow,
                     train_input_data,
                     train_output_data,
                     test_input_data,
                     test_output_data,
                     callbacks=[save_every_n, save_everything_last, save_options],
                     options=options)


# Benchmarks
def nn_benchmarks(model_name):
    options = handle_options(model_name, sys.argv[1:])
    # options['batch_size'] = np.round(np.logspace(0, np.log10(500-1), 10)).astype(int)
    options['batch_size'] = [128]
    options['error'] = ['error']
    if model_name == 'linkage_kinematics':
        options['description'] = "Neural Network Linkage Generalization Benchmark"
        options.update({'n_links': 3, 'n_angles': 3, 'n_lengths':0})
        options['model'] = robo_tensorflow
        options['n_outputs'] = 2
        options['gen_data'] = gen_rand_data
        options['n_inputs'] = 3
    if model_name == 'stanford_kinematics':
        options['description'] = "Neural Network Linkage Generalization Benchmark"
        options.update({'n_links': 6, 'n_angles': 5, 'n_lengths':1})
        options['model'] = stanford_tensorflow
        options['n_outputs'] = 12
        options['gen_data'] = gen_rand_data
        options['n_inputs'] = 6
    prefix = rand_string(5)
    test_everything(nn_supervised, options, ["batch_size", "error"], prefix=prefix)


def all_benchmarks(model_name):
    options = handle_options(model_name, sys.argv[1:])
    # options['batch_size'] = np.round(np.logspace(0, np.log10(500-1), 10)).astype(int)
    options['batch_size'] = [128]
    options['error'] = ['inv_fwd_error'] # , 'inv_fwd_error', 'error', 'sub_arrow_error']
    if model_name == 'linkage_kinematics':
        options['description'] = "Parametric Inverse Linkage Generalization Benchmark"
        options.update({'n_links': 3, 'n_angles': 3, 'n_lengths':0})
        options['model'] = robo_tensorflow
        options['gen_data'] = gen_rand_data
    if model_name == 'stanford_kinematics':
        options['description'] = "Parametric Inverse Linkage Generalization Benchmark"
        options.update({'n_links': 6, 'n_angles': 5, 'n_lengths':1})
        options['model'] = stanford_tensorflow
        options['gen_data'] = gen_rand_data

    prefix = rand_string(5)
    test_everything(pi_supervised, options, ["batch_size", "error"], prefix=prefix)
