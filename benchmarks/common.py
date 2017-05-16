"""Functions common for examples"""
import sys
from arrows.util.io import *
from arrows.apply.propagate import propagate
from arrows.util.misc import rand_string, getn
from metrics.generalization import test_everything
from reverseflow.train.common import layer_width
from reverseflow.train.reparam import *
from reverseflow.train.unparam import unparam
from reverseflow.train.loss import inv_fwd_loss_arrow, supervised_loss_arrow
from reverseflow.train.supervised import supervised_train
from reverseflow.train.callbacks import save_callback, save_options, save_every_n, save_everything_last
from reverseflow.invert import invert
from wacacore.util.misc import boolify
from analysis import best_hyperparameters
import numpy as np
import tensorflow as tf
from tensortemplates.module import template_module


def default_benchmark_options():
    """Default options"""
    options = {}
    options['learning_rate'] = (float, 0.01)
    options['update'] = (str, 'adam')
    options['params_file'] = (str, "")
    options['momentum'] = (float, 0.9)
    options['description'] = (str, "")
    options['batch_size'] = (int, 128)
    options['debug'] = (boolify, False)
    options['data_size'] = (int, 500)
    options['save_every'] = (int, 100)
    options['compress'] = (boolify, 0,)
    options['num_iterations'] = (int, 1000)
    options['dirname'] = (str, "dirname") # FIXME: Remove this
    options['datadir'] = (str, os.path.join(os.environ['DATADIR'], "rf"))

    # options['save'] = (boolify, True)
    options['template'] = (str, 'res_net')
    options['template_name'] = (str, 'res_net')
    # options['train'] = (bool, True)
    options['script'] = (boolify, False)
    return options

# FIXME: Delete me
def handle_options(name, argv):
    """Parse options from the command line and populate with defaults"""
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1, type='string')
    (poptions, args) = parser.parse_args(argv)
    # Get default options
    options = default_benchmark_options()
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template

    # Get template specific options
    template_kwargs = template_module[options['template']].kwargs()
    options.update(template_kwargs)
    options = handle_args(argv, options)
    options['template_name'] = options['template']
    options['template'] = template_module[options['template']].template
    return options


def add_additional_options(argv):
    """Add options which only exist depending on other options"""
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1,
                      type='string')
    (poptions, args) = parser.parse_args(argv)
    # Get default options
    options = {}
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template

    # Get template specific options
    options.update(template_module[options['template']].kwargs())
    return options


# Training stuff
def gen_arrow(batch_size, model_tensorflow, options):
    inputs, outputs = getn(model_tensorflow(**options), 'inputs', 'outputs')
    name = options['model_name']
    arrow = graph_to_arrow(outputs,
                           input_tensors=inputs,
                           name=name)
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
    data_size = options["data_size"]
    # assert data_size % batch_size == 0 or batch_size, "Dataset size must be multipel of batch_size"
    nruns = data_size  // batch_size

    # FIXME This hack
    if nruns == 0:
        nruns = 1
    with graph.as_default():
        sess = tf.Session()
        for i in range(nruns):
            inputs, outputs = getn(model_tensorflow(batch_size, n_links), 'inputs', 'outputs')
            input_data = rand_input(batch_size, n_angles, n_lengths)
            output_data = sess.run(outputs, feed_dict=dict(zip(inputs, input_data)))
            final_out_data.append(output_data)
            final_in_data.append(input_data)
        sess.close()

    noutputs = len(final_out_data[0])
    all_all_out_data = []
    for j in range(noutputs):
        all_data = [final_out_data[i][j] for i in range(nruns)]
        res = np.concatenate(all_data)
        all_all_out_data.append(res)

    ninputs = len(final_in_data[0])
    all_all_in_data = []
    for j in range(ninputs):
        all_data = [final_in_data[i][j] for i in range(nruns)]
        res = np.concatenate(all_data)
        all_all_in_data.append(res)

    return {'inputs': all_all_in_data, 'outputs': all_all_out_data}


def pi_supervised(options):
    """Neural network enhanced Parametric inverse! to do supervised learning
    Args:
      batch_size: the batch_size
      model_tensorflow: f: options -> {'inputs': inp_tensors, 'outputs': out_tensors}
      gen_data
    """
    tf.reset_default_graph()
    batch_size = options['batch_size']
    model_tensorflow = options['model']
    gen_data = options['gen_data']

    arrow = gen_arrow(batch_size, model_tensorflow, options)
    inv_arrow = invert(arrow)
    inv_arrow = inv_fwd_loss_arrow(arrow, inv_arrow)
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
    import pdb; pdb.set_trace()
    print("Number of params", num_params)
    # print("NNet Number of params", num_params)
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

    arrow = gen_arrow(batch_size, model_tensorflow, options)
    port_attr = propagate(arrow)
    tf_arrow = TfArrow(n_outputs, n_inputs, template=template, options=tp_options)
    for i, port in enumerate(tf_arrow.in_ports()):
        shp = port_attr[arrow.out_port(i)]['shape']
        set_port_shape(port, shp)

    for i, port in enumerate(tf_arrow.out_ports()):
        shp = port_attr[arrow.in_port(i)]['shape']
        set_port_shape(port, shp)

    #
    # # FIXME: This is not genreal
    # for port in tf_arrow.ports():
    #     set_port_shape(port, (batch_size, 1))

    tf_arrow = inv_fwd_loss_arrow(arrow, tf_arrow)

    sup_tf_arrow = supervised_loss_arrow(tf_arrow)
    num_params = get_tf_num_params(sup_tf_arrow)
    import pdb; pdb.set_trace()
    print("NNet Number of params", num_params)
    supervised_train(sup_tf_arrow,
                     train_input_data,
                     train_output_data,
                     test_input_data,
                     test_output_data,
                     callbacks=[save_every_n, save_everything_last, save_options],
                     options=options)


# Reparameterization
def pi_reparam(options):
    """Neural network enhanced Parametric inverse! to do supervised learning"""
    tf.reset_default_graph()
    batch_size = options['batch_size']
    model_tensorflow = options['model']
    gen_data = options['gen_data']
    phi_shape = options['phi_shape']
    n_links = options['n_links']

    arrow = gen_arrow(batch_size, model_tensorflow, options)
    inv_arrow = invert(arrow)
    inv_arrow = inv_fwd_loss_arrow(arrow, inv_arrow)
    rep_arrow = reparam(inv_arrow, (batch_size,) + phi_shape)
    def sampler(*x):
        return np.random.rand(*x)*n_links
    frac_repeat = 0.25
    nrepeats = int(np.ceil(batch_size * frac_repeat))
    train_input1 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))
    train_input2 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))
    test_input1 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))
    test_input2 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))
    d = [p for p in inv_arrow.out_ports() if not is_error_port(p)]
    # plot_cb = plot_callback(batch_size)

    # callbacks = [] + options['callbacks']
    reparam_train(rep_arrow,
                  d,
                  [train_input1, train_input2],
                  [test_input1, test_input2],
                  options=options)


# Benchmarks
def nn_benchmarks(model_name, options=None):
    options = {} if options is None else options
    options.update(handle_options(model_name, sys.argv[1:]))
    options['data_size'] = [1, 7, 15, 25, 36, 50, 70, 90, 120, 150]# [int(ds) for ds in np.round(np.logspace(0, np.log10(500-1), 10)).astype(int)]
    options['error'] = ['inv_fwd_error']
    prefix = rand_string(5)
    test_everything(nn_supervised, options, ["error",], prefix=prefix, nrepeats=3)


def pi_reparam_benchmarks(model_name, options=None):
    options = {} if options is None else options
    options.update(handle_options(model_name, sys.argv[1:]))
    options['error'] = ['inv_fwd_error'] # , 'inv_fwd_error', 'error', 'sub_arrow_error']
    prefix = rand_string(5)
    options['learning_rate'] = np.linspace(0.00001, 0.1, 30)
    options['lambda'] = np.linspace(1, 10, 30)
    # pi_reparam(options)
    test_everything(pi_reparam, options, ['learning_rate', 'lambda'], prefix=prefix, nrepeats=1)
    learning_rate, lmbda = best_hyperparameters(prefix, ['learning_rate', 'lambda'], options['num_iterations'])
    print(learning_rate, lmbda)


def pi_benchmarks(model_name, options=None):
    options = {} if options is None else options
    options.update(handle_options(model_name, sys.argv[1:]))
    options['data_size'] = [1, 7, 15, 25, 36, 50, 70, 90, 120, 150]# [int(ds) for ds in np.round(np.logspace(0, np.log10(500-1), 10)).astype(int)]
    # options['data_size'] = [int(ds) for ds in np.round(np.logspace(0, np.log10(500-1), 10)).astype(int)]
    options['error'] = ['supervised_error'] # , 'inv_fwd_error', 'error', 'sub_arrow_error']
    prefix = rand_string(5)
    test_everything(pi_supervised, options, ["error", 'data_size'], prefix=prefix, nrepeats=3)
