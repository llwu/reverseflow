"""(Inverse) kinematics of a linkage robot arm  in two dimensions"""
import tensorflow as tf
from arrows.util.viz import show_tensorboard_graph
from arrows.util.misc import print_one_per_line
from arrows.config import floatX
from arrows.util.io import mk_dir
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow
from reverseflow.train.loss import inv_fwd_loss_arrow, supervised_loss_arrow
from reverseflow.train.common import get_tf_num_params
from arrows.port_attributes import *
from arrows.apply.propagate import *
from reverseflow.train.reparam import *
from reverseflow.train.unparam import unparam
from reverseflow.train.callbacks import save_callback, save_options, save_every_n, save_everything_last
from reverseflow.train.supervised import supervised_train
from metrics.generalization import test_generalization, test_everything
from tensortemplates import res_net
from common import handle_options, gen_sfx_key
import sys
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Interactive Plotting
plt.ion()

def accum_sum(xs: Sequence):
    """Return accumulative reduction
    Args:
        xs: Input sequence of addable values

    Returns:
        [xs[0], xs[0]+xs[1],...,xs[0]+...+xs[n-1]]
    """
    accum = [xs[0]]
    total = xs[0]
    for i in range(1, len(xs)):
        total = total + xs[i]
        accum.append(total)
    return accum, total


def gen_robot(lengths: Sequence, angles: Sequence):
    """
    Create a tensorflow graph of robot arm (linkage) in 2d plane
    Args:
        lengths: linkage lengths
        angles: linkage angles

    Returns
        x: Scalar Tensor for x cartesian coordinate
        y: Scalar Tensor for y cartesian coordinate
    """
    assert len(lengths) == len(angles)
    n_links = len(lengths)
    accum_angles, total_angles = accum_sum(angles)
    # x_terms = [lengths[i]*tf.cos(accum_angles[i]) for i in range(n_links)]
    # y_terms = [lengths[i]*tf.sin(accum_angles[i]) for i in range(n_links)]
    x_terms = [tf.cos(accum_angles[i]) for i in range(n_links)]
    y_terms = [tf.sin(accum_angles[i]) for i in range(n_links)]
    # return tf.add_n(x_terms, name='x_cord'), tf.add_n(y_terms, name='y_cord')
    x_accum, x = accum_sum(x_terms)
    y_accum, y = accum_sum(y_terms)
    return x, y


def draw_lines(n_links, angles):
    accum_angles, total_angles = accum_sum(angles)
    x_terms = [np.cos(accum_angles[i]) for i in range(n_links)]
    y_terms = [np.sin(accum_angles[i]) for i in range(n_links)]
    x_accum, x = accum_sum(x_terms)
    y_accum, y = accum_sum(y_terms)
    return [0.0] + x_accum, [0.0] + y_accum


def plot_callback(batch_size):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    # axis = plt.axis([-3, 3, -3, 3])
    lines = None
    circle = None
    batch_num = 10
    def closure(fetch_res, feed_dict, i, **kwargs):
        nonlocal fig, ax, lines, circle, batch_num
        n_links = 3
        batch_angles = fetch_res['output_tensors'][0:n_links]
        x = fetch_res['input_tensors'][1][0, 0]
        y = fetch_res['input_tensors'][2][0, 0]
        repeats = 0
        for j in range(len(fetch_res['input_tensors'][1])):
            repeats += 1
            if fetch_res['input_tensors'][1][j, 0] != x:
                break
            if fetch_res['input_tensors'][2][j, 0] != y:
                break

        for j in range(repeats):
            angles = [batch_angles[i][j, 0] for i in range(len(batch_angles))]
            x, y = draw_lines(n_links, angles)
            if lines is None:
                lines = Line2D(x, y)
                ax.add_line(lines)
                x = fetch_res['input_tensors'][1][j, 0]
                y = fetch_res['input_tensors'][2][j, 0]
                circle = plt.Circle((x, y), 0.1, color='r')
                ax.add_artist(circle)
                plt.draw()
            else:
                lines.set_data(x, y)
                x = fetch_res['input_tensors'][1][j, 0]
                y = fetch_res['input_tensors'][2][j, 0]
                circle.center = (x, y)
                plt.draw()
                plt.show()
                plt.pause(0.05)
        plt.pause(0.2)
    return closure

def robo_tensorflow(batch_size, n_links):
    lengths = [1 for i in range(n_links)]
    with tf.name_scope("fwd_kinematics"):
        angles = []
        for _ in range(n_links):
            angles.append(tf.placeholder(floatX(),
                                         name="theta",
                                         shape=(None, 1)))
        x, y = gen_robot(lengths, angles)
    return {'inputs':angles, 'outputs':[x,y]}

def robot_arm_arrow(batch_size, n_links):
    angles, outputs = getn(robo_tensorflow(batch_size, n_links), 'inputs', 'outputs')
    arrow = graph_to_arrow(outputs,
                           input_tensors=angles,
                           name="robot_fwd_kinematics")
    return arrow

def gen_data(batch_size, n_links):
    """Generate data for training"""
    graph = tf.Graph()
    with graph.as_default():
        inputs, outputs = getn(robo_tensorflow(batch_size, n_links), 'inputs', 'outputs')
        input_data = [np.random.rand(batch_size, 1) for i in range(n_links)]
        sess = tf.Session()
        output_data = sess.run(outputs, feed_dict=dict(zip(inputs, input_data)))
        sess.close()
    return {'inputs': input_data, 'outputs': output_data}


def robot_arm(options):
    n_links = 3
    batch_size = options['batch_size']
    arrow = robot_arm_arrow(batch_size, n_links)
    tf.reset_default_graph()
    # inv_arrow = invert(arrow)
    inv_arrow = inv_fwd_loss_arrow(arrow)
    rep_arrow = reparam(inv_arrow, (batch_size, n_links))
    import pdb; pdb.set_trace()


    def sampler(*x):
        return np.random.rand(*x)*n_links
    frac_repeat = 0.25
    nrepeats = int(np.ceil(batch_size * frac_repeat))
    train_input1 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))
    train_input2 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))
    test_input1 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))
    test_input2 = repeated_random(sampler, batch_size, nrepeats, shape=(1,))

    d = [p for p in inv_arrow.out_ports() if not is_error_port(p)]
    plot_cb = plot_callback(batch_size)

    reparam_train(rep_arrow,
                  d,
                  [train_input1, train_input2],
                  [test_input1, test_input2],
                  error_filter=lambda port: has_port_label(port, "inv_fwd_error"),
                #   error_filter=lambda port: has_port_label(port, "sub_arrow_error"),
                #   error_filter="inv_fwd_error",
                  callbacks=[plot_cb, save_callback],
                  options=options)


def pi_supervised(options):
    """Neural network enhanced Parametric inverse! to do supervised learning"""
    tf.reset_default_graph()
    n_links = 3
    batch_size = options['batch_size']
    arrow = robot_arm_arrow(batch_size, n_links)
    inv_arrow = inv_fwd_loss_arrow(arrow)
    right_inv = unparam(inv_arrow)
    sup_right_inv = supervised_loss_arrow(right_inv)
    # Get training and test_data
    train_data = gen_data(batch_size, n_links)
    test_data = gen_data(1024, n_links)

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
    n_links = 3
    batch_size = options['batch_size']
    # Get training and test_data
    train_data = gen_data(batch_size, n_links)
    test_data = gen_data(1024, n_links)

    # Have to switch input from output because data is from fwd model
    train_input_data = train_data['outputs']
    train_output_data = train_data['inputs']
    test_input_data = test_data['outputs']
    test_output_data = test_data['inputs']

    template = res_net.template
    n_layers = 2
    l = round(max(*layer_width(2, n_links, n_layers, 630))) * 2
    tp_options = {'layer_width': l,
                  'num_layers': 2,
                  'nblocks': 1,
                  'block_size': 1,
                  'reuse': False}

    tf_arrow = TfArrow(2, n_links, template=template, options=tp_options)
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


def main(argv):
    options = handle_options('linkage_kinematics', argv)
    sfx = gen_sfx_key(('nblocks', 'block_size'), options)
    options['sfx'] = sfx
    robot_arm(options)


# Benchmarks
def nn_benchmarks():
    options = handle_options('linkage_kinematics', sys.argv[1:])
    options['batch_size'] = np.round(np.logspace(0, np.log10(500-1), 10))
    options['error'] = ['error']
    options['description'] = "Neural Network Linkage Generalization Benchmark"
    options['save'] = True
    prefix = rand_string(5)
    test_everything(nn_supervised, options, ["batch_size", "error"], prefix=prefix)


def all_benchmarks():
    options = handle_options('linkage_kinematics', sys.argv[1:])
    options['batch_size'] = np.round(np.logspace(0, np.log10(500-1), 10))
    options['error'] = ['supervised_error', 'inv_fwd_error', 'error', 'sub_arrow_error']
    options['description'] = "Parametric Inverse Linkage Generalization Benchmark"
    options['save'] = True
    prefix = rand_string(5)
    test_everything(pi_supervised, options, ["batch_size", "error"], prefix=prefix)

if __name__ == "__main__":
    nn_benchmarks()
