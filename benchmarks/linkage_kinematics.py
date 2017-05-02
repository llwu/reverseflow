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
from common import *
import sys
from typing import Sequence
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
import numpy as np

# Interactive Plotting
#plt.ion()

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


def plot_callback():
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
        x = fetch_res['input_tensors'][0][0, 0]
        y = fetch_res['input_tensors'][1][0, 0]
        repeats = 0
        if i % 1000 == 0:
            for j in range(len(fetch_res['input_tensors'][1])):
                repeats += 1
                if fetch_res['input_tensors'][0][j, 0] != x:
                    break
                if fetch_res['input_tensors'][1][j, 0] != y:
                    break

            for j in range(repeats):
                angles = [batch_angles[i][j, 0] for i in range(len(batch_angles))]
                x, y = draw_lines(n_links, angles)
                if lines is None:
                    lines = Line2D(x, y)
                    ax.add_line(lines)
                    x = fetch_res['input_tensors'][0][j, 0]
                    y = fetch_res['input_tensors'][1][j, 0]
                    circle = plt.Circle((x, y), 0.1, color='r')
                    ax.add_artist(circle)
                    plt.draw()
                else:
                    lines.set_data(x, y)
                    x = fetch_res['input_tensors'][0][j, 0]
                    y = fetch_res['input_tensors'][1][j, 0]
                    circle.center = (x, y)
                    plt.draw()
                    plt.show()
                    plt.pause(0.05)
            # plt.pause(0.2)
    return closure

def robo_tensorflow(batch_size, n_links, **options):
    lengths = [1 for i in range(n_links)]
    with tf.name_scope("fwd_kinematics"):
        angles = []
        for _ in range(n_links):
            angles.append(tf.placeholder(floatX(),
                                         name="theta",
                                         shape=(batch_size, 1)))
        x, y = gen_robot(lengths, angles)
    return {'inputs':angles, 'outputs':[x,y]}


from common import gen_rand_data
if __name__ == "__main__":
    #plot_cb = plot_callback()
    options = {'model': robo_tensorflow,
               'n_links': 3,
               'n_angles': 3,
               'n_lengths': 0,
               'n_inputs': 3,
               'n_outputs' : 2,
               'phi_shape' : (3,),
               'gen_data': gen_rand_data,
               'model_name': 'linkage_kinematics',
               'error': ['supervised_error', 'inv_fwd_error'],
               'callbacks': [save_options, save_every_n, save_everything_last]}
               #'error': ['supervised_error', 'inv_fwd_error']
    nn = False
    if nn:
        options["run"] = "Neural Network Linkage Generalization Benchmark"
        f = nn_benchmarks
    else:
        options['run'] = "Parametric Inverse Linkage Generalization Benchmark"
        # f = pi_benchmarks
        f = pi_reparam_benchmarks
    f('linkage_kinematics', options)
