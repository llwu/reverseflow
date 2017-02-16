"""(Inverse) kinematics of a linkage robot arm  in two dimensions"""
import tensorflow as tf
from arrows.util.viz import show_tensorboard_graph
from arrows.util.misc import print_one_per_line
from arrows.config import floatX
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow
from reverseflow.train.loss import inv_fwd_loss_arrow
from arrows.port_attributes import *
from arrows.apply.propagate import *
from reverseflow.train.reparam import *

from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Interactive Plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
# axis = plt.axis([-3, 3, -3, 3])
lines = None
circle = None


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

batch_num = 0
i = 0
BATCH_SIZE = 512

def plot_call_back(fetch_res):
    # import pdb; pdb.set_trace()
    global lines
    global ax
    global i
    global batch_num
    global circle
    global BATCH_SIZE
    batch_size = BATCH_SIZE
    i = i + 1
    n_links = 3
    batch_angles = fetch_res['output_tensors'][0:n_links]
    if i % 30 == 0:
        batch_num = np.random.randint(batch_size)

    angles = [batch_angles[i][batch_num, 0] for i in range(len(batch_angles))]
    x, y = draw_lines(n_links, angles)
    if lines is None:
        lines = Line2D(x, y)
        ax.add_line(lines)
        x = fetch_res['input_tensors'][1][batch_num, 0]
        y = fetch_res['input_tensors'][2][batch_num, 0]
        circle = plt.Circle((x, y), 0.1, color='r')
        ax.add_artist(circle)
        plt.draw()
    else:
        lines.set_data(x, y)
        x = fetch_res['input_tensors'][1][batch_num, 0]
        y = fetch_res['input_tensors'][2][batch_num, 0]
        circle.center = (x, y)
    plt.draw()
    plt.show()
    plt.pause(0.05)
    # robot_joints = output_values[3:3+6]
    # r = np.array(robot_joints).flatten()
    # plot_robot_arm(list(r), target)

def test_robot_arm():
    global BATCH_SIZE
    batch_size = BATCH_SIZE
    lengths = [1, 1, 1]
    with tf.name_scope("fwd_kinematics"):
        angles = [tf.placeholder(floatX(), name="theta", shape=(batch_size, 1)) for i in range(len(lengths))]
        x, y = gen_robot(lengths, angles)
    arrow = graph_to_arrow([x, y],
                           input_tensors=angles,
                           name="robot_fwd_kinematics")
    tf.reset_default_graph()
    # inv_arrow = invert(arrow)
    inv_arrow = inv_fwd_loss_arrow(arrow)
    rep_arrow = reparam(inv_arrow, (batch_size, len(lengths),))
    port_attr = propagate(rep_arrow)

    inv_input1 = np.tile([0.5], (batch_size, 1))
    inv_input2 = np.tile([0.5], (batch_size, 1))
    nlinks = len(lengths)
    # inv_input1 = np.random.rand(batch_size, 1)*(nlinks-1)
    # inv_input2 = np.random.rand(batch_size, 1)*(nlinks-1)
    test_input1 = np.random.rand(batch_size, 1)*(nlinks-1)
    test_input2 = np.random.rand(batch_size, 1)*(nlinks-1)


    d = [p for p in inv_arrow.out_ports() if not is_error_port(p)]
    reparam_arrow(rep_arrow,
                  d,
                  [inv_input1, inv_input2],
                  [test_input1, test_input2],
                  error_filter=lambda port: has_port_label(port, "inv_fwd_error"),
                #   error_filter=lambda port: has_port_label(port, "sub_arrow_error"),
                #   error_filter="inv_fwd_error",
                  batch_size=batch_size,
                  output_call_back=plot_call_back)
    # min_approx_error_arrow(rep_arrow,
    #                        [inv_input1, inv_input2],
    #                     #    error_filter=lambda port: has_port_label(port, "sub_arrow_error"),
    #                        output_call_back=plot_call_back)


test_robot_arm()
