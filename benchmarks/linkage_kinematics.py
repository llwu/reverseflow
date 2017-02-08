"""(Inverse) kinematics of a linkage robot arm  in two dimensions"""
import tensorflow as tf
from arrows.util.viz import show_tensorboard_graph
from arrows.config import floatX
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow
from reverseflow.train.loss import inv_fwd_loss_arrow
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Interactive Plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False)
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
# axis = plt.axis([-3, 3, -3, 3])
lines = None


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
    x_terms = [0.0] + [np.cos(accum_angles[i]) for i in range(n_links)]
    y_terms = [0.0] + [np.sin(accum_angles[i]) for i in range(n_links)]
    return x_terms, y_terms

def plot_call_back(fetch_res):
    # import pdb; pdb.set_trace()
    global lines
    global ax
    n_links = 2
    angles = fetch_res['output_tensors'][0:n_links]
    x, y = draw_lines(n_links, angles)
    # import pdb; pdb.set_trace()
    print(sum(x), sum(y))
    if lines is None:
        lines = Line2D(x, y)
        ax.add_line(lines)
        plt.draw()
    else:
        lines.set_data(x, y)
    plt.draw()
    plt.show()
    plt.pause(0.05)
    # robot_joints = output_values[3:3+6]
    # r = np.array(robot_joints).flatten()
    # plot_robot_arm(list(r), target)

from arrows.port_attributes import *
from arrows.apply.propagate import *


def print_one_per_line(xs:Sequence):
    for x in xs:
        print(x)

def test_robot_arm(batch_size=128):
    lengths = [1, 1]
    with tf.name_scope("fwd_kinematics"):
        angles = [tf.placeholder(floatX(), name="theta", shape=(batch_size, 1)) for i in range(len(lengths))]
        x, y = gen_robot(lengths, angles)
    arrow = graph_to_arrow([x, y],
                           input_tensors=angles,
                           name="robot_fwd_kinematics")
    show_tensorboard_graph()
    tf.reset_default_graph()
    inv_arrow = invert(arrow)
    # inv_arrow = inv_fwd_loss_arrow(arrow)
    port_attr = propagate(inv_arrow)
    lsa = list(inv_arrow.get_sub_arrows())
    import pdb; pdb.set_trace()

    min_approx_error_arrow(inv_arrow,
                           [0.5, 0.5],
                           error_filter=lambda port: has_port_label(port, "sub_arrow_error"),
                           output_call_back=plot_call_back)


test_robot_arm()
