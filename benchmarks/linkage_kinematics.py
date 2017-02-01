"""(Inverse) kinematics of a linkage robot arm  in two dimensions"""
import tensorflow as tf
from arrows.util.viz import show_tensorboard_graph
from arrows.config import floatX
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow
from typing import Sequence


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



def test_robot_arm():
    lengths = [0.9397378938990306, 1.7201786764100944]
    with tf.name_scope("fwd_kinematics"):
        angles = [tf.placeholder(floatX(), name="theta", shape=()) for i in range(len(lengths))]
        x, y = gen_robot(lengths, angles)
    arrow = graph_to_arrow([x, y], name="robot_fwd_kinematics")
    show_tensorboard_graph()
    tf.reset_default_graph()
    inv_arrow = invert(arrow)
    min_approx_error_arrow(inv_arrow, [0.9397378938990306, 1.7201786764100944])


test_robot_arm()
