"""Inverse Kinematics of a 5 Dimensional Linkage Robot Arm in Two Dimensions"""
import tensorflow as tf
from typing import Sequence
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from arrows.config import floatX
from reverseflow.train.train_y import min_approx_error_arrow


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
    return accum


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
    accum_angles = accum_sum(angles)
    x_terms = [lengths[i]*tf.cos(accum_angles[i]) for i in range(n_links)]
    y_terms = [lengths[i]*tf.sin(accum_angles[i]) for i in range(n_links)]
    return sum(x_terms), sum(y_terms)


def test_robot_arm():
    lengths = [1, 2, 3]
    angles = [tf.placeholder(floatX(), name="theta") for i in range(len(lengths))]
    x, y = gen_robot(lengths, angles)
    arrow = graph_to_arrow([x, y])
    inv_arrow = invert(arrow)
    min_approx_error_arrow(arrow)


test_robot_arm()
