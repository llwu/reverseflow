import tensorflow as tf
from pi.util import smthg_like
## Error measures
## ==============

## Interval Bounds
def linear_interval_loss(t, a, b, eps=1e-9):
    """
    Returns linear distance between a value and an inteveral a,b
    """
    with t.graph.name_scope("linear_interval_loss"):
        loss = tf.maximum(tf.maximum(eps,t-1), tf.maximum(eps,-t)) + eps
        return tf.maximum(loss, eps) + eps

## Integer Bounds ()
def nearest_integer_loss(t):
    """
    Returns the distance between a tensor t and the nearest Integer
    """
    with t.graph.name_scope("nearest_integer_loss"):
        return tf.abs(t - tf.round(t))

## A or B
def a_b_clamp(t, a, b, dist=tf.abs):
    """
    Clamp to the nearest one
    """
    with t.graph.name_scope("a_b_clamp"):
        a_dist = dist(t - a)
        b_dist = dist(t - b)
        tf.select(a_dist < b_dist, smthg_like(t, a), smthg_like(t, b))

def nearest_a_b_loss(t, a, b, dist=tf.abs):
    """
    Return 0 when t = a or t = b, otherwise return linear distance to nearest
    """
    with t.graph.name_scope("nearest_a_b_loss"):
        a_dist = dist(t - a)
        b_dist = dist(t - b)
        return tf.minimum(a_dist, b_dist)

## Clamps
## =====

# tf.clip_by_value

## Interval
