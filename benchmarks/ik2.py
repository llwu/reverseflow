import tensorflow as tf
from tensorflow import float32
import numpy as np


def tensor_rand(tensors):
    return {t: np.random.rand(*t.get_shape().as_list()) for t in tensors}

sin = tf.sin
cos = tf.cos

g = tf.get_default_graph()
with g.name_scope("fwd_g"):
    phi1 = tf.placeholder(float32, name="phi1", shape=())
    phi2 = tf.placeholder(float32, name="phi2", shape=())
    phi3 = tf.placeholder(float32, name="phi3", shape=())
    phi4 = tf.placeholder(float32, name="phi4", shape=())
    phi5 = tf.placeholder(float32, name="phi5", shape=())
    phi6 = tf.placeholder(float32, name="phi6", shape=())
    r11 = -sin(phi6)*(cos(phi4)*sin(phi1)+cos(phi1)*cos(phi2)*sin(phi4)) - \
        cos(phi6)*(cos(phi5)*(sin(phi1)*sin(phi4)-cos(phi1)*cos(phi2) * cos(phi4))+cos(phi1)*sin(phi2)*sin(phi5))
