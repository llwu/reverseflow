"""Primitive Random Variables"""
import tensorflow as tf

def exponential(w, l):
    """Exponentially distributed random variable"""
    return - tf.log(1-w)/l

def logistic(w, mu, s):
    """Logistically distributed random variable"""
    mu + s * tf.log(w/(1-w))
