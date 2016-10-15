"""Assortment of utilities"""
import tensorflow as tf
import numpy as np
from collections import OrderedDict


def infinite_input(gen_graph, batch_size, seed):
    generator_graph = tf.Graph()
    with generator_graph.as_default() as g:
        in_out_var = gen_graph(g, batch_size, False, seed=seed)
        sess = tf.Session(graph=generator_graph)
        init = tf.initialize_all_variables()

    while True:
        with generator_graph.as_default() as g:
            sess.run(init)
            output = sess.run(in_out_var['outputs'])
        yield output


def infinite_samples(sampler, shape):
    while True:
        yield sampler(*shape)


def dictionary_gen(x):
    while True:
        yield {k: next(v) for k, v in x.items()}


def same(xs) -> Bool:
    """All elements in xs are the same?"""
    if len(xs) == 0:
        return True
    else:
        x1 = xs[0]
        for xn in xs:
            if xn != x1:
                return False

    return True
