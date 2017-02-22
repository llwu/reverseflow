"""Assortment of utilities without dependencies"""
import tensorflow as tf  # TODO: Remove this dependency from this file
import numpy as np
from typing import Sequence, TypeVar

T = TypeVar('T')


def pos_in_seq(x: T, ys: Sequence[T]) -> int:
    """Return the index of a value in a list"""
    for i, y in enumerate(ys):
        if x == y:
            return i
    assert False, "element not in list"


def same(xs) -> bool:
    """All elements in xs are the same?"""
    if len(xs) == 0:
        return True
    else:
        x1 = xs[0]
        for xn in xs:
            if xn != x1:
                return False

    return True


def complement(indices: Sequence, shape: Sequence) -> Sequence:
    bools = np.zeros(shape)
    output = []
    for index in indices:
        bools[tuple(index)] = 1
    for index, value in np.ndenumerate(bools):
        if value == 0:
            output.append(index)
    return np.squeeze(np.array(output))


def complement_bool(indices: np.ndarray, shape: Sequence) -> Sequence:
    if len(indices.shape) > 2:
        indices = indices.reshape(-1, indices.shape[-1])
    bools = np.zeros(shape)
    for index in indices:
        bools[tuple(index)] = 1
    return bools


# Generators
# ==========
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
