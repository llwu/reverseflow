"""Generators"""
import numpy as np
from pdt.util.misc import identity

# Minibatching
def infinite_samples(sampler, batch_size, shape, add_batch=False):
    while True:
        if add_batch:
            shape = (batch_size,)+shape
        yield sampler(*shape)


def infinite_batches(inputs, batch_size, f=identity, shuffle=False):
    """Create generator which without termintation yields batch_size chunk
    of inputs
    Args:
        inputs:
        batch_size:
        f: arbitrary function to apply to batch
        Shuffle: If True randomly shuffles ordering
    """
    start_idx = 0
    nelements = len(inputs)
    indices = np.arange(nelements)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    while True:
        end_idx = start_idx + batch_size
        if end_idx > nelements:
            diff = end_idx - nelements
            excerpt = np.concatenate([indices[start_idx:nelements], indices[0:diff]])
            start_idx = diff
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
            start_idx = start_idx + batch_size
        yield f(inputs[excerpt])


def constant_batches(x, f):
    while True:
        data = yield
        yield f(x, data)


def iterate_batches(inputs, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_iiteratedx + batch_size)
        yield inputs[excerpt]
