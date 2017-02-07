"""Generators"""
import numpy as np
from pdt.util.misc import identity

# Minibatching
def infinite_samples(sampler, batchsize, shape):
    while True:
        to_sample_shape = (batchsize,)+shape
        yield sampler(*to_sample_shape)


def infinite_batches(inputs, batchsize, f=identity, shuffle=False):
    start_idx = 0
    nelements = len(inputs)
    indices = np.arange(nelements)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    while True:
        end_idx = start_idx + batchsize
        if end_idx > nelements:
            diff = end_idx - nelements
            excerpt = np.concatenate([indices[start_idx:nelements], indices[0:diff]])
            start_idx = diff
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
        else:
            excerpt = indices[start_idx:start_idx + batchsize]
            start_idx = start_idx + batchsize
        yield f(inputs[excerpt])


def constant_batches(x, f):
    while True:
        data = yield
        yield f(x, data)


def iterate_batches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_iiteratedx + batchsize)
        yield inputs[excerpt]
