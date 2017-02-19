"""Test generalization"""
import numpy as np
from typing import Sequence

def log_partition(xs: Sequence, n_cells: int=10):
    # FIXME: Make overlapping sets and cell size double
    """Partition xs into a n_cells cells of logarithmically increasing size"""
    indices = np.round(np.logspace(0, np.log10(len(xs)-1), n_cells))
    print(indices)
    start_i = 0
    partition = []
    for end_i in indices:
        partition.append(xs[start_i: end_i])
        start_i = end_i
    return partition


def test_generalization(run_me, options=None):
    """Train parametric inverse and vanilla neural network with different
    amounts of data and see the test_error
    """
    options = {} if options is None else options
    batch_sizes = [1, 4, 40, 100]
    for batch_size in batch_sizes:
        options['batch_size'] = batch_size
        run_me(options)
