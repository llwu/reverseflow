"""Test generalization"""
from arrows.util.generators import infinite_batches

import numpy as np
from typing import Sequence

def minimal_change(xs: Sequence) -> bool:
    ...

def stop_if_converged(xs: losses):
    losses = fetch_res['losses']
    okokok
    ...


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


def test_generalization(inv_arrow: Arrow, train_data, test_data,
                        batch_size=512, **kwargs):
    """Test how well inv_arrow generalizses from varying number of examples"""
    # Split up data into batches
    partition = log_partition(test_data)

    # train, record errors
    for sub_data in partition:
        batch_size = np.min(batch_size, len(sub_data))
        # Create generator
        gen = infinite_batches(sub_data, batch_size, shuffle=True)

        # Create convergence criteria
        stop = stop_if_converged()

        # train


    return errors



# TODO:
# 1. Deal with batching
# 2. Get test_data
