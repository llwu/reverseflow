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


def test_generalization(inv_arrow: Arrow, data):
    # Split up data into batches
    partition = log_partition(xs)

    # train, record errors
    for sub_data in partition:
        # Create generator
        # Create convergence criteria
        # train

    return errors


# TODO:
# 1. Deal with batching
# add generators to train
# add test evaluation to train
# Create convergence criteria
# simple composition with neural net
# combine arrow with tensor templates
# have way to save data
# automate plot generation
