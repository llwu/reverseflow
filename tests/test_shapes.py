import pdb

import numpy as np
from arrows import Arrow
from arrows.port_attributes import is_param_port
from arrows.apply.propagate import propagate
from reverseflow.inv_primitives.inv_math_arrows import InvAddArrow

from test_arrows import all_test_arrow_gens, test_inv_twoxyplusx
from totality_test import totality_test


def input_gen(arrow: Arrow):
    ndim = 3
    maxdim = 100
    input_shape = tuple([np.random.randint(1, maxdim) for i in range(ndim)])
    return {port: {'shape': (), 'value': 5} for port in arrow.out_ports()}  # if not is_param_port(port)}

def test_shapes():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(propagate, all_test_arrows, input_gen)


def manual_inspection():
    """Manually inspect output with PDB."""
    arrow = InvAddArrow()
    given_shapes = input_gen(arrow)
    known_shapes = propagate(arrow, given_shapes)
    pdb.set_trace()
    return known_shapes


if __name__ == '__main__':
    manual_inspection()
