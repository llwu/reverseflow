from arrows.apply.shapes import propagate_shapes
from test_arrows import all_test_arrow_gens
from totality_test import totality_test
from arrows import Arrow
import numpy as np


def input_gen(arrow: Arrow):
    ndim = 3
    maxdim = 100
    input_shape = tuple([np.random.randint(1, maxdim) for i in range(ndim)])
    input_shape = ()
    return [input_shape for i in range(arrow.num_in_ports())]


def test_shapes():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(propagate_shapes, all_test_arrows, input_gen,
                  test_name="shapes")
