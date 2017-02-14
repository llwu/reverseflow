from test_arrows import all_test_arrow_gens
from totality_test import totality_test
from arrows import Arrow
from arrows.apply.apply import apply
import numpy as np


def input_gen(arrow: Arrow):
    return [np.random.rand() for i in range(arrow.num_in_ports())]

def test_apply():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(apply, all_test_arrows, input_gen, test_name="apply")
