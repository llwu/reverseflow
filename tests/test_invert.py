"""Tests propagation of known ports."""
from reverseflow.invert import invert
from test_arrows import test_xyplusx_flat
from util import random_arrow_test

def invert_test() -> None:
    """Verifies the output of invert()."""
    arrow = test_xyplusx_flat()
    invert(arrow)


random_arrow_test(invert, "invert")

"""
from test_arrows import test_random_composite
from reverseflow.util.viz import show_tensorboard
a = test_random_composite()
show_tensorboard(a)
b = invert(a)
show_tensorboard(b)
import pdb; pdb.set_trace()
"""
