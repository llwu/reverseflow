"""Tests propagation of known ports."""
from reverseflow.invert import invert
from test_arrows import test_xyplusx_flat, test_twoxyplusx, test_random_composite
from util import random_arrow_test

def invert_test() -> None:
    """Verifies the output of mark()."""
    arrow = test_xyplusx_flat()
    invert(arrow)


random_arrow_test(invert, "invert")
