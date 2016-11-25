"""Tests propagation of known ports."""
from reverseflow.invert import invert
from test_arrows import test_xyplusx_flat, test_twoxyplusx, test_multicomb


def invert_test():
    """Verifies the output of mark()."""
    arrow = test_xyplusx_flat()
    inv_arrow = invert(arrow)

invert_test()
