"""Tests propagation of known ports."""
from reverseflow.invert import invert
from test_arrows import test_xyplusx, test_twoxyplusx, test_multicomb


def invert_test():
    """Verifies the output of mark()."""
    arrow = test_xyplusx()
    marked_inports, marked_outports = invert(arrow)

invert_test()
