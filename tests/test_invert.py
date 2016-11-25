"""Tests propagation of known ports."""
from reverseflow.invert import invert
from test_arrows import test_xyplusx_flat, test_twoxyplusx, test_random_composite


def invert_test() -> None:
    """Verifies the output of mark()."""
    arrow = test_xyplusx_flat()
    invert(arrow)

    # rand_arrow = test_random_composite()
    # invert(rand_arrow)

invert_test()
