"""Tests propagation of known ports."""

from reverseflow.arrows.marking import mark
from .test_arrows import test_multicomb


def marking_test():
    """Verifies the output of mark()."""
    arrow = test_multicomb()

marking_test()
