"""Tests propagation of known ports."""

from reverseflow.arrows.marking import mark
from .test_arrows import test_multicomb


def marking_test():
    """Verifies the output of mark()."""
    arrow = test_multicomb()
    marked_inports, marked_outports = mark(arrow, set(arrow.in_ports[:-1]))
    assert len(marked_inports) == 8
    assert len(marked_outports) == 3

marking_test()
