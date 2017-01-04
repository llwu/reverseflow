"""Tests propagation of known ports."""
from arrows.marking import mark
# from util import random_arrow_test


def marking_test() -> None:
    """Verifies the output of mark()."""
    arrow = test_multicomb()
    marked_inports, marked_outports = mark(arrow, set(arrow.in_ports[:-1]))
    assert len(marked_inports) == 8
    assert len(marked_outports) == 3


def marking_visual_test() -> None:
    from test_arrows import test_random_composite, test_multicomb
    from reverseflow.util.viz import show_tensorboard
    a = test_multicomb()
    show_tensorboard(a)
    b, c = mark(a, set(a.inner_in_ports()[:-1]))
    import pdb; pdb.set_trace()


#random_arrow_test(lambda arrow: mark(arrow, set(arrow.inner_in_ports()[:-1])), "mark")
# marking_visual_test()
