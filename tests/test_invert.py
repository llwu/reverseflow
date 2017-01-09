"""Tests propagation of known ports."""
from reverseflow.invert import invert
from test_arrows import test_xyplusx_flat
from test_arrows import all_test_arrow_gens
from totality_test import totality_test


def invert_visual_test() -> None:
    from test_arrows import test_random_composite
    from reverseflow.util.viz import show_tensorboard
    a = test_random_composite()
    show_tensorboard(a)
    b = invert(a)
    show_tensorboard(b)

#random_arrow_test(invert, "invert")
# invert_visual_test()

def test_symbolic_apply():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(invert, all_test_arrows)
