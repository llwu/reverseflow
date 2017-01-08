from test_arrows import test_xyplusx_flat
from arrows.apply.shapes import propagate_shapes

def test_shapes():
    arrow = test_xyplusx_flat()
    propagate_shapes(arrow, [(1,2,3), (1,2,3)])
