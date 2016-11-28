from reverseflow.arrows.port import OutPort, InPort
from reverseflow.arrows.primitive.math_arrows import MulArrow, AddArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.apply.symbolic import symbolic_apply
from tests.test_arrows import test_xyplusx_flat
from reverseflow.invert import invert


def test_symbolic_apply() -> None:
    """inverse of f(x, y) = x * y + x"""
    arrow = test_xyplusx_flat()
    inv_arrow = invert(arrow)
    print(inv_arrow.is_parametric())
    (symbolic_map, constraints) = symbolic_apply(inv_arrow)
    print(constraints)
    print(symbolic_map)
    # import pdb; pdb.set_trace()

test_symbolic_apply()
