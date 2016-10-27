from reverseflow.arrows.port import OutPort, InPort
from reverseflow.arrows.primitive.math_arrows import MulArrow, AddArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.apply.symbolic import symbolic_apply


def test_xyplusx() -> CompositeArrow:
    """f(x,y) = x * y + x"""
    mul = MulArrow()
    add = AddArrow()
    dupl = DuplArrow()
    edges = Bimap()  # type: EdgeMap
    edges.add(dupl.out_ports[0], mul.in_ports[0])  # dupl -> mul
    edges.add(dupl.out_ports[1], add.in_ports[0])  # dupl -> add
    edges.add(mul.out_ports[0], add.in_ports[1])  # mul -> add
    d = CompositeArrow(in_ports=[dupl.in_ports[0], mul.in_ports[1]],
                       out_ports=[add.out_ports[0]], edges=edges)
    return d


def test_symbolic_apply() -> None:
    arrow = test_xyplusx()
    symbolic_map = symbolic_apply(arrow)
    import pdb; pdb.set_trace()
    


test_symbolic_apply()
