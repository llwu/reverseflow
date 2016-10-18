from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.port import InPort, OutPort
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.primitive.math_arrows import AddArrow


class InvAddArrow(CompositeArrow, ParametricArrow):
    def __init__(self):
        self.name = "InvAdd"

    def inverse_of() -> Arrow:
        return AddArrow

    def procedure(self):
        # consider having theta be something other than an InPort
        inv_add = CompositeArrow()
        edges = Bimap()  # type: EdgeMap
        theta = ParamOutPort(inv_add, 0)
        z = OutPort(inv_add, 0)
        z_minus_theta = InPort(inv_add, 0)
        theta_pass = InPort(inv_add, 1)

        dupl_theta = DuplArrow()
        sub = SubArrow()

        edges.add(theta, dupl_theta.in_ports[0])
        edges.add(dupl_theta.out_ports[0], theta_pass)
        edges.add(z, sub.in_ports[0])
        edges.add(dupl_theta.out_ports[1], sub.in_ports[1])

        inv_add.add_edges(edges)
        return inv_add
