"""Inverse Ops for Arrows

Parametric inverses are not unique
"""
def inverse_arrow() -> Arrow:
    # consider having theta be something other than an InPort
    z_minus_theta = SubArrow()
    dupl_theta = DuplArrow()
    edges = Bimap()  # type: Bimap[OutPort, InPort]
    edges.add(dupl_theta.out_ports[0], z_minus_theta.in_ports[1])
    return CompositeArrow([z_minus_theta, dupl_theta], edges)

register(AddArrow, inverse_arrow)


class InvArrow(CompositeArrow):
    def is_primitive() -> bool:
        return False

    def __init__(self):
        self.name = "InvAdd"

    def inverse_of(self):
        return AddArrow

    def procedure(self):
        # consider having theta be something other than an InPort
        z_minus_theta = SubArrow()
        dupl_theta = DuplArrow()
        edges = Bimap()  # type: Bimap[OutPort, InPort]
        edges.add(dupl_theta.out_ports[0], z_minus_theta.in_ports[1])
        return CompositeArrow([z_minus_theta, dupl_theta], edges)
