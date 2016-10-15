from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.primitivearrow import PrimitiveArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.arrows.port import InPort, OutPort


class AddArrow(PrimitiveArrow):
    """Addition op"""

    def __init__(self):
        self.name = 'Add'
        self.in_ports = [InPort(self, 0), InPort(self, 1)]
        self.out_ports = [OutPort(self, 0)]

    # def invert(self) -> Arrow:
    #     # consider having theta be something other than an InPort
    #     z_minus_theta = SubArrow()
    #     dupl_theta = DuplArrow()
    #     edges = Bimap()  # type: Bimap[OutPort, InPort]
    #     edges.add(dupl_theta.out_ports[0], z_minus_theta.in_ports[1])
    #     return CompositeArrow([z_minus_theta, dupl_theta], edges)


class SubArrow(PrimitiveArrow):
    """Subtraction op. Out[1] = In[0] - In[1]"""

    def __init__(self):
        self.name = 'Sub'
        self.in_ports = [InPort(self, 0), InPort(self, 1)]
        self.out_ports = [OutPort(self, 0)]

    def invert(self):
        pass


class MulArrow(PrimitiveArrow):
    """Multiplication op"""

    def __init__(self):
        self.name = 'Mul'
        self.in_ports = [InPort(self, 0), InPort(self, 1)]
        self.out_ports = [OutPort(self, 0)]

    def invert(self):
        pass
