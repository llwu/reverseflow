from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.primitivearrow import PrimitiveArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.arrows.port import InPort, OutPort


class AddArrow(PrimitiveArrow):
    """Addition op"""

    def __init__(self):
        self.name = 'Add'
        # FIXME, this seems redundant! all we need ot know is hte number of
        # ports
        self.in_ports = [InPort(self, 0), InPort(self, 1)]
        self.out_ports = [OutPort(self, 0)]
        # self.type = Type((ShapeType))


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
