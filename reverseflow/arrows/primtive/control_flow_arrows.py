"""These are arrows for control flow of input"""

from reverseflow.arrows.arrows import PrimitiveArrow


class DuplArrow(PrimitiveArrow):
    """
    Duplicate input
    f(x) = (x, x, ..., x)
    """

    def __init__(self, n_duplications=2) -> None:
        self.in_ports = [InPort(self, 0)]
        self.out_ports = [OutPort(self, i) for i in range(n_duplications)]

    def invert(self):
        pass


class IdentityArrow(PrimitiveArrow):
    """
    Identity input
    f(x) = x
    """

    def __init__(self) -> None:
        self.in_ports = [InPort(self, 0)]
        self.out_ports = [OutPort(self, 0)]

    def invert(self):
        pass
