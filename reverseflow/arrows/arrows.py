from typing import List, TypeVar, Generic


class Arrow:
    """Abstract arrow class"""

    def __init__(self):
        pass

    def get_in_ports(self):
        return self.in_ports

    def get_out_ports(self):
        return self.out_ports


class Port():
    """
    Port

    An entry or exit to an Arrow, analogous to argument position of multivariate
    function.

    A port is uniquely determined by the arrow it belongs to and a pin.

    On the boundary of a composite arrow, ports are simultaneously inports
    (since they take input from outside world) and outputs (since inside they
    project outward to
    """

    def __init__(self, arrow: Arrow, index: int) -> None:
        self.arrow = arrow
        self.index = index


class InPort(Port):
    """Input port"""
    pass


class OutPort(Port):
    """Output port"""
    pass


class PrimitiveArrow(Arrow):
    """Primitive arrow"""

    def __init__(self):
        pass

L = TypeVar('L')
R = TypeVar('R')


class Bimap(Generic[L, R]):
    """Bidirectional map"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def add(self, left, right):
        self.left_to_right[left] = right
        self.right_to_left[right] = left


class CompositeArrow(Arrow):
    """
    Composite arrow
    A composite arrow is a composition of simpler arrows, which may be either
    primtive arrows or themselves compositions.
    """

    def __init__(self, arrows: List[Arrow], edges: Bimap[OutPort, InPort]) -> None:
        self.arrows = arrows
        self.edges = edges
        self.in_ports = []  # type: List[InPort]
        self.out_ports = []  # type: List[OutPort]
        in_i = 0
        out_i = 0

        for arrow in arrows:
            for in_port in arrow.get_in_ports():
                if in_port not in edges.right_to_left:
                    boundary_outport = OutPort(self, out_i)
                    out_i += 1
                    self.out_ports.append(boundary_outport)
                    self.edges.add(boundary_outport, in_port)

        for arrow in arrows:
            for out_port in arrow.get_out_ports():
                if out_port not in edges.left_to_right:
                    boundary_inport = InPort(self, in_i)
                    in_i += 1
                    self.in_ports.append(boundary_inport)
                    self.edges.add(out_port, boundary_inport)


class AddArrow(PrimitiveArrow):
    """Addition op"""

    def __init__(self):
        self.in_ports = [InPort(self, 0), InPort(self, 1)]
        self.out_ports = [OutPort(self, 0)]

    def invert(self):
        pass


class MulArrow(PrimitiveArrow):
    """Multiplication op"""

    def __init__(self):
        self.in_ports = [InPort(self, 0), InPort(self, 1)]
        self.out_ports = [OutPort(self, 0)]

    def invert(self):
        pass


class DuplArrow(PrimitiveArrow):
    """Duplication op"""

    def __init__(self):
        self.in_ports = [InPort(self, 0)]
        self.out_ports = [OutPort(self, 0), OutPort(self, 1)]

    def invert(self):
        pass
