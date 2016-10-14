from typing import List

class Arrow:
    """Abstract arrow class"""

    def __init__(self):
        pass

    def get_in_ports(self):
        return self.in_ports

    def get_out_ports(self):
        return self.out_ports

class PrimitiveArrow(Arrow):
    """Primitive arrow"""

    def __init__(self):
        pass

class Bimap:
    """Bidirectional map"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def add(self, left, right):
        self.left_to_right[left] = right
        self.right_to_left[right] = left

class CompositeArrow(Arrow):
    """Composite arrow"""

    def __init__(self, arrows: List[Arrow], edges: Bimap):
        self.arrows = arrows
        self.edges = edges
        self.in_ports = []
        self.out_ports = []
        in_i = 0
        out_i = 0
        for arrow in arrows:
            for in_port in arrow.get_in_ports():
                if in_port not in edges.left_to_right:
                    new_port = InPort(self, in_i)
                    in_i += 1
                    self.in_ports.append(new_port)
                    self.edges.add(new_port, in_port)
            for out_port in arrow.get_out_ports():
                if out_port not in edges.right_to_left:
                    new_port = OutPort(self, out_i)
                    out_i += 1
                    self.out_ports.append(new_port)
                    self.edges.add(out_port, new_port)
        # create ports

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

class Port():
    """Port"""

    def __init__(self, arrow, index):
        self.arrow = arrow
        self.index = index

class InPort(Port):
    """Input port"""
    pass

class OutPort(Port):
    """Output port"""
    pass

a = MulArrow()
b = AddArrow()
c = DuplArrow()
edges = Bimap()
# change the rest
edges.add(c.get_out_ports()[0], a.get_in_ports()[0]) # dupl -> mul
edges.add(c.get_out_ports()[1], b.get_in_ports()[0]) # dupl -> add
edges.add(a.get_out_ports()[0], b.get_in_ports()[1]) # mul -> add
d = CompositeArrow([a, b, c], edges)
import pdb; pdb.set_trace()
