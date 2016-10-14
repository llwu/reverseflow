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
