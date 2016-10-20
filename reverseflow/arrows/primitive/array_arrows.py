class GatherArrow(PrimitiveArrow):
    """Gather Arrow"""

    def __init__(self):
        self.name = 'Gather'
        self.in_ports = [InPort(self, 0), InPort(self, 1)]
        self.out_ports = [OutPort(self, 0)]
        # self.type = Type((ShapeType))
