"""Major classes for Arrow data structures"""


class Arrow:
    """Abstract arrow class"""

    def __init__(self):
        pass

    def invert(self):
        pass

    def num_in_ports(self):
        return len(self.in_ports)

    def num_out_ports(self):
        return len(self.out_ports)
