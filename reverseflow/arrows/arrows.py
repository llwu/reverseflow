"""Major classes for Arrow data structures"""


class Arrow:
    """Abstract arrow class"""

    def __init__(self):
        pass

    def get_in_ports(self):
        return self.in_ports

    def get_out_ports(self):
        return self.out_ports
