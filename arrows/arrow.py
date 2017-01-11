"""Major classes for Arrow data structures"""


class Arrow:
    """Abstract arrow class"""

    def __init__(self, name: str, parent=None) -> None:
        self.name = name
        self.parent = parent

    def num_in_ports(self) -> int:
        return len(self.get_in_ports())

    def num_out_ports(self) -> int:
        return len(self.get_out_ports())

    def is_primitive(self) -> bool:
        return False

    def is_composite(self) -> bool:
        return False

    def is_source(self) -> bool:
        return False

    def is_parametric(self) -> bool:
        return False

    def is_approximate(self) -> bool:
        return False

    def is_tf(self) -> bool:
        return False

    # def inner_in_ports(self):
    #     return self.get_in_ports()
    #
    # def inner_out_ports(self):
    #     return self.get_out_ports()

    def inner_error_ports(self):
        return []

    def get_sub_arrows(self):
        return []
