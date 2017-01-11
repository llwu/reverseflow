from typing import List
from arrows.arrow import Arrow
from arrows.port import Port, InPort, OutPort


class SourceArrow(Arrow):
    """
    A source arrow is a constant, it takes no input and has one output
    """

    def get_out_ports(self):
        return self.out_ports

    def get_in_ports(self) -> List[InPort]:
        return []

    def get_ports(self) -> List[Port]:
        return self.get_out_ports()

    def __init__(self, value, name: str = None) -> None:
        super().__init__(name=name)
        self.out_ports = [OutPort(self, 0)]
        self.value = value

    def is_source(self):
        return True
