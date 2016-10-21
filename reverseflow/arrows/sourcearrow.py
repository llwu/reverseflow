from typing import List
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.port import InPort, OutPort


class SourceArrow(Arrow):
    """
    A source arrow is a constant, it takes no input and has one output
    """

    def __init__(self, value) -> None:
        self.in_ports = []  # type: List[InPort]
        self.out_ports = [OutPort(self, 0)]
        self.value = value
