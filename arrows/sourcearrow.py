from typing import List
from arrows.arrow import Arrow
from arrows.port import InPort, OutPort


class SourceArrow(Arrow):
    """
    A source arrow is a constant, it takes no input and has one output
    """

    def __init__(self, value, name: str = None) -> None:
        super().__init__(name=name)
        self.in_ports = []  # type: List[InPort]
        self.out_ports = [OutPort(self, 0)]
        self.value = value

    def is_source(self):
        return True

    def get_shape(self):
        v = self.value
        shape = []
        while hasattr(v, '__len__'):
            shape.append(len(v))
            v = v[0]
        return tuple(shape)
