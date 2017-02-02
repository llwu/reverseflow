from typing import List, Dict
from arrows.arrow import Arrow
from arrows.port import Port, InPort, OutPort
from arrows.port_attributes import make_out_port
from arrows.apply.shapes import *

class SourceArrow(Arrow):
    """
    A source arrow is a constant, it takes no input and has one output
    """

    def ports(self):
        return self._ports

    def out_ports(self):
        return self._ports

    def in_ports(self) -> List[InPort]:
        return []

    def __deepcopy__(self, memo):
        new_name = None
        if self.name != None:
            new_name = self.name + "_copy"
        return SourceArrow(value=self.value, name=new_name)

    def __init__(self, value, name: str = None) -> None:
        super().__init__(name=name)
        self._ports = [Port(self, 0)]
        self.port_attributes = [{}]
        make_out_port(self._ports[0])
        self.value = value

    def is_source(self):
        return True

    def eval(self, ptv: Dict):
        o = self.out_ports()
        assert len(o) == 1
        ptv[o[0]] = self.value
        return ptv

    def get_dispatches(self):
        return {source_predicate: source_dispatch}
