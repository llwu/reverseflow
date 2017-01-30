from typing import List, Dict
from arrows.arrow import Arrow
from arrows.port import Port, InPort, OutPort
from arrows.port_attributes import make_out_port

class SourceArrow(Arrow):
    """
    A source arrow is a constant, it takes no input and has one output
    """

    def get_ports(self):
        return self.ports

    def get_out_ports(self):
        return self.ports

    def get_in_ports(self) -> List[InPort]:
        return []

    def __deepcopy__(self, memo):
        new_name = None
        if self.name != None:
            new_name = self.name + "_copy"
        return SourceArrow(value=self.value, name=new_name)

    def __init__(self, value, name: str = None) -> None:
        super().__init__(name=name)
        self.ports = [Port(self, 0)]
        self.port_attributes = [{}]
        make_out_port(self.ports[0])
        self.value = value

    def is_source(self):
        return True

    def eval(self, ptv: Dict):
        o = self.get_out_ports()
        assert len(o) == 1
        ptv[o[0]] = self.value
        return ptv
