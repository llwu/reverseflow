from arrows.arrow import Arrow
from arrows.port import Port, InPort, OutPort
from arrows.port_attributes import make_in_port, make_out_port
from typing import Dict, List, MutableMapping, Set
from sympy import Expr, Rel
from copy import copy

class PrimitiveArrow(Arrow):
    """Primitive arrow"""
    def is_primitive(self) -> bool:
        return True

    def get_ports(self) -> List[Port]:
        return self.ports

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        return []

    def __deepcopy__(self, memo):
        new_arrow = copy(self)
        if self.name != None:
            new_arrow.name = self.name + "_copy"
        new_arrow.parent = None
        n_ports = self.n_in_ports + self.n_out_ports
        assert new_arrow.n_in_ports + new_arrow.n_out_ports == n_ports, "incorrect copy"
        new_arrow.ports = [Port(new_arrow, i) for i in range(n_ports)]
        return new_arrow

    def __init__(self, n_in_ports: int, n_out_ports: int, name: str) -> None:
        super().__init__(name=name)
        n_ports = n_in_ports + n_out_ports
        self.n_in_ports = n_in_ports
        self.n_out_ports = n_out_ports
        self.ports = [Port(self, i) for i in range(n_ports)]
        self.port_attributes = [{} for i in range(n_ports)]
        for i in range(n_in_ports):
            make_in_port(self.ports[i])
        for i in range(n_in_ports, n_ports):
            make_out_port(self.ports[i])
