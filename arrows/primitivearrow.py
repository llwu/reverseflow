from arrows.arrow import Arrow
from arrows.port import InPort, OutPort
from typing import Dict, List, MutableMapping, Set
from sympy import Expr, Rel

class PrimitiveArrow(Arrow):
    """Primitive arrow"""
    def is_primitive(self) -> bool:
        return True

    def get_in_ports(self) -> List[InPort]:
        return self.in_ports

    def get_out_ports(self) -> List[OutPort]:
        return self.out_ports

    def gen_constraints(self, input_expr: MutableMapping[int, Expr], output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
        return []

    def __init__(self, n_in_ports: int, n_out_ports: int, name: str) -> None:
        super().__init__(name=name)
        self.n_in_ports = n_in_ports
        self.n_out_ports = n_out_ports
        self.in_ports = [InPort(self, i) for i in range(n_in_ports)]
        self.out_ports = [OutPort(self, i) for i in range(n_out_ports)]
