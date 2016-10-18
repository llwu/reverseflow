from typing import Set

from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.port import InPort, ParamPort


class ParametricArrow(CompositeArrow):
    """Parametric arrow"""

    def __init__(self):
        self.in_ports = []  # type: List[InPort]
        self.out_ports = []  # type: List[OutPort]
        self.param_ports = []  # type: List[ParamPort]
        self.edges = Bimap()  # type: EdgeMap

    def add_params(self, params: Set[InPort]) -> None:
        """
        Moves some ports from in_ports[] to param_ports[].
        """
        i = len(self.param_ports)
        for in_port in self.in_ports:
            if in_port in params:
                self.in_ports.remove(in_port)
                inner_port = self.edges.fwd(in_port)
                self.edges.remove(in_port, inner_port)
                param_port = ParamPort(self, i)
                self.param_ports.append(param_port)
                self.edges.add(param_port, inner_port)
                i += 1
