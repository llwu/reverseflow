from typing import Set
from reverseflow.arrows.arrow import Arrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.port import InPort, OutPort


class CompositeArrow(Arrow):
    """
    Composite arrow
    A composite arrow is a composition of arrows, which may be either
    primtive arrows or themselves compositions.
    """
    def is_primitive(self) -> bool:
        return False

    def is_parametric(self) -> bool:
        return len(self.param_inport) > 0

    def get_sub_arrows(self) -> Set[Arrow]:
        """Return all the constituent arrows of composition"""
        arrows = set()
        for (out_port, in_port) in self.edges.items():
            arrows.add(out_port.arrow)
            arrows.add(in_port.arrow)

        return arrows

    def __init__(self, edges: Bimap[OutPort, InPort]) -> None:
        """
        init this bad boy
        """
        self.edges = edges
        self.in_ports = []  # type: List[InPort]
        self.out_ports = []  # type: List[OutPort]
        self.param_inport = [] # type: List[ParamInPort]
        in_i = 0
        out_i = 0

        ## FIXME: Right now the arrows are connected to the inner ports
        ##        arbitrarily.  This makes it difficult to create composite
        ##        arrows while knowing how they will be wired up.

        arrows = self.get_sub_arrows()
        for arrow in arrows:
            for in_port in arrow.in_ports:
                if in_port not in edges.right_to_left:
                    boundary_outport = OutPort(self, out_i)
                    out_i += 1
                    self.out_ports.append(boundary_outport)
                    self.edges.add(boundary_outport, in_port)
            for out_port in arrow.out_ports:
                if out_port not in edges.left_to_right:
                    boundary_inport = InPort(self, in_i)
                    in_i += 1
                    self.in_ports.append(boundary_inport)
                    self.edges.add(out_port, boundary_inport)

    def get_boundary_outports(self) -> Set[OutPort]:
        """
        Get the boundary outports
        """
        out_ports = set()  # type: Set[OutPort]
        for (out_port, in_port) in self.edges.items():
            if out_port.arrow is self:
                out_ports.add(out_port)

        return out_ports
