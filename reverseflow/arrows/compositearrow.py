from typing import Set
from reverseflow.arrows.arrow import Arrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.port import InPort, OutPort


class CompositeArrow(Arrow):
    """
    Composite arrow
    A composite arrow is a composition of simpler arrows, which may be either
    primtive arrows or themselves compositions.
    """

    def get_arrows(self) -> Set[Arrow]:
        """Return all the constituent arrows of composition"""
        arrows = set()
        for (out_port, in_port) in self.edges.items():
            arrows.add(out_port.arrow)
            arrows.add(in_port.arrow)

        self.arrows = arrows
        return arrows

    def __init__(self, edges: Bimap[OutPort, InPort]) -> None:
        """
        init this bad boy
        """
        self.edges = edges
        self.in_ports = []  # type: List[InPort]
        self.out_ports = []  # type: List[OutPort]
        in_i = 0
        out_i = 0

        arrows = self.get_arrows()
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
