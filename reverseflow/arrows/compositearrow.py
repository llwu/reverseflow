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

    def get_sub_arrows(self) -> Set[Arrow]:
        """Return all the constituent arrows of composition"""
        arrows = set()
        for (out_port, in_port) in self.edges.items():
            if out_port.arrow is not self: arrows.add(out_port.arrow)
            if in_port.arrow is not self: arrows.add(in_port.arrow)

        return arrows

    def __init__(self):
        self.in_ports = []  # type: List[InPort]
        self.out_ports = []  # type: List[OutPort]
        self.edges = Bimap()  # type: Bimap[OutPort, InPort]

    def add_edges(self, edges: Bimap[OutPort, InPort]) -> None:
        """
        Add edges, check for correctness.
        """
        in_i = 0
        out_i = 0
        # TODO: Assertions
        # TODO: Assert There is be at least one edge from self.outport
        # TODO: Assert There must be at least one edge to self inports
        # TODO: Assert There must be no cycles
        # TODO: Assert Edges are bijective this is true if Bimap is correct AND
        #       if we have the correct notion of equality for Ports
        # TODO: Assert No dangling ports (must be contiguous, 0, 1, 2, .., n)
        self.edges = edges


    def get_boundary_outports(self) -> Set[OutPort]:
        """
        Get the boundary outports
        """
        out_ports = set()  # type: Set[OutPort]
        for (out_port, in_port) in self.edges.items():
            if out_port.arrow is self:
                out_ports.add(out_port)
        return out_ports

    def neigh_inport(self, out_port: OutPort) -> InPort:
        return self.edges.fwd(out_port)

    def neigh_outport(self, in_port: InPort) -> OutPort:
        return self.edges.inv(in_port)
