from typing import Set, List
from reverseflow.arrows.arrow import Arrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.port import InPort, OutPort

EdgeMap = Bimap[OutPort, InPort]

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
            arrows.add(out_port.arrow)
            arrows.add(in_port.arrow)

        return arrows

    def __init__(self, in_ports: List[InPort], out_ports: List[OutPort],
                 edges: EdgeMap) -> None:
        # TODO: Assertions
        # TODO: Assert There is be at least one edge from self.outport
        # TODO: Assert There must be at least one edge to self inports
        # TODO: Assert There must be no cycles
        # TODO: Assert Edges are bijective this is true if Bimap is correct AND
        #       if we have the correct notion of equality for Ports
        # TODO: Assert No dangling ports (must be contiguous, 0, 1, 2, .., n)
        self.in_ports = in_ports  # type: List[InPort]
        self.out_ports = out_ports  # type: List[OutPort]
        self.edges = edges

    def neigh_in_port(self, out_port: OutPort) -> InPort:
        return self.edges.fwd(out_port)

    def neigh_out_port(self, in_port: InPort) -> OutPort:
        return self.edges.inv(in_port)
