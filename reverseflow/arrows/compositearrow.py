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

    def is_composite(self) -> bool:
        return True

    def get_sub_arrows(self) -> Set[Arrow]:
        """Return all the constituent arrows of composition"""
        arrows = set()
        for (out_port, in_port) in self.edges.items():
            arrows.add(out_port.arrow)
            arrows.add(in_port.arrow)

        return arrows

    def __init__(self,
                 edges: EdgeMap,
                 in_ports: List[InPort],
                 out_ports: List[OutPort],
                 name: str = None) -> None:
        super().__init__(name=name)
        # TODO: Assertions
        assert len(in_ports) > 0, "Composite Arrow must have in ports"
        assert len(out_ports) > 0, "Composite Arrow must have in ports"
        self.edges = edges
        arrows = self.get_sub_arrows()
        for in_port in in_ports:
            assert in_port.arrow in arrows, "Designated in_port not in edges"
            assert in_port not in edges.values(), "in_port must be unconnected"

        for out_port in out_ports:
            assert out_port.arrow in arrows, "Designated in_port not in edges"
            assert out_port not in edges.keys(), "out_port must be unconnected"

        # TODO: Assert There must be no cycles
        # TODO: Assert Every inport must be on end of edge or be in in_ports
        # TODO: Assert Every outport must be on start of edge or in out_ports
        self.in_ports = in_ports  # type: List[InPort]
        self.out_ports = out_ports  # type: List[OutPort]

    def neigh_in_port(self, out_port: OutPort) -> InPort:
        return self.edges.fwd(out_port)

    def neigh_out_port(self, in_port: InPort) -> OutPort:
        return self.edges.inv(in_port)
