from typing import Set, Sequence, List
from arrows.arrow import Arrow
from reverseflow.util.mapping import Bimap, Relation
from arrows.port import InPort, OutPort, ErrorPort, ParamPort

EdgeMap = Bimap[OutPort, InPort]
RelEdgeMap = Relation[OutPort, InPort]


class CompositeArrow(Arrow):
    """
    Composite arrow
    A composite arrow is a composition of SubArrows
    """

    def has_in_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port) for port in self.in_ports))

    def has_out_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port) for port in self.out_ports))

    def neigh_in_ports(self, out_port: OutPort) -> Sequence[InPort]:
        return self.edges.fwd(out_port)

    def neigh_out_ports(self, in_port: InPort) -> Sequence[OutPort]:
        return self.edges.inv(in_port)


    def inner_in_ports(self) -> List[InPort]:
        return self._inner_in_ports

    def inner_out_ports(self) -> List[OutPort]:
        return self._inner_out_ports

    def get_sub_arrows(self) -> Set[Arrow]:
        """Return all the constituent arrows of composition"""
        arrows = set()
        for (out_port, in_port) in self.edges.items():
            arrows.add(out_port.arrow)
            arrows.add(in_port.arrow)

        return arrows

    def is_wired_correctly(self) -> bool:
        """Is this composite arrow wired up correctly"""
        out_port_fan = {}
        in_port_fan = {}
        for sub_arrow in self.get_sub_arrows():
            for out_port in sub_arrow.out_ports:
                out_port_fan[out_port] = 0

            for in_port in sub_arrow.in_ports:
                in_port_fan[in_port] = 0

        for out_port, in_port in self.edges.items():
            out_port_fan[out_port] += 1
            in_port_fan[in_port] += 1

        for in_port in self._inner_in_ports:
            in_port_fan[in_port] += 1

        for out_port in self._inner_out_ports:
            out_port_fan[out_port] += 1

        for out_port, fan in out_port_fan.items():
            if not fan > 0:
                print("%s unused" % out_port)
                return False

        for in_port, fan in in_port_fan.items():
            if not fan == 1:
                print("%s has %s inp, expected 1" % (in_port, fan))
                return False

        return True

    def are_sub_arrows_parentless(self) -> bool:
        return all((arrow.parent is None for arrow in self.get_sub_arrows()))

    def change_in_port_type(self, InPortType, index) -> "CompositeArrow":
        """
        Convert an in_port to a different in_port type.
        """
        # asert Porttype is a subclass of InPort
        port = self.in_ports[index]
        self.in_ports[index] = InPortType(port.arrow, port.index)

    def change_out_port_type(self, OutPortType, index) -> "CompositeArrow":
        """
        Convert an out_port to a different out_port type.
        """
        # TODO: assert
        port = self.out_ports[index]
        self.out_ports[index] = OutPortType(port.arrow, port.index)

    def __str__(self):
        return "Comp_%s_%s" % (self.name, hex(id(self)))

    def __init__(self,
                 edges: RelEdgeMap,
                 in_ports: Sequence[InPort],
                 out_ports: Sequence[OutPort],
                 name: str=None,
                 parent=None) -> None:
        super().__init__(name=name)
        self.edges = Relation()
        for out_port, in_port in edges.items():
            self.edges.add(out_port, in_port)
        self.in_ports = [InPort(self, i) for i in range(len(in_ports))]
        self.out_ports = [OutPort(self, i) for i in range(len(out_ports))]
        self._inner_in_ports = in_ports  # type: List[InPort]
        self._inner_out_ports = out_ports  # type: List[OutPort]
        assert self.is_wired_correctly(), "The arrow is wired incorrectly"
        assert self.are_sub_arrows_parentless(), "subarrows must be parentless"
        # Make this arrow the parent of each sub_arrow
        for sub_arrow in self.get_sub_arrows():
            sub_arrow.parent = self
