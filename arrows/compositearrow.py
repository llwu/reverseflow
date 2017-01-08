from typing import Set, Sequence, List
from arrows import Arrow
from reverseflow.util.mapping import Bimap, Relation
from arrows.port import Port, InPort, OutPort

EdgeMap = Bimap[OutPort, InPort]
RelEdgeMap = Relation[Port, Port]


def is_exposed(port: Port, context: "CompositeArrow") -> bool:
    """Is this Port exposed within this arrow, i.e. can it be connected
       to an edge.
       """
    return context == port.arrow or port.arrow.parent == context


def is_projecting(port: Port, context: "CompositeArrow") -> bool:
    """Is this port projecting in this context"""
    # assert is_exposed(port, context), "Port not expsoed in this context"
    if port.arrow == context:
        return isinstance(port, InPort)
    else:
        return isinstance(port, OutPort)


def is_receiving(port: Port, context: "CompositeArrow") -> bool:
    # A port is receiving (in some context) if it is not projecting
    return not is_projecting(port, context)


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
    #
    # def inner_in_ports(self) -> List[InPort]:
    #     return self._inner_in_ports
    #
    # def inner_out_ports(self) -> List[OutPort]:
    #     return self._inner_out_ports
    def get_all_arrows(self) -> Set[Arrow]:
        """Return all arrows including self"""
        arrows = set()
        for (out_port, in_port) in self.edges.items():
            arrows.add(out_port.arrow)
            arrows.add(in_port.arrow)

        return arrows

    def get_sub_arrows(self) -> Set[Arrow]:
        """Return all the constituent arrows of composition"""
        arrows = self.get_all_arrows()
        arrows.remove(self)
        return arrows


    def is_wired_correctly(self) -> bool:
        """Is this composite arrow wired up correctly"""

        # Ensure that each left hand node is projecting and right receiving
        for left, right in self.edges.items():
            assert is_projecting(left, self), "port %s not projecting" % left
            assert is_receiving(right, self), "port %s not receiving" % right

        # Ensure no dangling ports
        out_port_fan = {}
        in_port_fan = {}
        # for sub_arrow in self.get_all_arrows():
        #     for out_port in sub_arrow.out_ports:
        #         out_port_fan[out_port] = 0
        #     for in_port in sub_arrow.in_ports:
        #         in_port_fan[in_port] = 0
        #     assert sub_arrow is not self
        #
        # for out_port, in_port in self.edges.items():
        #     out_port_fan[out_port] += 1
        #     in_port_fan[in_port] += 1
        #
        # for out_port, fan in out_port_fan.items():
        #     if not fan > 0:
        #         print("%s unused" % out_port)
        #         return False
        #
        # for in_port, fan in in_port_fan.items():
        #     if not fan == 1:
        #         print("%s has %s inp, expected 1" % (in_port, fan))
        #         return False
        #
        return True

    def are_sub_arrows_parentless(self) -> bool:
        return all((arrow.parent is None for arrow in self.get_sub_arrows()))

    def add_port_attribute(self, index: int, attribute: str):
        self.port_attributes[i].add(attribute)


    # def change_in_port_type(self, InPortType, index) -> "CompositeArrow":
    #     """
    #     Convert an in_port to a different in_port type.
    #     """
    #     # asert Porttype is a subclass of InPort
    #     port = self.in_ports[index]
    #     self.in_ports[index] = InPortType(port.arrow, port.index)
    #
    # def change_out_port_type(self, OutPortType, index) -> "CompositeArrow":
    #     """
    #     Convert an out_port to a different out_port type.
    #     """        # self._inner_in_ports = in_ports  # type: List[InPort]
    #     # self._inner_out_ports = out_ports  # type: List[OutPort]
    #     # TODO: assert
    #     port = self.out_ports[index]
    #     self.out_ports[index] = OutPortType(port.arrow, port.index)

    def __str__(self):
        return "Comp_%s_%s" % (self.name, hex(id(self)))

    def __init__(self,
                 edges: RelEdgeMap,
                 in_ports: Sequence[InPort],
                 out_ports: Sequence[OutPort],
                 name: str=None,
                 parent=None,
                 port_attributes=None) -> None:
        """
        Args:
            edges: wires mapping out_ports to in_ports
            in_ports: list of out_ports of sub_arrows to be linked to in_ports of composition
            out_ports: list of out_ports of sub_arrows to be linked to out_ports of composition
            name: name of composition
            parent: Composite arrow this arrow is embedded in
            port_attributes: tags for ports
        Returns:
            Composite Arrow
        """
        super().__init__(name=name)
        n_ports = len(in_ports) + len(out_ports)

        self.edges = Relation()
        for out_port, in_port in edges.items():
            self.edges.add(out_port, in_port)

        # InPorts are number 0, .., n, OutPorts n+1, ..., m
        out_port_0_id = len(in_ports)
        self.in_ports = [InPort(self, i) for i in range(out_port_0_id)]
        self.out_ports = [OutPort(self, i) for i in range(out_port_0_id,
                                                          out_port_0_id + len(out_ports))]
        if port_attributes is None:
            self.port_attributes = [set(), for i in range(n_ports)]
        else:
            assert len(port_attributes) == n_ports
        self.ports = self.in_ports + self.out_ports

        # create new edges for composition
        for i, in_port in enumerate(in_ports):
            self.edges.add(self.in_ports[i], in_port)

        for i, out_port in enumerate(out_ports):
            self.edges.add(out_port, self.out_ports[i])

        assert self.is_wired_correctly(), "The arrow is wired incorrectly"
        assert self.are_sub_arrows_parentless(), "subarrows must be parentless"
        # Make this arrow the parent of each sub_arrow
        for sub_arrow in self.get_sub_arrows():
            sub_arrow.parent = self
