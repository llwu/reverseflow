from typing import Set, List
from arrows.arrow import Arrow
from reverseflow.util.mapping import Bimap
from arrows.port import InPort, OutPort, ErrorPort, ParamPort

from arrows.config import floatX
EdgeMap = Bimap[OutPort, InPort]


class CompositeArrow(Arrow):
    """
    Composite arrow
    A composite arrow is a composition of arrows, which may be either
    primtive arrows or themselves compositions.
    """

    def is_composite(self) -> bool:
        return True

    def is_primitive(self) -> bool:
        return False

    def has_in_port_type(self, InPortType) -> bool:
        return any((isinstance(port, InPortType) for port in self.in_ports))

    def has_out_port_type(self, OutPortType) -> bool:
        return any((isinstance(port, OutPortType) for port in self.out_ports))

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
                 name: str=None) -> None:
        super().__init__(name=name)
        assert len(in_ports) > 0, "Composite Arrow must have in ports"
        assert len(out_ports) > 0, "Composite Arrow must have out ports"
        self.edges = edges

        # Check edge direction is out_port to in_port
        for out_port, in_port in edges.items():
            assert isinstance(out_port, OutPort), "Expected OutPort got %s" % out_port
            assert isinstance(in_port, InPort), "Expected InPort got %s" % in_port

        # Check each in_port is not connected to another arrow (only counted to outside world)
        arrows = self.get_sub_arrows()
        for in_port in in_ports:
            assert in_port not in edges.values(), "in_port must be unconnected"  % in_port.arrow
            assert in_port.arrow in arrows, "InPort arrow (%s) not in composition" % in_port.arrow

        # Check each inport is connected to an actual arrow
        for out_port in out_ports:
            assert out_port.arrow in arrows, "OutPort arrow (%s) not in composition" % out_port.arrow
            assert out_port not in edges.keys(), "out_port must be unconnected"

        # vanilla_out_ports = [OutPort(o.arrow, o.index) for o in out_ports]
        # vanilla_in_ports = [InPort(i.arrow, i.index) for i in in_ports]

        vanilla_out_ports = out_ports
        vanilla_in_ports = in_ports

        for sub_arrow in self.get_sub_arrows():
            for in_port in sub_arrow.in_ports:
                assert (in_port in edges.values()) or (in_port in vanilla_in_ports)

            for out_port in sub_arrow.out_ports:
                assert (out_port in edges.keys()) or (out_port in vanilla_out_ports)

        # TODO: Assert There must be no cycles
        # TODO: Assert Every inport must be on end of edge or be in in_ports
        # TODO: Assert Every outport must be on start of edge or in out_ports
        self.in_ports = [InPort(self, i) for i in range(len(in_ports))]
        # TODO: propagate paramport-ness
        self.out_ports = [OutPort(self, i) for i in range(len(out_ports))]
        self._inner_in_ports = in_ports  # type: List[InPort]
        self._inner_out_ports = out_ports  # type: List[OutPort]

    def neigh_in_port(self, out_port: OutPort) -> InPort:
        return self.edges.fwd(out_port)

    def neigh_out_port(self, in_port: InPort) -> OutPort:
        return self.edges.inv(in_port)

    def inner_in_ports(self) -> List[InPort]:
        return self._inner_in_ports

    def inner_out_ports(self) -> List[OutPort]:
        return self._inner_out_ports

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


def is_parametric(comp_arrow: CompositeArrow) -> bool:
    return comp_arrow.has_in_port_type(ParamPort)


def is_approximate(comp_arrow: CompositeArrow) -> bool:
    return comp_arrow.has_out_port_type(ErrorPort)


def unparam_all(comp_arrow: CompositeArrow):
    """
    Convert all parametric ports to in_ports
    """
    for i in range(comp_arrow.n_in_ports):
        comp_arrow.change_in_port_type(ParamPort, i)
