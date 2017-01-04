from typing import Set, Sequence
from arrows.arrow import Arrow
from reverseflow.util.mapping import Bimap
from arrows.port import InPort, OutPort, ErrorPort, ParamPort

EdgeMap = Bimap[OutPort, InPort]

class CompositeArrow(Arrow):
    """
    Composite arrow
    A composite arrow is a composition of SubArrows
    """

    def has_in_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port), for port in self.in_ports))

    def has_out_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port), for port in self.out_ports))

    def is_wired_correctly(self) -> bool:
        """Is this composite arrow wired up correctly"""
        for sub_arrow in self.get_sub_arrows():
            for out_port in sub_arrow.out_ports:
                out_port_fan[out_port] = 0
                in_port_fan[in_port] = 0

        for out_port, in_port in edges.items():
            out_port_fan[out_port] += 1
            in_port_fan[in_port] += 1

        for in_port in in_ports:
            in_port_fan[in_port] += 1

        for out_port in out_ports:
            out_port_fan[out_port] += 1

        for out_port, fan in out_port_fan.items():
            assert fan > 0, "%s unused" % out_port

        for in_port, fan in in_port_fan.items():
            assert fan == 1, "%s has %s inp, expected 1" % (in_port, fan)

        return True

    def are_sub_arrows_parentless(self) -> bool:
        return all((arrow.parent == None for arrow in self.get_sub_arrows()))

    def __init__(self,
                 edges: EdgeMap,
                 in_ports: Sequence[InPort],
                 out_ports: Sequence[OutPort],
                 name: str=None,
                 parent=None) -> None:
        super().__init__(name=name)
        self.in_ports = [InPort(self, i) for i in range(len(in_ports))]
        self.out_ports = [OutPort(self, i) for i in range(len(out_ports))]
        self._inner_in_ports = in_ports  # type: List[InPort]
        self._inner_out_ports = out_ports  # type: List[OutPort]
        assert is_wired_correctly(), "The arrow is wired incorrectly"
        assert are_sub_arrows_parentless(), "Sub_arrows must be parentless"
        # Make this arrow the parent of each sub_arrow
        for i in sub_arrow in self.get_sub_arrows():
            sub_arrow.parent = self
