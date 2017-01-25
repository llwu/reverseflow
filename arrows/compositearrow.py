"""Compositions of Primitive Arrows"""
from typing import Set, Sequence, List
from copy import deepcopy, copy

from arrows import Arrow
from arrows.port import Port
from arrows.port_attributes import is_in_port, is_out_port
from reverseflow.util.mapping import Bimap, Relation

EdgeMap = Bimap[Port, Port]
RelEdgeMap = Relation[Port, Port]


def is_exposed(port: Port, context: "CompositeArrow") -> bool:
    """Is this Port exposed within this arrow, i.e. can it be connected
       to an edge."""
    return context == port.arrow or port.arrow.parent == context


def is_projecting(port: Port, context: "CompositeArrow") -> bool:
    """Is this port projecting in this context"""
    if port.arrow == context:
        return is_in_port(port)
    elif port.arrow in context.get_sub_arrows():
        return is_out_port(port)
    else:
        assert False, "Port %s is not in context %s" % (port, context)


def is_receiving(port: Port, context: "CompositeArrow") -> bool:
    """A port is receiving (in some context) if it is not projecting."""
    return not is_projecting(port, context)

# from arrows.primitive.controlflow import DuplArrow

class CompositeArrow(Arrow):
    """Composite arrow
    A composite arrow is a composition of SubArrows
    """
    def duplify(self) -> None:
        for out_port in self.edges.keys():
            in_ports = self.neigh_in_ports(out_port)
            if len(in_ports) > 1:
                dupl = DuplArrow(n_duplications=len(in_ports))
                # add edge from to dupl and remove all other edges
                self.add_edge(out_port, dupl.get_in_ports()[0])
                for i, neigh_port in enumerate(in_ports):
                    self.remove_edge(out_port, neigh_port)
                    self.add_edge(dupl.get_out_ports()[i], neigh_port)
                assert len(self.neigh_in_ports(out_port)) == 1

    def has_in_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port) for port in self.get_in_ports()))

    def has_out_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port) for port in self.get_out_ports()))

    def neigh_in_ports(self, out_port: Port) -> Sequence[Port]:
        return self.edges.fwd(out_port)

    def neigh_out_ports(self, in_port: Port) -> Sequence[Port]:
        return self.edges.inv(in_port)

    def neigh_ports(self, port: Port) -> Sequence[Port]:
        if port in self.edges:
            return self.edges.fwd(port)
        elif port in self.edges.right_to_left:
            return self.edges.inv(port)
        else:
            return []

    def get_all_arrows(self) -> Set[Arrow]:
        """Return all arrows including self"""
        arrows = set()
        for (out_port, in_port) in self.edges.items():
            arrows.add(out_port.arrow)
            arrows.add(in_port.arrow)
        if self not in arrows:
            arrows.add(self)

        return arrows

    def get_sub_arrows(self) -> Set[Arrow]:
        """Return all the constituent arrows of composition"""
        arrows = self.get_all_arrows()
        if self in arrows:
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
        #     for out_port in sub_arrow.get_out_ports():
        #         out_port_fan[out_port] = 0
        #     for in_port in sub_arrow.get_in_ports():
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
        self.port_attributes[index].add(attribute)

    def add_in_port_attribute(self, index: int, attribute: str):
        assert index < self.num_in_ports()
        self.add_port_attribute(index, attribute)

    def add_out_port_attribute(self, index: int, attribute: str):
        assert self.num_in_ports() <= index + self.num_in_ports() < self.num_ports()
        self.add_port_attribute(index, attribute)

    def __str__(self):
        return "Comp_%s_%s" % (self.name, hex(id(self)))

    def add_edge(self, left: Port, right: Port):
        """Add an edge to the composite arrow
        Args:
            left: Projecting Port
            right: receiving Port
        """
        assert left.arrow.parent is self or left.arrow.parent is None
        assert right.arrow.parent is self or right.arrow.parent is None
        left.arrow.parent = self
        right.arrow.parent = self
        self.edges.add(left, right)

    def remove_edge(self, left: Port, right: Port):
        """Remove an edge from the composite arrow
        Args:
            left: Projecting Port
            right: receiving Port
        """
        self.edges.remove(left, right)


    def add_port(self, port_attributes=None) -> Port:
        """Add a port to the arrow"""
        idx = self.num_ports()
        port = Port(self, idx)
        self.ports.append(port)
        if port_attributes is not None:
            self.port_attributes.append(port_attributes)
        else:
            self.port_attributes.append({})
        return port

    def get_ports(self) -> List[Port]:
        return self.ports

    def __deepcopy__(self, memo):
        new_arrow = copy(self)
        new_edges = Relation()
        new_name = None
        if self.name != None:
            new_name = self.name + "_copy"
        new_arrow.name = new_name
        new_arrow.parent = None

        new_port_attributes = [] # type: List[Dict]
        for attribute in self.port_attributes:
            new_port_attributes.append(deepcopy(attribute))
        new_arrow.port_attributes = new_port_attributes

        new_arrow.ports = [Port(new_arrow, i) for i in range(self.num_ports())]

        copies = {self: new_arrow}
        for sub_arrow in self.get_sub_arrows():
            copies[sub_arrow] = deepcopy(sub_arrow)
            assert copies[sub_arrow].parent == None, "sub_arrow can't have a parent yet"

        for out_port, in_port in self.edges.items():
            assert out_port.arrow in copies, "sub_arrow not copied"
            assert in_port.arrow in copies, "sub_arrow not copied"
            new_out = copies[out_port.arrow].get_ports()[out_port.index]
            new_in = copies[in_port.arrow].get_ports()[in_port.index]
            assert new_out.arrow == copies[out_port.arrow], "port not copied properly"
            assert new_in.arrow == copies[in_port.arrow], "port not copied properly"
            new_edges.add(new_out, new_in)

        new_arrow.edges = new_edges
        assert new_arrow.num_ports() == len(new_arrow.port_attributes), "incorrect number of attributes"
        assert new_arrow.is_wired_correctly(), "arrow copy is wired incorrectly"
        assert new_arrow.are_sub_arrows_parentless(), "sub_arrows can't have a parent yet"
        for sub_arrow in new_arrow.get_sub_arrows():
            sub_arrow.parent = new_arrow

        return new_arrow


    def __init__(self,
                 edges: RelEdgeMap=None,
                 in_ports: Sequence[Port]=None,
                 out_ports: Sequence[Port]=None,
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

        Port indices are continugous 0 ... n_ports, but types are not.
        """
        super().__init__(name=name)

        self.edges = Relation()
        self.ports = []
        self.port_attributes = []

        if edges:
            for out_port, in_port in edges.items():
                self.edges.add(out_port, in_port)

        # create new edges for composition
        if in_ports:
            for in_port in in_ports:
                port = self.add_port({"InOut": "InPort"})
                self.edges.add(port, in_port)

        if out_ports:
            for out_port in out_ports:
                port = self.add_port({"InOut": "OutPort"})
                self.edges.add(out_port, port)

        n_ports = self.num_ports()
        if port_attributes:
            assert len(port_attributes) == n_ports
            self.port_attributes = port_attributes

        assert self.is_wired_correctly(), "The arrow is wired incorrectly"
        assert self.are_sub_arrows_parentless(), "subarrows must be parentless"
        # Make this arrow the parent of each sub_arrow
        for sub_arrow in self.get_sub_arrows():
            sub_arrow.parent = self
