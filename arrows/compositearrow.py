"""Compositions of Primitive Arrows"""
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
    """Composite arrow
    A composite arrow is a composition of SubArrows
    """

    def has_in_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port) for port in self.get_in_ports()))

    def has_out_port_type(self, PortType) -> bool:
        return any((isinstance(PortType, port) for port in self.get_out_ports()))

    def neigh_in_ports(self, out_port: OutPort) -> Sequence[InPort]:
        return self.edges.fwd(out_port)

    def neigh_out_ports(self, in_port: InPort) -> Sequence[OutPort]:
        return self.edges.inv(in_port)

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
        assert self.is_wired_correctly(), "The arrow is wired incorrectly"

    def get_in_ports(self) -> List[InPort]:
        """Get InPorts of an Arrow
        Returns:
            List of InPorts"""
        return [port for port in self.ports if isinstance(port, OutPort)]

    def get_out_ports(self) -> List[OutPort]:
        """Get OutPorts of an Arrow
        Returns:
            List of OutPorts"""
        return [port for port in self.ports if isinstance(port, OutPort)]

    # def get_ports(self) -> List[Port]:
    #     ports = set()
    #     for out_port, in_port in self.edges.items():
    #         if out_port.arrow is self:
    #             ports.add(out_port)
    #         if in_port.arrow is self:
    #             ports.add(in_port)
    #     sorted_ports = sorted(list(ports), key=lambda port: port.index)
    #     return sorted_ports

    def add_port(self, port_attributes=None) -> Port:
        """Add a port to the arrow"""
        idx = self.num_ports()
        port = Port(self, idx)
        self.ports.append(port)
        if port_attributes:
            self.port_attributes.append(port_attributes)
        else:
            self.port_attributes.append({})
        return port

    def get_ports(self) -> List[Port]:
        return self.ports

    def __init__(self,
                 edges: RelEdgeMap=None,
                 in_ports: Sequence[InPort]=None,
                 out_ports: Sequence[OutPort]=None,
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
        for in_port in in_ports:
            port = self.add_port()
            self.edges.add(port, in_port)
            # TODO Designate these ports as OutPorts (or whatever)

        for out_port in out_ports:
            port = self.add_port()
            self.edges.add(out_port, port)
            # TODO Designate these ports as InPorts

        n_ports = self.num_ports()
        if port_attributes:
            assert len(port_attributes) == n_ports
            self.port_attributes = port_attributes

        assert self.is_wired_correctly(), "The arrow is wired incorrectly"
        assert self.are_sub_arrows_parentless(), "subarrows must be parentless"
        # Make this arrow the parent of each sub_arrow
        for sub_arrow in self.get_sub_arrows():
            sub_arrow.parent = self

# Issues:
# Should Edges be between InPort and OutPorts or just Por
# - I say just Port
# Should we keep an explicit list of the in_ports and out_ports or dynamically generate it
## -- Dynamic means that we need to check that they are contiguous
## -- Slower
## Do I want to replace type with  In / Out attribute ?!?!?!?!?!


## Port attributes
## Right now I have it as just s set of things "Param, Error"
## Could have "InPort" and "OutPort" too
## the reason I wanted it at as an attribute as so that I could (1) not have this weird thing of an outport going to an inport
## its a type failure
## But effectively the same thing will still be true
## Btu also to be able to cast a port into a particular type for its use
## It doesn't feel satisfcatory.
## Essentially it doesnt make sense to talk about a port as being an out port or an in port because it is both
## At the SAME TIME, there is a polarity.  There are ports which take signal from the outside world and pass it to the inside world.
## and ports which take signals and pass them from the inside world and pass them to the outside world.
## What's valid OI -> OI, OI -> IO, IO -> IO, IO - OI.  Without concern for what the arrow is, all combinatiosn are possible
## But whats not possible si a:OI -> a IO for example. That is, taking the arrow into account, not all projections are valid
## What makes something invalid.


## Question is how to handle port attributes
## Could have a mapping from Ports to Attributes
## First question is whether In/Out is an attribute or part of the type
## Second is how to handle attributes in general

## JMFD
## An edge is between ports
## its not from outport to inport



## TODO:1
## Have port attributes be list
## What data structure, dict
## {"InOut": "InPort", "name": "Blergh", "param": True}

## Port attributes is a dict
## The benefits of this as opposed to having a type hierarchy is that
## 1. The type can be switched without breaking everything
## 2.
