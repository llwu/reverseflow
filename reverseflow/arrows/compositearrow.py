class CompositeArrow(Arrow):
    """
    Composite arrow
    A composite arrow is a composition of simpler arrows, which may be either
    primtive arrows or themselves compositions.
    """

    def __init__(self, arrows: List[Arrow],
                 edges: Bimap[OutPort, InPort]) -> None:
        """
        init this bad boy
        """
        self.arrows = arrows
        self.edges = edges
        self.in_ports = []  # type: List[InPort]
        self.out_ports = []  # type: List[OutPort]
        in_i = 0
        out_i = 0

        # TODO: checks that each arrow in edges is an arrow in list
        # TODO: actually do we even need the list? maybe have (possibly cached)
        #       self.get_arrows()
        #

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
