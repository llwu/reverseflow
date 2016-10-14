from reverseflow.arrows.arrow import Arrow


class Port():
    """
    Port

    An entry or exit to an Arrow, analogous to argument position of multivariate
    function.

    A port is uniquely determined by the arrow it belongs to and a pin.

    On the boundary of a composite arrow, ports are simultaneously inports
    (since they take input from outside world) and outputs (since inside they
    project outward to
    """

    def __init__(self, arrow: Arrow, index: int) -> None:
        self.arrow = arrow
        self.index = index


class InPort(Port):
    """Input port"""
    pass


class OutPort(Port):
    """Output port"""
    pass
