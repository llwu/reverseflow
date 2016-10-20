from reverseflow.arrows.arrow import Arrow


class Port():
    """
    Port

    An entry or exit to an Arrow, analogous to argument position of multi-arg
    function.

    A port is uniquely determined by the arrow it belongs to and a index.

    On the boundary of a composite arrow, ports are simultaneously inports
    (since they take input from outside world) and outputs (since inside they
    project outward to
    """

    def __init__(self, arrow: Arrow, index: int) -> None:
        self.arrow = arrow
        self.index = index

    def __str__(self):
        return "Port[%s:%s]" % (self.arrow.name, self.index)


class InPort(Port):
    """Input port"""
    def __str__(self):
        return "In%s" % super().__str__()


class OutPort(Port):
    """Output port"""
    def __str__(self):
        return "Out%s" % super().__str__()


class ParamPort(Port):
    """Parametric port"""
    def __str__(self):
        return "Param%s" % super().__str__()


class ErrorPort(Port):
    """Error Port"""
    def __str__(self):
        return "Error%s" % super().__str__()
