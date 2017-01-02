"""Immutable port classes, should be identified by arrow, index, and type."""

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
        return "Port[%s@%s]" % (self.arrow, self.index)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if (isinstance(self, InPort) and isinstance(other, InPort)) or (isinstance(self, OutPort) and isinstance(other, OutPort)):
            return ((self.arrow is other.arrow) and (self.index == other.index))
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())


class InPort(Port):
    """Input port"""
    def __str__(self):
        return "In%s" % super().__str__()

    def __repr__(self):
        return str(self)


class OutPort(Port):
    """Output port"""
    def __str__(self):
        return "Out%s" % super().__str__()

    def __repr__(self):
        return str(self)

class ParamPort(InPort):
    """Parametric port"""
    def __str__(self):
        return "Param%s" % super().__str__()

    def __repr__(self):
        return str(self)

class ErrorPort(OutPort):
    """Error Port"""
    def __str__(self):
        return "Error%s" % super().__str__()

    def __repr__(self):
        return str(self)
