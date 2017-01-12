"""Immutable port classes, should be identified by arrow, index, and type."""
from arrows.arrow import Arrow


class Port():
    """
    Port

    An entry or exit to an Arrow, analogous to argument position of multi-arg
    function.

    A port is uniquely determined by the arrow it belongs to and a index.
    """

    def __init__(self, arrow: Arrow, index: int) -> None:
        self.arrow = arrow
        self.index = index

    def __str__(self):
        return "Port[%s@%s]" % (self.arrow, self.index)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.arrow == other.arrow and self.index == other.index

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.arrow) + hash(self.index)


class InPort(Port):
    """Input port
    Transfers data from the outside arrow to 'In' side"""
    def __str__(self):
        return "In%s" % super().__str__()

    def __repr__(self):
        return str(self)


class OutPort(Port):
    """Output port
    Transfers data from the inside arrow to 'Out' side"""
    def __str__(self):
        return "Out%s" % super().__str__()

    def __repr__(self):
        return str(self)
