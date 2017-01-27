from arrows.port import Port
from typing import Sequence

def make_in_port(port: Port) -> None:
    """Make 'port' an InPort"""
    port.arrow.port_attributes[port.index]["InOut"] = "InPort"


def is_in_port(port: Port) -> bool:
    """Is `port` an InPort"""
    port_attributes = port.arrow.port_attributes[port.index]
    return "InOut" in port_attributes and port_attributes["InOut"] == "InPort"


def make_out_port(port: Port) -> None:
    """Make 'port' an OutPort"""
    port.arrow.port_attributes[port.index]["InOut"] = "OutPort"


def is_out_port(port: Port):
    """Is `port` an OutPort"""
    port_attributes = port.arrow.port_attributes[port.index]
    return "InOut" in port_attributes and port_attributes["InOut"] == "OutPort"


def make_param_port(port: Port) -> None:
    """Make `port` as a parametric port"""
    assert is_in_port(port), "A port must be parametric to be an in_port"
    port.arrow.port_attributes[port.index]["parametric"] = True


def is_param_port(port: Port) -> bool:
    """Is `port` a parametric port"""
    port_attributes = port.arrow.port_attributes[port.index]
    return "parametric" in port_attributes and port_attributes["parametric"] is True


def make_error_port(port: Port) -> None:
    """Make `port` as a error port"""
    assert is_out_port(port), "An error port must be error to be an out_port"
    port.arrow.port_attributes[port.index]["error"] = True


def is_error_port(port: Port) -> bool:
    """Is `port` an error port"""
    port_attributes = port.arrow.port_attributes[port.index]
    return "error" in port_attributes and port_attributes["error"] is True


Shape = Sequence[int]


def get_port_shape(port: Port) -> Shape:
    """Get the shape of `port"""
    port_attributes = port.arrow.port_attributes[port.index]
    return port_attributes["shape"]


def set_port_shape(port: Port, shape: Shape):
    """Set the shape of `port` to `shape`"""
    port_attributes = port.arrow.port_attributes[port.index]
    port_attributes["shape"] = shape
