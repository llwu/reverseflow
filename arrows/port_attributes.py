from arrows.port import Port
from typing import Sequence, Dict, Any
PortAttributes = Dict[Port, Dict[str, Any]]

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


def add_port_label(port: Port, label: str):
    """Add a string label to a port"""
    port_attributes = port.arrow.port_attributes[port.index]
    if "labels" not in port_attributes:
        port_attributes["labels"] = set()
    port_attributes["labels"].add(label)


def has_port_label(port: Port, label: str) -> bool:
    """Does port have this label"""
    port_attributes = port.arrow.port_attributes[port.index]
    return "labels" in port_attributes and label in port_attributes['labels']


Shape = Sequence[int]


def get_port_shape(port: Port) -> Shape:
    """Get the shape of `port"""
    port_attributes = port.arrow.port_attributes[port.index]
    return port_attributes["shape"]


def set_port_shape(port: Port, shape: Shape):
    """Set the shape of `port` to `shape`"""
    port_attributes = port.arrow.port_attributes[port.index]
    port_attributes["shape"] = shape


def get_port_attributes(port: Port):
    """Get the attributes of the port"""
    return port.arrow.get_port_attributes(port)

def port_has(port: Port, attribute: str, port_attr: PortAttributes) -> bool:
    """Does port have `attribute` in port_attr"""
    return port in port_attr and attribute in port_attr[port]


def ports_has(ports: Sequence[Port], attr: str, port_attr: PortAttributes):
    return all((port_has(port, attr, port_attr) for port in ports))


def extract_attribute(attr: str, port_attr: PortAttributes):
    return {port: port_attr[port][attr] for port in port_attr.keys() \
            if attr in port_attr[port]}
