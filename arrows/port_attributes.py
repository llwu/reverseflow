from arrows.port import Port
from typing import Sequence, Dict, Any
PortAttributes = Dict[Port, Dict[str, Any]]

def make_in_port(port: Port) -> None:
    """Make 'port' an InPort"""
    port.arrow.port_attr[port.index]["InOut"] = "InPort"


def is_in_port(port: Port) -> bool:
    """Is `port` an InPort"""
    port_attr = port.arrow.port_attr[port.index]
    return "InOut" in port_attr and port_attr["InOut"] == "InPort"


def make_out_port(port: Port) -> None:
    """Make 'port' an OutPort"""
    port.arrow.port_attr[port.index]["InOut"] = "OutPort"


def is_out_port(port: Port):
    """Is `port` an OutPort"""
    port_attr = port.arrow.port_attr[port.index]
    return "InOut" in port_attr and port_attr["InOut"] == "OutPort"


def make_param_port(port: Port) -> None:
    """Make `port` as a parametric port"""
    assert is_in_port(port), "A port must be parametric to be an in_port"
    port.arrow.port_attr[port.index]["parametric"] = True


def is_param_port(port: Port) -> bool:
    """Is `port` a parametric port"""
    port_attr = port.arrow.port_attr[port.index]
    return "parametric" in port_attr and port_attr["parametric"] is True


def make_error_port(port: Port) -> None:
    """Make `port` as a error port"""
    assert is_out_port(port), "An error port must be error to be an out_port"
    port.arrow.port_attr[port.index]["error"] = True


def is_error_port(port: Port) -> bool:
    """Is `port` an error port"""
    port_attr = port.arrow.port_attr[port.index]
    return "error" in port_attr and port_attr["error"] is True


def add_port_label(port: Port, label: str):
    """Add a string label to a port"""
    port_attr = port.arrow.port_attr[port.index]
    if "labels" not in port_attr:
        port_attr["labels"] = set()
    port_attr["labels"].add(label)


def has_port_label(port: Port, label: str) -> bool:
    """Does port have this label"""
    port_attr = port.arrow.port_attr[port.index]
    return "labels" in port_attr and label in port_attr['labels']

def get_port_labels(port: Port):
    """Get all the port labels"""
    port_attr = port.arrow.port_attr[port.index]
    if "labels" in port_attr:
        return port_attr['labels']
    else:
        return set()

def transfer_labels(from_port: Port, to_port: Port):
    """Transfer all the labels from from_port to to_port"""
    # FIXME: Should we just propagate labels?
    pls = get_port_labels(from_port)
    for label in pls:
        add_port_label(to_port, label)


Shape = Sequence[int]


def get_port_shape(port: Port, port_attr: PortAttributes=None) -> Shape:
    """Get the shape of `port"""
    # FIXME: (Maybe) Make port.port_attr same type as propagate port_attr
    # ir provider a helper function
    # add port_attr is None for all these functions
    port_attr = port.arrow.port_attr[port.index] if port_attr is None \
        else port_attr[port]
    assert "shape" in port_attr, "%s has no shape attr" % (port)
    return port_attr["shape"]


def set_port_shape(port: Port, shape: Shape):
    """Set the shape of `port` to `shape`"""
    port_attr = port.arrow.port_attr[port.index]
    port_attr["shape"] = shape


def get_port_attr(port: Port):
    """Get the attributes of the port"""
    return port.arrow.get_port_attr(port)

def port_has(port: Port, attribute: str, port_attr: PortAttributes) -> bool:
    """Does port have `attribute` in port_attr"""
    return port in port_attr and attribute in port_attr[port]


def ports_has(ports: Sequence[Port], attr: str, port_attr: PortAttributes):
    return all((port_has(port, attr, port_attr) for port in ports))


def any_port_has(ports: Sequence[Port], attr: str, port_attr: PortAttributes):
    return any((port_has(port, attr, port_attr) for port in ports))


def extract_attribute(attr: str, port_attr: PortAttributes):
    return {port: port_attr[port][attr] for port in port_attr.keys() \
            if attr in port_attr[port]}

def arrow_filter(arrow: str, port_attr: PortAttributes):
    return {k: v for k, v in port_attr.items() if k.arrow == arrow}
