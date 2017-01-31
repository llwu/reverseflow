from arrows.arrow import Arrow
from arrows.port_attributes import (PortAttributes, port_has, ports_has,
    extract_attribute)


def constant_pred(arr: Arrow, port_attr: PortAttributes):
    return all((port_has(port, 'constant', port_attr) for port in arr.get_in_ports()))

def constant_dispatch(arr: Arrow, port_attr: PortAttributes):
    assert False

def shape_pred(arr: Arrow, port_attr: PortAttributes):
    return any((port_has(port, 'shape', port_attr) for port in arr.get_ports()))

def shape_dispatch(arr: Arrow, port_attr: PortAttributes):
    pts = extract_attribute('shape', port_attr)
    shape = list(pts.values())[0]
    return {port: {'shape': shape} for port in arr.get_ports()}
