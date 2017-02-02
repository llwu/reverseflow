"""Propagate constants through graph"""
from arrows.port_attributes import PortAttributes
from arrows.arrow import Arrow
from enum import Enum
from arrows.port_attributes import extract_attribute, ports_has

class ValueType(Enum):
    CONSTANT = 0
    VARIABLE = 1

CONST = ValueType.CONSTANT
VAR = ValueType.VARIABLE

def constant_pred(arr: Arrow, port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'constant', port_attr)


def constant_dispatch(arr: Arrow, port_attr: PortAttributes, state=None):
    ptc = extract_attribute('constant', port_attr)
    # All the outputs are constant if and only if all the inputs are constant
    if all((value == CONST for value in ptc.values())):
        return {port: {'constant': CONST} for port in arr.ports()}
    else:
        return {port: {'constant': ptc[port]} if port in ptc \
    else {'constant': VAR} for port in arr.ports()}
