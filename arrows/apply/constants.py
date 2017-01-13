"""Propagate constants through graph"""
from arrows.port import Port
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.apply.propagate import propagate
from overloading import overload
from enum import Enum
from typing import Dict


class ValueType(Enum):
    CONSTANT = 0
    VARIABLE = 1

PortValues = Dict[Port, ValueType]


@overload
def sub_propagate(a: Arrow, port_to_known: PortValues, state=None):
    const = True
    var = False
    for port in a.get_in_ports():
        if port not in port_to_known:
            const = False
        elif port_to_known[port] == ValueType.VARIABLE:
            const = False
            var = True
    if const:
        return {port: ValueType.CONSTANT for port in a.get_out_ports()}
    elif var:
        return {port: ValueType.VARIABLE for port in a.get_out_ports()}
    else:
        return {}
    pass


@overload
def sub_propagate(a: CompositeArrow, port_values: PortValues, state=None):
    return propagate(sub_propagate, a, port_values, state)


def propagate_constants(comp_arrow: CompositeArrow) -> PortValues:
    """Propagate constants (originating in SourceArrows) around the graph"""
    return propagate(sub_propagate, comp_arrow, {})
