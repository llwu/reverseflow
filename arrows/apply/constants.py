"""Propagate constants through graph"""
from arrows.port import Port
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.sourcearrow import SourceArrow
from arrows.apply.propagate import propagate
from overloading import overload
from enum import Enum
from typing import Dict


class ValueType(Enum):
    CONSTANT = 0
    VARIABLE = 1

CONST = ValueType.CONSTANT
VAR = ValueType.VARIABLE

PortValues = Dict[Port, ValueType]

def const_iff_const(a: Arrow, port_values: PortValues, state=None):
    # All the outputs are constant if and only if all the inputs are constant
    all_in_ports_resolved = (set(a.get_in_ports()) == set(port_values.keys()))
    if all_in_ports_resolved:
        if all((value == CONST for value in port_values.values())):
            return {port: CONST for port in a.get_ports()}
        else:
            return {port: port_values[port] if port in port_values else VAR for port in a.get_ports()}
    else:
        # no change
        return port_values


@overload
def sub_propagate(a: Arrow, port_values: PortValues, state=None):
    # most arrows are all constant output iff all constant input
    return const_iff_const(a, port_values, state=state)

@overload
def sub_propagate(a: SourceArrow, port_values: PortValues, state=None):
    return {port: CONST for port in a.get_ports()}


@overload
def sub_propagate(a: CompositeArrow, port_values: PortValues, state=None):
    return propagate(sub_propagate, a, port_values, state)


def propagate_constants(comp_arrow: CompositeArrow) -> PortValues:
    """Propagate constants (originating in SourceArrows) around the graph"""
    port_values = {port: VAR for port in comp_arrow.get_in_ports()}
    return propagate(sub_propagate, comp_arrow, port_values)
