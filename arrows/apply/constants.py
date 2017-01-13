"""Propagate constants through graph"""
from arrows.port import Port
from arrows.std_arrows import SourceArrow
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
    pass
    # TODO:
    # If all the inputs are constant then outputs should be constant
    # If not all inputs are known then make no changed
    # If one input is variable then all outputs should be variable


@overload
def sub_propagate(a: SourceArrow, port_to_known: PortValues, state=None):
    return {a.get_ports()[0]: ValueType.CONSTANT}


@overload
def sub_propagate(a: CompositeArrow, port_values: PortValues, state=None):
    return propagate(sub_propagate, a, port_values, state=None)


def propagate_constants(comp_arrow: CompositeArrow) -> PortValues:
    """Propagate constants (originating in SourceArrows) around the graph"""
    propagate(sub_propagate, comp_arrow, {}, {})
