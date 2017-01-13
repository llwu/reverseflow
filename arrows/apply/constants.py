"""Propagate constants through graph"""
from arrows.port import Port
from arrows.std_arrows import SourceArrow
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from overloading import overload
from enum import Enum

class Const(Enum):
    CONSTANT = 0
    VARIABLE = 1

PortKnowns = Dict[Port, Const]

@overload
def sub_propagate(a: Arrow, port_to_known: PortKnowns, state=None):
    # TODO:
    # If all the inputs are constant then outputs should be constant
    # If not all inputs are known then make no changed
    # If one input is variable then all outputs should be variable

@overload
def sub_propagate(a: SourceArrow, port_to_known: PortKnowns, state=None):
    return {a.get_ports()[0]: Const.CONSTANT}


@overload
def sub_propagate(a: CompositeArrow, port_values: PortKnowns, state=None):
    return propagate(sub_propagate, a, port_values, state=None)
