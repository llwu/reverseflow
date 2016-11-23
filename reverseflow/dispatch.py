"""Inverse Dispatches for Inverses"""
from reverseflow.arrows.primitive.math_arrows import AddArrow, SubArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.inv_primitives.inv_math_arrows import InvAddArrow
from reverseflow.inv_primitives.inv_control_flow_arrows import InvDuplArrow
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.port import Port, InPort
from reverseflow.util.mapping import Bimap
from typing import Set, Tuple, Dict

PortMap = Dict[Port, Port]

def std_port_map(arrow, inv_arrow):
    assert len(arrow.in_ports) == len(inv_arrow.out_ports)
    assert len(arrow.out_ports) == len(inv_arrow.in_ports)
    port_map = Bimap # type: Bimap[Port, Port]
    for i in range(len(arrow.out_ports)):
        port_map.add(arrow.out_ports[i])


def inv_add(arrow: AddArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    port_0_const = arrow.in_ports[0] in const_in_ports
    port_1_const = arrow.in_ports[1] in const_in_ports

    assert not(port_0_const and port_1_const), "Both inputs are constant, cannot invert"
    if not(port_0_const or port_1_const):
        inv_arrow = InvAddArrow()
        port_map  = {arrow.in_ports[0]:inv_arrow.out_ports[0],
                     arrow.in_ports[1]:inv_arrow.out_ports[1],
                     arrow.out_ports[0]:inv_arrow.in_ports[0]}

    elif port_0_const:
        inv_arrow = SubArrow()
        port_map = {arrow.in_ports[0]:inv_arrow.in_ports[1],
                    arrow.in_ports[1]:inv_arrow.out_ports[0],
                    arrow.out_ports[0]:inv_arrow.in_ports[0]}
    else:
        assert port_1_const
        inv_arrow = SubArrow()
        port_map = {arrow.in_ports[1]:inv_arrow.in_ports[1],
                    arrow.in_ports[0]:inv_arrow.out_ports[0],
                    arrow.out_ports[0]:inv_arrow.in_ports[0]}
    return inv_arrow, port_map

def inv_dupl(arrow: DuplArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    assert arrow.in_ports[0] not in const_in_ports, "Dupl is constant"
    inv_arrow = InvDuplArrow()
    port_map = {arrow.in_ports[0]: inv_arrow.out_ports[0],
                arrow.out_ports[0]: inv_arrow.in_ports[0],
                arrow.out_ports[1]: inv_arrow.in_ports[1]}
    return inv_arrow, port_map