"""Inverse Dispatches for Inverses"""
from arrows.primitive.math_arrows import *
from arrows.primitive.control_flow_arrows import *
from reverseflow.inv_primitives.inv_math_arrows import *
from reverseflow.inv_primitives.inv_control_flow_arrows import *
from arrows.composite.approx import *
from arrows.compose import compose_comb
from arrows.arrow import Arrow
from arrows.port import Port, InPort
from reverseflow.util.mapping import Bimap
from typing import Set, Tuple, Dict

PortMap = Dict[Port, Port]


def generic_binary_inv(arrow: Arrow,
                       const_in_ports: Set[InPort],
                       PInverseArrow,
                       Port0ConstArrow,
                       Port1ConstArrow) -> Tuple[Arrow, PortMap]:
    port_0_const = arrow.in_ports[0] in const_in_ports
    port_1_const = arrow.in_ports[1] in const_in_ports

    assert not(port_0_const and port_1_const), "No inverse,Both inputs constant"
    if not(port_0_const or port_1_const):
        inv_arrow = PInverseArrow()
        port_map = {arrow.in_ports[0]: inv_arrow.out_ports[0],
                    arrow.in_ports[1]: inv_arrow.out_ports[1],
                    arrow.out_ports[0]: inv_arrow.in_ports[0]}
    elif port_0_const:
        inv_arrow = Port0ConstArrow()
        port_map = {arrow.in_ports[0]: inv_arrow.in_ports[1],
                    arrow.in_ports[1]: inv_arrow.out_ports[0],
                    arrow.out_ports[0]: inv_arrow.in_ports[0]}
    else:
        assert port_1_const
        inv_arrow = Port1ConstArrow()
        port_map = {arrow.in_ports[1]: inv_arrow.in_ports[1],
                    arrow.in_ports[0]: inv_arrow.out_ports[0],
                    arrow.out_ports[0]: inv_arrow.in_ports[0]}
    return inv_arrow, port_map


def inv_add(arrow: AddArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow, const_in_ports, PInverseArrow=InvAddArrow,
                              Port0ConstArrow=SubArrow, Port1ConstArrow=SubArrow)

def inv_sub(arrow: SubArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow, const_in_ports, PInverseArrow=InvSubArrow,
                              Port0ConstArrow=SubArrow, Port1ConstArrow=AddArrow)

def inv_mul(arrow: MulArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow, const_in_ports, PInverseArrow=InvMulArrow,
                              Port0ConstArrow=DivArrow, Port1ConstArrow=DivArrow)

def inv_div(arrow: DivArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow, const_in_ports, PInverseArrow=InvDivArrow,
                              Port0ConstArrow=DivArrow, Port1ConstArrow=MulArrow)


def inv_dupl(arrow: DuplArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    assert arrow.in_ports[0] not in const_in_ports, "Dupl is constant"
    n_duplications = arrow.n_out_ports
    inv_arrow = InvDuplArrow(n_duplications=n_duplications)
    port_map = {arrow.in_ports[0]: inv_arrow.out_ports[0]}
    port_map.update({arrow.out_ports[i]: inv_arrow.in_ports[i] for i in range(n_duplications)})
    return inv_arrow, port_map


def inv_dupl_approx(arrow: DuplArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
    assert arrow.in_ports[0] not in const_in_ports, "Dupl is constant"
    n_duplications = arrow.n_out_ports
    inv_dupl = InvDuplArrow(n_duplications=n_duplications)
    approx_id = ApproxIdentityArrow(n_inputs=n_duplications)
    edges = Bimap()  # type: EdgeMap
    for i in range(n_duplications):
        edges.add(approx_id.out_ports[i], inv_dupl.in_ports[i])
    error_ports = [approx_id.out_ports[n_duplications]]
    out_ports=inv_dupl.out_ports+error_ports
    inv_arrow = CompositeArrow(edges=edges,
                               in_ports=approx_id.in_ports,
                               out_ports=out_ports,
                               name="InvDuplApprox")
    inv_arrow.add_out_port_attribute(len(out_ports)-1, "Error")
    # inv_arrow.change_out_port_type(ErrorPort, len(out_ports)-1)
    # inv_arrow = compose_comb(approx_id, inv_dupl, {i: i for i in range(n_duplications)})
    port_map = {arrow.in_ports[0]: inv_arrow.out_ports[0]}
    port_map.update({arrow.out_ports[i]: inv_arrow.in_ports[i] for i in range(n_duplications)})
    inv_arrow.name = "InvDuplApprox"
    return inv_arrow, port_map
