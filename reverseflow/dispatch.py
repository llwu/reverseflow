"""Inverse Dispatches for Inverses"""
from arrows import Arrow, Port, InPort
from arrows.port_attributes import make_error_port, make_param_port
from arrows.std_arrows import *
from arrows.apply.constants import PortValues, CONST, VAR
from reverseflow.inv_primitives.inv_math_arrows import *
from reverseflow.util.mapping import Bimap
from typing import Set, Tuple, Dict

PortMap = Dict[int, int]

def generic_binary_inv(arrow: Arrow,
                       port_values: PortValues,
                       PInverseArrow,
                       Port0ConstArrow,
                       Port1ConstArrow) -> Tuple[Arrow, PortMap]:
    # FIXME: Is this actually correct for mul/add/sub
    port_0_const = port_values[arrow.get_in_ports()[0]] == CONST
    port_1_const = port_values[arrow.get_in_ports()[1]] == CONST

    if port_0_const and port_1_const:
        # If both ports constant just return arrow as is
        inv_arrow = arrow
        port_map = {0: 0, 1: 1, 2: 2}
    elif port_0_const:
        inv_arrow = Port0ConstArrow()
        port_map = {0: 1, 1: 2, 2: 0}
    elif port_1_const:
        inv_arrow = Port1ConstArrow()
        port_map = {0: 2, 1: 1, 2: 2}
    else:
        # Neither constant, do 'normal' parametric inversison
        inv_arrow = PInverseArrow()
        port_map = {0: 2, 1: 3, 2: 0}

    return inv_arrow, port_map


def inv_add(arrow: AddArrow, port_values: PortValues) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow, port_values, PInverseArrow=InvAddArrow,
                              Port0ConstArrow=SubArrow, Port1ConstArrow=SubArrow)


def inv_mul(arrow: AddArrow, port_values: PortValues) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow, port_values, PInverseArrow=InvMulArrow,
                              Port0ConstArrow=DivArrow, Port1ConstArrow=DivArrow)


def inv_sin(arrow: SinArrow, port_values: PortValues) -> Tuple[Arrow, PortMap]:
    ibi = IntervalBoundIdentity(-1.0, 1.0)
    asin = ASinArrow()

    comp_arrow = CompositeArrow(name="approx_asin")
    in_port = comp_arrow.add_port()
    make_in_port(in_port)
    out_port = comp_arrow.add_port()
    make_out_port(out_port)
    error_port = comp_arrow.add_port()
    make_out_port(error_port)
    make_error_port(error_port)

    comp_arrow.add_edge(in_port, ibi.get_in_ports()[0])
    comp_arrow.add_edge(ibi.get_out_ports()[0], asin.get_in_ports()[0])
    comp_arrow.add_edge(asin.get_out_ports()[0], out_port)
    comp_arrow.add_edge(ibi.get_out_ports()[1], error_port)
    comp_arrow.is_wired_correctly()
    return comp_arrow, {0: 1, 1: 0}


def inv_cos(arrow: CosArrow, port_values: PortValues) -> Tuple[Arrow, PortMap]:
    ibi = IntervalBoundIdentity(-1.0, 1.0)
    acos = ACosArrow()

    comp_arrow = CompositeArrow(name="approx_acos")
    in_port = comp_arrow.add_port()
    make_in_port(in_port)
    out_port = comp_arrow.add_port()
    make_out_port(out_port)
    error_port = comp_arrow.add_port()
    make_out_port(error_port)
    make_error_port(error_port)

    comp_arrow.add_edge(in_port, ibi.get_in_ports()[0])
    comp_arrow.add_edge(ibi.get_out_ports()[0], acos.get_in_ports()[0])
    comp_arrow.add_edge(acos.get_out_ports()[0], out_port)
    comp_arrow.add_edge(ibi.get_out_ports()[1], error_port)
    comp_arrow.is_wired_correctly()
    return comp_arrow, {0: 1, 1: 0}


# def inv_sub(arrow: SubArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
#     return generic_binary_inv(arrow, const_in_ports, PInverseArrow=InvSubArrow,
#                               Port0ConstArrow=SubArrow, Port1ConstArrow=AddArrow)
#
# def inv_div(arrow: DivArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
#     return generic_binary_inv(arrow, const_in_ports, PInverseArrow=InvDivArrow,
#                               Port0ConstArrow=DivArrow, Port1ConstArrow=MulArrow)
#

# def inv_dupl(arrow: DuplArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
#     assert arrow.get_in_ports()[0] not in const_in_ports, "Dupl is constant"
#     n_duplications = arrow.n_out_ports
#     inv_arrow = InvDuplArrow(n_duplications=n_duplications)
#     port_map = {arrow.get_in_ports()[0].index: inv_arrow.get_out_ports()[0].index}
#     port_map.update({arrow.get_out_ports()[i].index: inv_arrow.get_in_ports()[i].index for i in range(n_duplications)})
#     return inv_arrow, port_map


def inv_dupl_approx(arrow: DuplArrow, port_values: PortValues) -> Tuple[Arrow, PortMap]:
    assert port_values[arrow.get_in_ports()[0]] == VAR, "Dupl is constant"
    n_duplications = arrow.n_out_ports
    inv_dupl = InvDuplArrow(n_duplications=n_duplications)
    approx_id = ApproxIdentityArrow(n_inputs=n_duplications)
    edges = Bimap()  # type: EdgeMap
    for i in range(n_duplications):
        edges.add(approx_id.get_out_ports()[i], inv_dupl.get_in_ports()[i])
    error_ports = [approx_id.get_out_ports()[n_duplications]]
    out_ports=inv_dupl.get_out_ports()+error_ports
    inv_arrow = CompositeArrow(edges=edges,
                               in_ports=approx_id.get_in_ports(),
                               out_ports=out_ports,
                               name="InvDuplApprox")
    make_error_port(inv_arrow.get_out_ports()[-1])
    port_map = {0: inv_arrow.get_ports()[-2].index}
    port_map.update({i+1:i for i in range(n_duplications)})
    inv_arrow.name = "InvDuplApprox"
    return inv_arrow, port_map
