"""Inverse Dispatches for Inverses"""
from arrows import Arrow, Port, InPort
from arrows.port_attributes import (PortAttributes, make_error_port,
    make_param_port, get_port_attributes)
from arrows.std_arrows import *
from arrows.apply.constants import CONST, VAR
from arrows.util.misc import extract
from reverseflow.inv_primitives.inv_math_arrows import *
from reverseflow.util.mapping import Bimap
from reverseflow.util.misc import complement
import numpy as np
from typing import Set, Tuple, Dict, Sequence
from copy import deepcopy

PortMap = Dict[int, int]

def is_constant(p: Port, pv: PortAttributes):
    return p in pv and 'constant' in pv[p] and pv[p]['constant'] == CONST


def generic_binary_inv(arrow: Arrow,
                       port_values: PortAttributes,
                       PInverseArrow,
                       Port0ConstArrow,
                       Port0ConstPortMap,
                       Port1ConstArrow,
                       Port1ConstPortMap) -> Tuple[Arrow, PortMap]:
    # FIXME: Is this actually correct for mul/add/sub
    port_0_const = is_constant(arrow.in_ports()[0], port_values)
    port_1_const = is_constant(arrow.in_ports()[1], port_values)

    if port_0_const and port_1_const:
        # If both ports constant just return arrow as is
        inv_arrow = deepcopy(arrow)
        port_map = {0: 0, 1: 1, 2: 2}
    elif port_0_const:
        inv_arrow = Port0ConstArrow()
        port_map = Port0ConstPortMap
    elif port_1_const:
        inv_arrow = Port1ConstArrow()
        port_map = Port1ConstPortMap
    else:
        # Neither constant, do 'normal' parametric inversison
        inv_arrow = PInverseArrow()
        port_map = {0: 2, 1: 3, 2: 0}

    return inv_arrow, port_map


def inv_add(arrow: AddArrow, port_values: PortAttributes) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow,
                              port_values,
                              PInverseArrow=InvAddArrow,
                              Port0ConstArrow=SubArrow,
                              Port0ConstPortMap={0: 1, 1: 2, 2: 0},
                              Port1ConstArrow=SubArrow,
                              Port1ConstPortMap={0: 2, 1: 1, 2: 0})

def inv_sub(arrow: SubArrow, port_values: PortAttributes) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow,
                              port_values,
                              PInverseArrow=InvSubArrow,
                              Port0ConstArrow=SubArrow,
                              Port0ConstPortMap={0: 0, 1: 2, 2: 1},
                              Port1ConstArrow=AddArrow,
                              Port1ConstPortMap={0: 2, 1: 1, 2: 0})


def inv_cos(arrow: CosArrow, port_attr: PortAttributes) -> Tuple[Arrow, PortMap]:
    if is_constant(arrow.in_ports()[0], port_attr):
        return deepcopy(arrow), {0: 0, 1: 1}
    #FIXME: More rigorous than 0.999, should be 1.0 but get NaNs
    ibi = IntervalBoundIdentity(-0.999, 0.999)
    acos = ACosArrow()

    comp_arrow = CompositeArrow(name="approx_acos")
    in_port = comp_arrow.add_port()
    make_in_port(in_port)
    out_port = comp_arrow.add_port()
    make_out_port(out_port)
    error_port = comp_arrow.add_port()
    make_out_port(error_port)
    make_error_port(error_port)

    comp_arrow.add_edge(in_port, ibi.in_ports()[0])
    comp_arrow.add_edge(ibi.out_ports()[0], acos.in_ports()[0])
    comp_arrow.add_edge(acos.out_ports()[0], out_port)
    comp_arrow.add_edge(ibi.out_ports()[1], error_port)
    comp_arrow.is_wired_correctly()
    return comp_arrow, {0: 1, 1: 0}


# def inv_dupl(arrow: DuplArrow, const_in_ports: Set[InPort]) -> Tuple[Arrow, PortMap]:
#     assert arrow.in_ports()[0] not in const_in_ports, "Dupl is constant"
#     n_duplications = arrow.n_out_ports
#     inv_arrow = InvDuplArrow(n_duplications=n_duplications)
#     port_map = {arrow.in_ports()[0].index: inv_arrow.out_ports()[0].index}
#     port_map.update({arrow.out_ports()[i].index: inv_arrow.in_ports()[i].index for i in range(n_duplications)})
#     return inv_arrow, port_map

def inv_dupl_approx(arrow: DuplArrow, port_values: PortAttributes) -> Tuple[Arrow, PortMap]:
    # assert port_values[arrow.in_ports()[0]] == VAR, "Dupl is constant"
    n_duplications = arrow.n_out_ports
    inv_dupl = InvDuplArrow(n_duplications=n_duplications)
    approx_id = ApproxIdentityArrow(n_inputs=n_duplications)
    edges = Bimap()  # type: EdgeMap
    for i in range(n_duplications):
        edges.add(approx_id.out_ports()[i], inv_dupl.in_ports()[i])
    error_ports = [approx_id.out_ports()[n_duplications]]
    out_ports=inv_dupl.out_ports()+error_ports
    inv_arrow = CompositeArrow(edges=edges,
                               in_ports=approx_id.in_ports(),
                               out_ports=out_ports,
                               name="InvDuplApprox")
    make_error_port(inv_arrow.out_ports()[-1])
    port_map = {0: inv_arrow.ports()[-2].index}
    port_map.update({i+1:i for i in range(n_duplications)})
    inv_arrow.name = "InvDuplApprox"
    return inv_arrow, port_map


def inv_dupl(arrow: DuplArrow, port_values: PortAttributes):
    const_outs = []
    var_outs = []
    for i, out_port in enumerate(arrow.out_ports()):
        if out_port in port_values and 'constant' in port_values[out_port] and port_values[out_port]['constant'] == CONST:
            const_outs.append(i + 1)
        else:
            var_outs.append(i + 1)
    n_duplications = len(const_outs) + 1
    dupl = DuplArrow(n_duplications=n_duplications)
    in_ports = dupl.in_ports()
    out_ports = dupl.out_ports()
    edges = Bimap()  # type: EdgeMap
    n_duplications = len(var_outs)
    if n_duplications > 1:
        inv_dupl = InvDuplArrow(n_duplications=n_duplications)
        approx_id = ApproxIdentityArrow(n_inputs=n_duplications)
        in_ports = approx_id.in_ports()
        edges.add(inv_dupl.out_ports()[0], dupl.in_ports()[0])
        for i in range(n_duplications):
            edges.add(approx_id.out_ports()[i], inv_dupl.in_ports()[i])
        out_ports.append(approx_id.out_ports()[n_duplications])
    inv_arrow = CompositeArrow(edges=edges,
                               in_ports=in_ports,
                               out_ports=out_ports,
                               name="InvDupl")
    if n_duplications > 1:
        make_error_port(inv_arrow.out_ports()[-1])
    port_map = {0: arrow.num_out_ports()}
    i = 0
    for port in var_outs:
        port_map[port] = i
        i += 1
    for port in const_outs:
        port_map[port] = i
        i += 1
    return inv_arrow, port_map


def inv_exp(arrow: ExpArrow, port_attr: PortAttributes) -> Tuple[Arrow, PortMap]:
    if is_constant(arrow.in_ports()[0], port_attr):
        return deepcopy(arrow), {0: 0, 1: 1}
    log = LogArrow()
    return log, {0: 1, 1: 0}


def inv_gather(arrow: GatherArrow, port_attr: PortAttributes) -> Tuple[Arrow, PortMap]:
    if is_constant(arrow.in_ports()[0], port_attr) and is_constant(arrow.in_ports()[1], port_attr):
        return deepcopy(arrow), {0: 0, 1: 1, 2: 2}
    tensor_shape = port_attr[arrow.in_ports()[0]]['shape']
    if isinstance(tensor_shape, tuple):
        tensor_shape = list(tensor_shape)
    index_list_value = port_attr[arrow.in_ports()[1]]['value']
    index_list_compl = complement(index_list_value, tensor_shape)
    std1 = SparseToDenseArrow()
    std2 = SparseToDenseArrow()
    dupl1 = DuplArrow()
    dupl2 = DuplArrow()
    # TODO: don't do this, complement could be huge
    source_compl = SourceArrow(np.array(index_list_compl))
    source_tensor_shape = SourceArrow(np.array(tensor_shape))
    add = AddArrow()
    edges = Bimap()
    edges.add(source_compl.out_ports()[0], std1.in_ports()[0])
    edges.add(source_tensor_shape.out_ports()[0], dupl1.in_ports()[0])
    edges.add(dupl1.out_ports()[0], std1.in_ports()[1])
    edges.add(dupl1.out_ports()[1], std2.in_ports()[1])
    edges.add(std1.out_ports()[0], add.in_ports()[0])
    edges.add(std2.out_ports()[0], add.in_ports()[1])
    # orig_out_port, params, inp_list
    in_ports = [std2.in_ports()[2], std1.in_ports()[2], std2.in_ports()[0]]
    out_ports = [add.out_ports()[0]]
    op = CompositeArrow(in_ports=in_ports,
                        out_ports=out_ports,
                        edges=edges,
                        name="InvGather")
    make_param_port(op.in_ports()[1])
    return op, {0: 3, 1: 2, 2: 0}


def dict_subset(keys, dict):
    return {key: dict[key] for key in keys}

def inv_reshape(arrow: GatherArrow, port_values: PortAttributes) -> Tuple[Arrow, PortMap]:
    import pdb; pdb.set_trace()


def inv_mul(arrow: MulArrow, port_values: PortAttributes) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow,
                              port_values,
                              PInverseArrow=InvMulArrow,
                              Port0ConstArrow=DivArrow,
                              Port0ConstPortMap={0: 1, 1: 2, 2: 0},
                              Port1ConstArrow=DivArrow,
                              Port1ConstPortMap={0: 2, 1: 1, 2: 0})

def inv_div(arrow: DivArrow, port_values: PortAttributes) -> Tuple[Arrow, PortMap]:
    return generic_binary_inv(arrow,
                              port_values,
                              PInverseArrow=InvDivArrow,
                              Port0ConstArrow=DivArrow,
                              Port0ConstPortMap={0: 0, 1: 2, 2: 1},
                              Port1ConstArrow=MulArrow,
                              Port1ConstPortMap={0: 2, 1: 1, 2: 0})


def inv_neg(arrow: NegArrow, port_attr: PortAttributes) -> Tuple[Arrow, PortMap]:
    if is_constant(arrow.in_ports()[0], port_attr):
        return deepcopy(arrow), {0: 0, 1: 1}
    sub_port_attr = extract(arrow.ports(), port_attr)
    neg = NegArrow()
    return neg, {0: 1, 1: 0}


def inv_reshape(arrow: ReshapeArrow, port_attr: PortAttributes) -> Tuple[Arrow, PortMap]:
    import pdb; pdb.set_trace()
    sub_port_attr = extract(arrow.ports(), port_attr)



def inv_sin(arrow: SinArrow, port_attr: PortAttributes) -> Tuple[Arrow, PortMap]:
    if is_constant(arrow.in_ports()[0], port_attr):
        return deepcopy(arrow), {0: 0, 1: 1}
    ibi = IntervalBoundIdentity(-0.999, 0.999)
    asin = ASinArrow()

    comp_arrow = CompositeArrow(name="approx_asin")
    in_port = comp_arrow.add_port()
    make_in_port(in_port)
    out_port = comp_arrow.add_port()
    make_out_port(out_port)
    error_port = comp_arrow.add_port()
    make_out_port(error_port)
    make_error_port(error_port)

    comp_arrow.add_edge(in_port, ibi.in_ports()[0])
    comp_arrow.add_edge(ibi.out_ports()[0], asin.in_ports()[0])
    comp_arrow.add_edge(asin.out_ports()[0], out_port)
    comp_arrow.add_edge(ibi.out_ports()[1], error_port)
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
