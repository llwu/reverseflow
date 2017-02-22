"""Eliminate Redundant Parameters"""
from arrows.port_attributes import *
from arrows.compositearrow import CompositeArrow
from arrows.apply.propagate import propagate
from arrows.transform.symbolic_tensor import SymbolicTensor, unroll
from arrows.std_arrows import GatherArrow, SourceArrow, ReshapeArrow
import numpy as np
# from arrows.port_attributes import get_sh

def filter_arrows(fil, arrow: CompositeArrow, deep=True):
    good_arrows = set()
    for sub_arrow in arrow.get_sub_arrows():
        if fil(sub_arrow):
            good_arrows.add(sub_arrow)
        if deep and isinstance(sub_arrow, CompositeArrow):
            for sub_sub_arrow in filter_arrows(fil, sub_arrow, deep=deep):
                good_arrows.add(sub_sub_arrow)
    return good_arrows

dupl_names = ['InvDuplApprox']

def num_unique_elem(equiv_thetas):
    return len(set(equiv_thetas.values()))

def find_equivalent_thetas(dupl_to_equiv):
    equiv_thetas = {}
    group_index = 0
    dupl_unrolled = {k: list(map(unroll, v)) for k, v in dupl_to_equiv.items()}

    # Find the equivalence classes
    for dupl, constraints in dupl_unrolled.items():
        constraint_len = len(constraints[0])
        assert all((len(constraint) == constraint_len for constraint in constraints))
        for i in range(constraint_len):
            # Find the mapping index
            already_in_map = False
            all_zero = True
            for j in range(len(constraints)):
                theta = constraints[j][i]
                if theta != 0 and theta in equiv_thetas:
                    if already_in_map:
                        assert equiv_thetas[theta] == group_index
                    group_index = equiv_thetas[theta]
                    already_in_map = True
            # Now assign them all
            for j in range(len(constraints)):
                theta = constraints[j][i]
                if theta != 0:
                    all_zero = False
                    equiv_thetas[theta] = group_index

            if not already_in_map and not all_zero:
                group_index += 1

    return equiv_thetas

def create_arrow(arrow: CompositeArrow, equiv_thetas, port_attr, valid_ports, symbt_ports):
    # New parameter space should have nclasses elements
    nclasses = num_unique_elem(equiv_thetas)
    new_arrow = CompositeArrow(name="%s_elim" % arrow.name)
    for out_port in arrow.out_ports():
        c_out_port = new_arrow.add_port()
        make_out_port(out_port)
        transfer_labels(out_port, c_out_port)
        if is_error_port(out_port):
            make_out_port(c_out_port)
            make_error_port(c_out_port)
        new_arrow.add_edge(out_port, c_out_port)

    slim_param = new_arrow.add_port()
    make_in_port(slim_param)
    make_param_port(slim_param)
    set_port_shape(slim_param, (nclasses,))
    for in_port in arrow.in_ports():
        if in_port in valid_ports:
            symbt = symbt_ports[in_port]['symbolic_tensor']
            indices = []
            for theta in symbt.symbols:
                setid = equiv_thetas[theta]
                indices.append(setid)
            shape = get_port_shape(in_port, port_attr)
            gather = GatherArrow()
            src = SourceArrow(np.array(indices))
            shape_shape = SourceArrow(np.array(shape))
            reshape = ReshapeArrow()
            new_arrow.add_edge(slim_param, gather.in_port(0))
            new_arrow.add_edge(src.out_port(0), gather.in_port(1))
            new_arrow.add_edge(gather.out_port(0), reshape.in_port(0))
            new_arrow.add_edge(shape_shape.out_port(0), reshape.in_port(1))
            new_arrow.add_edge(reshape.out_port(0), in_port)
        else:
            new_in_port = new_arrow.add_port()
            make_in_port(new_in_port)
            if is_param_port(in_port):
                make_param_port(new_in_port)
            transfer_labels(in_port, new_in_port)
            new_arrow.add_edge(new_in_port, in_port)

    assert new_arrow.is_wired_correctly()
    return new_arrow


def eliminate(arrow: CompositeArrow):
    """Eliminates redundant parameter
    Args:
        a: Parametric Arrow prime for eliminate!
    Returns:
        New Parameteric Arrow with fewer parameters"""

    # Warning: This is a huge hack

    # Get the shapes of param ports
    port_attr = propagate(arrow)
    symbt_ports = {}
    for port in arrow.in_ports():
        if is_param_port(port):
            shape = get_port_shape(port, port_attr)
            symbt_ports[port] = {}
            # Create a symbolic tensor for each param port
            st = SymbolicTensor(shape=shape, name="port%s" % port.index, port=port)
            symbt_ports[port]['symbolic_tensor'] = st

    # repropagate
    port_attr = propagate(arrow, symbt_ports)
    # as a hack, just look on ports of duples to  find symbolic tensors which
    # should be equivalent
    dupls = filter_arrows(lambda a: a.name in dupl_names, arrow)
    dupl_to_equiv = {}

    # Not all ports contain ports with symbolic tensor constraints
    valid_ports = set()
    for dupl in dupls:
        equiv = []
        for p in dupl.ports():
            if 'symbolic_tensor' in port_attr[p]:
                valid_ports.add(port_attr[p]['symbolic_tensor'].port)
                equiv.append(port_attr[p]['symbolic_tensor'])
        dupl_to_equiv[dupl] = equiv

    equiv_thetas = find_equivalent_thetas(dupl_to_equiv)
    return create_arrow(arrow, equiv_thetas, port_attr, valid_ports, symbt_ports)
