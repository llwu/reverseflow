"""Reparameterization"""
from arrows.apply.propagate import propagate
from arrows.port_attributes import *
from arrows.compositearrow import CompositeArrow
from arrows.tfarrow import TfArrow
from typing import Tuple
from reverseflow.train.common import *


def reparam(comp_arrow: CompositeArrow,
            phi_shape: Tuple,
            nn_takes_input=True):
    """Reparameterize an arrow.  All parametric inputs now function of phi
    Args:
        comp_arrow: Arrow to reparameterize
        phi_shape: Shape of parameter input
    """
    port_attr = propagate(comp_arrow)
    reparam = CompositeArrow(name="%s_reparam" % comp_arrow.name)
    phi = reparam.add_port()
    set_port_shape(phi, phi_shape)
    make_in_port(phi)
    make_param_port(phi)
    nn = TfArrow(n_in_ports=1, n_out_ports=comp_arrow.num_param_ports())
    reparam.add_edge(phi, nn.in_port(0))
    i = 0
    for port in comp_arrow.ports():
        if is_param_port(port):
            reparam.add_edge(nn.out_port(i), port)
            i += 1
        else:
            re_port = reparam.add_port()
            if is_out_port(port):
                make_out_port(re_port)
                reparam.add_edge(port, re_port)
            if is_in_port(port):
                make_in_port(re_port)
                reparam.add_edge(re_port, port)
            if is_error_port(port):
                make_error_port(re_port)

    assert reparam.is_wired_correctly()
    return reparam


def minimum_gap(theta_tensors: Sequence[Tensor]):
    return ming_loss


def train_reparam(theta_tensors: Sequence[Tensor]):
    """
    Reparameterize a network by maximising the gap_ratio
    Args:
        theta_tensors: tensors for original parameterization
    """

    # compute the minimum gap and the maximum gap
    ming_loss = minimum_gap(theta_tensors)
    maxg_loss = accumulate_losses(error_tensors)
    loss = []
    train_y_tf()
    # We have different minimum gaps for different outputs
    # Just think of it as one big vector space and deal with that
    # then use the error as maximum gap


def reparam_arrow(arrow: Arrow,
                  theta_ports: Sequence[Port],
                  input_data: List,
                  error_filter=is_error_port,
                  **kwargs) -> CompositeArrow:

    import pdb; pdb.set_trace
    with tf.name_scope(arrow.name):
        input_tensors = gen_input_tensors(arrow, param_port_as_var=False)
        port_grab = {p: None for p in theta_ports}
        output_tensors = arrow_to_graph(arrow, input_tensors, port_grab=port_grab)

    theta_tensors = list(port_grab.values())
    param_tensors = [t for i, t in enumerate(input_tensors) if is_param_port(arrow.in_ports()[i])]
    error_tensors = [t for i, t in enumerate(output_tensors) if error_filter(arrow.out_ports()[i])]

    # FIXME: Add broadcasting nodes
    assert len(param_tensors) > 0, "Must have parametric inports"
    assert len(error_tensors) > 0, "Must have error outports"
    train_reparam(param_tensors, error_tensors, input_tensors, output_tensors,
                  input_data, **kwargs)
