from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.std_arrows import *
from arrows.port_attributes import (is_error_port, make_error_port,
    make_out_port, is_param_port, make_param_port, add_port_label)
from reverseflow.invert import invert

def inv_fwd_loss_arrow(arrow: Arrow,
                       DiffArrow=SquaredDifference) -> CompositeArrow:
    """
    Arrow wihch computes |f(f^-1(y)) - y|
    Args:
        arrow: Forward function
    Returns:
        CompositeArrow
    """
    inverse = invert(arrow)
    c = CompositeArrow(name="%s_inv_fwd_loss" % arrow.name)

    # Make all in_ports of inverse inputs to composition
    for inv_in_port in inverse.in_ports():
        in_port = c.add_port()
        make_in_port(in_port)
        if is_param_port(inv_in_port):
            make_param_port(in_port)
        c.add_edge(in_port, inv_in_port)

    # Connect all out_ports of inverse to in_ports of f
    for i, out_port in enumerate(inverse.out_ports()):
        if not is_error_port(out_port):
            c.add_edge(out_port, arrow.in_port(i))
            c_out_port = c.add_port()
            # add edge from inverse output to composition output
            make_out_port(c_out_port)
            c.add_edge(out_port, c_out_port)

    # Pass errors (if any) of parametric inverse through as error_ports
    for i, out_port in enumerate(inverse.out_ports()):
        if is_error_port(out_port):
            error_port = c.add_port()
            make_out_port(error_port)
            make_error_port(error_port)
            add_port_label(error_port, "sub_arrow_error")
            c.add_edge(out_port, error_port)

    # find difference between inputs to inverse and outputs of fwd
    # make error port for each
    for i, out_port in enumerate(arrow.out_ports()):
        diff = DiffArrow()
        c.add_edge(c.in_port(i), diff.in_port(0))
        c.add_edge(out_port, diff.in_port(1))
        error_port = c.add_port()
        make_out_port(error_port)
        make_error_port(error_port)
        add_port_label(error_port, "inv_fwd_error")
        c.add_edge(diff.out_port(0), error_port)

    assert c.is_wired_correctly()
    return c
