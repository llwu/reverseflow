"""Unparameterize A parametric Inverse"""
from arrows.port_attributes import is_param_port, transfer_labels, make_in_port, make_out_port, make_error_port, is_error_port
from arrows.compositearrow import CompositeArrow
from arrows.arrow import Arrow
from arrows.tfarrow import TfArrow

def unparam(arrow: Arrow):
    """Unparameerize an arrow by sticking a tfArrow between its normal inputs,
    and any parametric inputs
    """
    c = CompositeArrow(name="%s_unparam" % arrow.name)
    in_ports = [p for p in arrow.in_ports() if not is_param_port(p)]
    param_ports = [p for p in arrow.in_ports() if is_param_port(p)]
    nn = TfArrow(n_in_ports=len(in_ports), n_out_ports=len(param_ports))
    for i, in_port in enumerate(in_ports):
        c_in_port = c.add_port()
        make_in_port(c_in_port)
        transfer_labels(in_port, c_in_port)
        c.add_edge(c_in_port, in_port)
        c.add_edge(c_in_port, nn.in_port(i))

    for i, param_port in enumerate(param_ports):
        c.add_edge(nn.out_port(i), param_port)

    for out_port in arrow.out_ports():
        c_out_port = c.add_port()
        make_out_port(c_out_port)
        if is_error_port(out_port):
            make_error_port(c_out_port)
        transfer_labels(out_port, c_out_port)
        c.add_edge(out_port, c_out_port)

    assert c.is_wired_correctly()
    return c
