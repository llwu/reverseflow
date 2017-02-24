"""Eliminate Redundant Parameters in Gather"""
import numpy as np

from arrows.port_attributes import *
from arrows.compositearrow import CompositeArrow
from arrows.transform.eliminate import filter_arrows, dupl_names
from arrows.std_arrows import IdentityArrow


def eliminate_gathernd(arrow: CompositeArrow):
    """Eliminates redundant parameters in GatherNd
    Args:
        a: Parametric Arrow prime for eliminate!
    Returns:
        New Parameteric Arrow with fewer parameters"""
    dupls = filter_arrows(lambda a: a.name in dupl_names, arrow)
    new_arrow = CompositeArrow(name="%s_elimgathernd" % arrow.name)
    eliminated = set()
    slim_param_arrows = set()
    for dupl in dupls:
        inv_gathers = set()
        for p in dupl.in_ports():
            inv_gathers.add(arrow.neigh_ports(p)[0].arrow)
        if len(inv_gathers) == 0:
            continue
        slim_param_arrow = IdentityArrow()
        slim_param_arrows.add(slim_param_arrow)
        slim_param = slim_param_arrow.out_port(0)
        for p in arrow.in_ports():
            if is_param_port(p) and arrow.neigh_ports(p)[0].arrow in inv_gathers:
                new_arrow.add_edge(slim_param, p)
                eliminated.add(p)
    for in_port in arrow.in_ports():
        if in_port not in eliminated:
            new_in_port = new_arrow.add_port()
            make_in_port(new_in_port)
            if is_param_port(in_port):
                make_param_port(new_in_port)
            transfer_labels(in_port, new_in_port)
            new_arrow.add_edge(new_in_port, in_port)
    for out_port in arrow.out_ports():
        c_out_port = new_arrow.add_port()
        make_out_port(c_out_port)
        transfer_labels(out_port, c_out_port)
        if is_error_port(out_port):
            make_error_port(c_out_port)
        new_arrow.add_edge(out_port, c_out_port)
    for slim_param_arrow in slim_param_arrows:
        slim_param = new_arrow.add_port()
        make_in_port(slim_param)
        make_param_port(slim_param)
        new_arrow.add_edge(slim_param, slim_param_arrow.in_port(0))

    import pdb; pdb.set_trace()
    assert new_arrow.is_wired_correctly()
    return new_arrow
