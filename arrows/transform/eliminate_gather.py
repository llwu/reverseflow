"""Eliminate Redundant Parameters in Gather"""
import numpy as np

from arrows.port_attributes import *
from arrows.compositearrow import CompositeArrow
from arrows.transform.eliminate import filter_arrows, dupl_names
from arrows.std_arrows import AddArrow, MulArrow, SourceArrow, ScatterNdArrow
from reverseflow.util.misc import complement_bool


def eliminate_gathernd(arrow: CompositeArrow):
    """Eliminates redundant parameters in GatherNd
    Args:
        a: Parametric Arrow prime for eliminate!
    Returns:
        New Parameteric Arrow with fewer parameters"""
    dupls = filter_arrows(lambda a: a.name in dupl_names, arrow)
    for dupl in dupls:
        slim_param_arrow = AddArrow()
        constraints = None
        free = None
        for p in dupl.in_ports():
            inv = arrow.neigh_ports(p)[0].arrow
            if inv.name == 'InvGatherNd':
                out, theta, indices = inv.in_ports()
                make_not_param_port(theta)
                arrow.add_edge(slim_param_arrow.out_port(0), theta)

                indices_val = get_port_value(indices)
                shape = np.array(get_port_shape(inv.out_port(0)))
                shape_source = SourceArrow(shape)

                out = arrow.neigh_ports(out)[0]
                indices = arrow.neigh_ports(indices)[0]

                # free_d is which new values of parameters we learned. then free is all known values so far.
                unset = complement_bool(indices_val, shape)
                free_d = 1 - unset if free is None else free - unset * free
                free = unset if free is None else unset * free
                free_d_source = SourceArrow(free_d)

                # add in a constraint for these new known values
                constrain = MulArrow()
                arrow.add_edge(free_d_source.out_port(0), constrain.in_port(0))
                scatter = ScatterNdArrow()
                arrow.add_edge(scatter.out_port(0), constrain.in_port(1))
                arrow.add_edge(indices, scatter.in_port(0))
                arrow.add_edge(out, scatter.in_port(1))
                arrow.add_edge(shape_source.out_port(0), scatter.in_port(2))
                if constraints is None:
                    constraints = constrain
                else:
                    tmp = AddArrow()
                    arrow.add_edge(constraints.out_port(0), tmp.in_port(0))
                    arrow.add_edge(constrain.out_port(0), tmp.in_port(1))
                    constraints = tmp
        if constraints is not None:
            # put in knowns
            arrow.add_edge(constraints.out_port(0), slim_param_arrow.in_port(0))
            # zero out params which would conflict with knowns
            filter_param_arrow = MulArrow()
            make_param_port(filter_param_arrow.in_port(0))
            free_source = SourceArrow(free)
            arrow.add_edge(free_source.out_port(0), filter_param_arrow.in_port(1))
            arrow.add_edge(filter_param_arrow.out_port(0), slim_param_arrow.in_port(1))
        import pdb; pdb.set_trace()
