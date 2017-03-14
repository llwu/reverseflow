"""Eliminate Redundant Parameters in Gather"""
import numpy as np

from arrows.port_attributes import *
from arrows.compositearrow import CompositeArrow
from arrows.transform.eliminate import filter_arrows, dupl_names
from arrows.std_arrows import AddArrow, MulArrow, SourceArrow, ScatterNdArrow, GatherNdArrow, UpdateArrow
from reverseflow.util.misc import complement_bool_list


def eliminate_gathernd(arrow: CompositeArrow):
    """Eliminates redundant parameters in GatherNd
    Args:
        a: Parametric Arrow prime for eliminate!
    Returns:
        New Parameteric Arrow with fewer parameters"""
    dupls = filter_arrows(lambda a: a.name in dupl_names, arrow)
    for dupl in dupls:
        slim_param_arrow = UpdateArrow()
        constraints = None
        free = None
        shape = None
        shape_source = None
        for p in dupl.in_ports():
            inv = arrow.neigh_ports(p)[0].arrow
            if inv.name == 'InvGatherNd':
                out, theta, indices = inv.in_ports()
                make_not_param_port(theta)
                arrow.add_edge(slim_param_arrow.out_port(0), theta)

                indices_val = get_port_value(indices)
                if shape is not None:
                    assert np.array_equal(shape, np.array(get_port_shape(inv.out_port(0))))
                else:
                    shape = np.array(get_port_shape(inv.out_port(0)))
                    shape_source = SourceArrow(shape)

                out = arrow.neigh_ports(out)[0]

                unset, unique = complement_bool_list(indices_val, shape)
                free = unset if free is None else np.logical_and(unset, free)  # FIXME: don't really need bool anymore
                unique_source = SourceArrow(unique)
                unique_inds = SourceArrow(indices_val[tuple(np.transpose(unique))])

                # add in a constraint for these new known values
                unique_upds = GatherNdArrow()
                arrow.add_edge(out, unique_upds.in_port(0))
                arrow.add_edge(unique_source.out_port(0), unique_upds.in_port(1))
                tmp = UpdateArrow()
                if constraints is None:
                    constraints = SourceArrow(np.zeros(shape, dtype=np.float32))
                arrow.add_edge(constraints.out_port(0), tmp.in_port(0))
                arrow.add_edge(unique_inds.out_port(0), tmp.in_port(1))
                arrow.add_edge(unique_upds.out_port(0), tmp.in_port(2))
                arrow.add_edge(shape_source.out_port(0), tmp.in_port(3))
                constraints = tmp
        if constraints is not None:
            # put in knowns
            arrow.add_edge(constraints.out_port(0), slim_param_arrow.in_port(0))
            # put in params
            make_param_port(slim_param_arrow.in_port(2))
            arrow.add_edge(shape_source.out_port(0), slim_param_arrow.in_port(3))
            inds = np.transpose(np.nonzero(free))
            if (free == free[0]).all():
                print("Assuming batched input")
                inds = np.array(np.split(inds, shape[0]))
            else:
                print("WARNING: Unbatched input, haven't designed for this case")
            inds_source = SourceArrow(inds)
            arrow.add_edge(inds_source.out_port(0), slim_param_arrow.in_port(1))
        # import pdb; pdb.set_trace()
