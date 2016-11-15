from reverseflow.util.mapping import Out
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.port import InPort, OutPort
from typing import List, Dict, Callable


def cached_invert(arrow: Arrow, determined, arrow_to_inv):
    if arrow in arrow_to_inv:
        return arrow_to_inv[arrow]
    else:
        assert arrow.is_primitive()
        determined_in_ports = [port in determined for port in arrow.in_ports]
        inv_arrow_classes = fwd_inv.fwd(arrow.type)
        inv_arrow_class = next(iter(inv_arrow_classes))
        arrow_to_inv[arrow] = inv_arrow
        return inv_arrow


def find_all_subarrows(arrow: Arrow, arrow_type) -> List[Arrow]:
    pass


def naive(a: Arrow, fwd_inv: OneToManyList[Arrow, Arrow]) -> Arrow:
    source_arrows = find_all_subarrows(a, SourceArrow)
    determined_ports = [sa.out_ports[0] for sa in source_arrows]
    return invert_determined(a, determined_ports, fwd_inv)


def invert_determined(arrow: Arrow,
                      marked_inports: List[InPort],
                      marked_outports: List[OutPort],
                      dispatch: Dict[Arrow, Callable]) -> Arrow:
    # For edge in edge
    # Invert
    for edge in arrow.edges:
        ...


def invert_determined(arrow: Arrow,
                      marked_inports: List[InPort],
                      marked_outports: List[OutPort],
                      dispatch: Dict[Arrow, Callable]) -> Arrow:
    """Invert an arrow to construct a parametric inverse"""

    # A priority queue for sub_arrows
    # priority is the number of inputs it has which have already been seen
    arrow_colors = pqdict()

    # 'See' inports
    for in_port in arrow.in_ports:
        # TODO: Decrement for constant inputs
        if in_port.arrow in arrow_colors:
            arrow_colors[in_port.arrow] = arrow_colors[in_port.arrow] - 1
        else:
            arrow_colors[in_port.arrow] = in_port.arrow.num_in_ports() - 1

    print_arrow_colors(arrow_colors)

    # Bijection between arrows and their inverses
    arrow_to_inv = Bimap()  # type: Bimap[Arrow, Arrow]
    edges = Bimap()  # type: EdgeMap
    while len(arrow_colors) > 0:
        print_arrow_colors(arrow_colors)
        sub_arrow, priority = arrow_colors.popitem()
        print("Inverting ", sub_arrow.name)
        assert priority == 0, "Must resolve all inputs to sub_arrow first"
        assert sub_arrow.is_primitive(), "Cannot convert unflat arrow"

        inv_sub_arrow, corres = dispatch[arrow.type](arrow)
        for out_port in inv_arrow.out_ports:
            corres_in_port = sub_arrow.in_ports[i]
            if corres_in_port in arrow.in_ports:
                pass
                # Set as an out_port
            else:
                fwd_neigh_out_port = arrow.edges.inv(corres_in_port)
                neigh_inv_arrow = arrow_to_inv.fwd(fwd_neigh_out_port.arrow)
                neigh_in_port = neigh_inv_arrow.in_ports[fwd_neigh_out_port.id]
                edges.add_edge()


        for out_port in inv_arrow.out_ports:
            edge.add(out_port, arrow.in_ports)

        inputs = list(arrow_exprs[sub_arrow].values())
        # import pdb; pdb.set_trace()
        outputs = conv(sub_arrow, inputs)
        assert len(outputs) == len(sub_arrow.out_ports), "diff num outputs"

        for i, out_port in enumerate(sub_arrow.out_ports):
            # FIXME: this is linear search, encapsulate
            if out_port not in comp_arrow.out_ports:
                neigh_port = comp_arrow.neigh_in_port(out_port)
                neigh_arrow = neigh_port.arrow
                if neigh_arrow is not comp_arrow:
                    assert neigh_arrow in arrow_colors
                    arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1
                    default_add(arrow_exprs, neigh_arrow, neigh_port.index,
                                outputs[i])

    return arrow_exprs
