from reverseflow.util.mapping import ImageBimap


def cached_invert(arrow: Arrow, arrow_to_inv):
    if arrow in arrow_to_inv:
        return arrow_to_inv[arrow]
    else:
        assert arrow.is_primitive()
        inv_arrows = fwd_inv.back(a)
        inv_arrow = next(iter(inv_arrows))
        arrow_to_inv[arrow] = inv_arrow
        return inv_arrow


def naive_invert(a: Arrow, fwd_inv: ImageBimap[Arrow, Arrow]) -> Arrow:
    """Invert an arrow to construct a parametric inverse"""

    # If primitive arrow return arbitrary parametric inverses
    if is_primitive(a):
        inv_arrows = fwd_inv.back(a)
        return next(iter(inv_arrows))
    elif a.is_composite():
        new_edges = Bimap()  # type: Bimap[OutPort, InPort])
        arrow_to_inv = Dict{}
        for out_port, in_port in a.edges.items():
            # TODO handle case when constituent arrwos are composite
            new_inport = InPort(cached_invert(out_port.arrow, arrow_to_inv),
                                out_port.index)
            new_outport = OutPort(cached_invert(in_port.arrow, arrow_to_inv),
                                  in_port.index)
            new_edges.add(new_outport, new_inport)
        # Construct parametric arrow
