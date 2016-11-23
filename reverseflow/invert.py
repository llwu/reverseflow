from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.port import InPort, OutPort
from reverseflow.defaults import default_dispatch
from reverseflow.arrows.marking import mark_source
from typing import Dict, Callable, Set
from reverseflow.util.mapping import Bimap


def is_constant(arrow: Arrow, const_in_ports: Set[InPort]):
    """Is the arrow purely a function of a source arrow"""
    return all((in_port in const_in_ports for in_port in arrow.in_ports))


def get_inverse(arrow: Arrow,
                const_in_ports: Set[InPort],
                const_out_ports: Set[OutPort],
                dispatch: Dict[Arrow, Callable],
                arrow_to_inv: Dict[Arrow, Arrow]):
    if arrow in arrow_to_inv:
        return arrow_to_inv[arrow]
    elif arrow.is_composite():
        return invert_const(arrow, const_in_ports, const_out_ports, dispatch)
    else:
        inv_arrow, port_map = dispatch[arrow.__class__](arrow, const_in_ports)
        arrow_to_inv[arrow] = (inv_arrow, port_map)
        return inv_arrow, port_map

def invert_const(arrow: CompositeArrow,
                 const_in_ports: Set[InPort],
                 const_out_ports: Set[OutPort],
                 dispatch: Dict[Arrow, Callable]) -> Arrow:
    """Invert an arrow assuming constants are known"""
    arrow_to_inv = dict()  # type: Dict[Arrow, Arrow]
    edges = Bimap()  # type: EdgeMap
    for out_port, in_port in arrow.edges.items():
        # Edge is constant
        if out_port in const_out_ports:
            inverse_in_arrow, port_map = get_inverse(in_port.arrow,
                                                     const_in_ports,
                                                     const_out_ports,
                                                     dispatch,
                                                     arrow_to_inv)
            new_in_port = port_map[in_port]
            edges.add(OutPort, new_in_port)
        else:
            inverse_in_arrow, in_port_map = get_inverse(in_port.arrow,
                                                     const_in_ports,
                                                     const_out_ports,
                                                     dispatch,
                                                     arrow_to_inv)
            inverse_out_arrow, out_port_map = get_inverse(out_port.arrow,
                                                      const_in_ports,
                                                      const_out_ports,
                                                      dispatch,
                                                      arrow_to_inv)
            edges.add(out_port_map[out_port], in_port_map[in_port])

    # Every inport is an outport
    inv_in_ports = []
    inv_out_ports = []
    for out_port in arrow.out_ports:
        inv_arrow, port_map = arrow_to_inv[out_port.arrow]
        in_port = port_map[out_port]
        assert isinstance(in_port, InPort)
        inv_in_ports.append(in_port)

    for in_port in arrow.in_ports:
        inv_arrow, port_map = arrow_to_inv[in_port.arrow]
        out_port = port_map[in_port]
        assert isinstance(out_port, OutPort)
        inv_out_ports.append(out_port)

    import pdb; pdb.set_trace()
    

    inv_name = "%s_inv" % arrow.name
    return CompositeArrow(edges=edges,
                          in_ports=inv_in_ports,
                          out_ports=inv_out_ports,
                          name=inv_name)


def invert(arrow: CompositeArrow,
           dispatch: Dict[Arrow, Callable]=default_dispatch) -> Arrow:
    """Construct a parametric inverse of arrow"""
    const_in_ports, const_out_ports = mark_source(arrow)
    return invert_const(arrow, const_in_ports, const_out_ports, dispatch)
