"""Defines mark()."""

from typing import Set, Tuple

from pqdict import pqdict

from reverseflow.arrows.port import InPort, OutPort
from reverseflow.arrows.arrow import Arrow


def mark(arrow: Arrow,
         knowns: Set[InPort]) -> Tuple[Set[InPort], Set[OutPort]]:
    """Propagates knowns throughout the arrow.
    Won't propagate to outside of the arrow.

    Args:
        arrow (Arrow): The arrow to propagate throughout.
        knowns (Set[InPort]): The in ports which we know to be known.

    Returns:
        Set[InPort]: The in ports which are known as a result.
    """
    to_mark = pqdict()
    marked_inports = set()
    marked_outports = set()

    def dec(sub_arrow):
        """Bumps sub_arrow up in the queue."""
        if sub_arrow in to_mark:
            to_mark[sub_arrow] -= 1
        else:
            to_mark[sub_arrow] = sub_arrow.num_in_ports() - 1

    def add(out_port):
        """Marks out_port and (if it exists) the neighboring in port."""
        marked_outports.add(out_port)
        if out_port in arrow.edges.keys():
            in_port = arrow.neigh_in_port(out_port)
            marked_inports.add(in_port)
            dec(in_port.arrow)

    for known in knowns:
        marked_inports.add(known)
        dec(known.arrow)

    while len(to_mark) > 0:
        sub_arrow, priority = to_mark.popitem()
        assert priority >= 0, "knowns > num_in_ports?"
        if priority == 0:
            for i in range(sub_arrow.num_out_ports()):
                add(OutPort(sub_arrow, i))
        elif sub_arrow.is_composite():
            sub_knowns = set()
            for i, in_port in enumerate(sub_arrow.in_ports):
                if InPort(sub_arrow, i) in marked_inports:  # outer InPort
                    sub_knowns.add(in_port)  # inner InPort
            _, sub_marked_outports = mark(sub_arrow, sub_knowns)
            for i, out_port in enumerate(sub_arrow.out_ports):
                if out_port in sub_marked_outports:  # inner OutPort
                    add(OutPort(sub_arrow, i))  # outer OutPort

    return marked_inports, marked_outports
