"""Defines mark()."""

from typing import List

from overloading import overload

from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.apply.interpret import interpret


@overload
def conv(a: Arrow, marked: List[bool]) -> List[bool]:
    return [all(marked) for i in range(a.num_out_ports())]


@overload
def conv(a: CompositeArrow, marked: List[bool]) -> List[bool]:
    return interpret(conv, a, marked)


def mark(arrow: Arrow, knowns=set()) -> List:
    marks = interpret(conv, arrow, [port in knowns for port in arrow.in_ports])
    return set([arrow.out_ports[i] for i in range(arrow.num_out_ports()) if marks[i]])
