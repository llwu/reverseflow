"""Compute shapes of outputs"""
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.std_arrows import *
from reverseflow.util.misc import same
from typing import Tuple, List, Dict, MutableMapping, Sequence
from arrows.apply.interpret import interpret
from overloading import overload
from numpy import ndarray

ShapeList = Sequence

@overload
def constant_to_shape(x: int):
    return [()]

@overload
def constant_to_shape(x: float):
    return [()]

@overload
def constant_to_shape(x: ndarray):
    return [x.shape]

def same_to_n(shapes, n=1):
    assert same(shapes), "Shapes must be the same"
    return [shapes[0] for i in range(n)]


def same_conv(a: Arrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    shapes = [shape for shape in shapes if shape is not None]
    assert len(shapes) > 0, "At least one input must have known shape"
    return same_to_n(shapes, a.num_out_ports()), [(port, shapes[0]) for port in a.in_ports]


@overload
def conv(a: Arrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    assert False, "Error, no conversion for %s implemented" % a.__class__.__name__

@overload
def conv(a: SourceArrow, shapes: ShapeList) -> ShapeList:
    return constant_to_shape(a.value)


@overload
def conv(a: AddArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: SubArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: NegArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: PowArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: ExpArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: LogArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: LogBaseArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: MulArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: DivArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)


@overload
def conv(a: DuplArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)

@overload
def conv(a: AddNArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)

@overload
def conv(a: CastArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)

@overload
def conv(a: AbsArrow, shapes: ShapeList) -> ShapeList:
    return same_conv(a, shapes)

@overload
def conv(a: RankArrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    assert shapes[0] is not None, "Input shape must be known"
    return [()], [(a.in_ports[0], shapes[0])]

@overload
def conv(a: RangeArrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    assert False

@overload
def conv(a: ReduceMeanArrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    assert False


@overload
def conv(a: SourceArrow, shapes: ShapeList) -> ShapeList:
    return [a.get_shape()]


@overload
def conv(a: CompositeArrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    result, emit = interpret(conv, a, shapes, return_emit=True)
    emit = dict(emit)
    to_emit = [(port, emit[inner_port]) for port in a.ports
               if port in a.edges
               for inner_port in a.edges.fwd(port)
               if inner_port in emit]
    return result, to_emit

def propagate_shapes(comp_arrow: CompositeArrow,
                     input_shapes: ShapeList):
    result, emit = conv(comp_arrow, input_shapes)
    return result, dict(emit)
