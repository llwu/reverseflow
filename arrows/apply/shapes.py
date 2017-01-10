"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor, Graph, Variable
from pqdict import pqdict
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow, EdgeMap
from arrows.primitive.math_arrows import *
from arrows.primitive.control_flow_arrows import *
from arrows.primitive.cast_arrows import *
from arrows.primitive.constant import *
from reverseflow.util.misc import same
from typing import Tuple, List, Dict, MutableMapping, Union, Sequence
from collections import OrderedDict
from overloading import overload
from arrows.apply.interpret import interpret

ShapeList = Sequence[Tuple[int, ...]]


def same_to_n(shapes, n=1):
    assert same(shapes), "Shapes must be the same"
    return [shapes[0] for i in range(n)]


def same_conv(a: Arrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    return same_to_n(shapes, a.num_out_ports()), [(port, shapes[0]) for port in a.in_ports]


@overload
def conv(a: Arrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    assert False, "Error, no conversion for %s implemented" % a.name


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
    # TODO: param shape
    return [()]

@overload
def conv(a: RangeArrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    assert False

@overload
def conv(a: ReduceMeanArrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    assert False


@overload
def conv(a: CompositeArrow, shapes: ShapeList) -> ShapeList:
    assert len(shapes) == a.num_in_ports()
    return interpret(conv, a, shapes, return_emit=True)

def propagate_shapes(comp_arrow: CompositeArrow,
                     input_shapes: ShapeList):
    result, emit = interpret(conv, comp_arrow, input_shapes, return_emit=True)
    return result, dict(emit)
