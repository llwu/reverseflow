"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor, Graph, Variable
from pqdict import pqdict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from reverseflow.arrows.primitive.cast_arrows import *
from reverseflow.arrows.primitive.constant import *
from typing import Tuple, List, Dict, MutableMapping, Union, Sequence
from collections import OrderedDict
from overloading import overload
from reverseflow.arrows.apply.interpret import interpret

TensorVarList = Union[Sequence[Tensor], Sequence[Variable]]

@overload
def conv(a: Arrow, args: TensorVarList) -> Sequence[Tensor]:
    assert False, "Error, no conversion for %s implemented" % a.name


@overload
def conv(a: AddArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.add(*args)]


@overload
def conv(a: SubArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.sub(*args)]


@overload
def conv(a: NegArrow, args: Sequence[Tensor]) -> Sequence[Tensor]:
    return [tf.neg(*args)]


@overload
def conv(a: PowArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.pow(*args)]


@overload
def conv(a: ExpArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.exp(*args)]


@overload
def conv(a: LogArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.log(*args)]


@overload
def conv(a: LogBaseArrow, args: TensorVarList) -> Sequence[Tensor]:
    # Tensorflow has no log of arbitrary base
    # so, use log _{b}(x)=log _{k}(x)}/log _{k}(b)
    return [tf.log(args[1]) / tf.log(args[0])]


@overload
def conv(a: MulArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.mul(*args)]


@overload
def conv(a: DivArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.div(*args)]


@overload
def conv(a: DuplArrow, args: TensorVarList) -> Sequence[Tensor]:
    # TODO: Genralize to n outputs
    return [args[0] for i in range(a.n_out_ports)]

@overload
def conv(a: AddNArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.add_n(args)]

@overload
def conv(a: CastArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.cast(args[0], dtype=a.to_dtype)]

@overload
def conv(a: AbsArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.abs(args[0])]

@overload
def conv(a: RankArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.rank(args[0])]

@overload
def conv(a: RangeArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.range(args[0], args[1])]

@overload
def conv(a: ReduceMeanArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.reduce_mean(args[0], reduction_indices=args[1])]


@overload
def conv(a: CompositeArrow, args: TensorVarList) -> Sequence[Tensor]:
    graph = tf.get_default_graph()
    assert len(args) == a.n_in_ports
    arrow_colors, arrow_tensors = inner_convert(a, args)
    result = arrow_to_graph(a,
                            args,
                            arrow_colors,
                            arrow_tensors,
                            graph)
    return result['output']

def arrow_to_new_graph(comp_arrow: CompositeArrow,
                       input_tensors: Sequence[Tensor],
                       graph: Graph):

    with graph.as_default():
        return interpret(conv, comp_arrow, input_tensors)
