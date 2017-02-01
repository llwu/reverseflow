"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor, Graph, Variable
import numpy as np
from arrows.config import floatX
from arrows.port import InPort
from arrows.port_attributes import is_in_port, is_param_port
from arrows.arrow import Arrow
from arrows.sourcearrow import SourceArrow
from arrows.compositearrow import CompositeArrow, EdgeMap
from arrows.std_arrows import *
from arrows.apply.interpret import interpret
from typing import List, Dict, MutableMapping, Union, Sequence
from overloading import overload

def gen_input_tensors(arrow: Arrow):
    input_tensors = []
    for in_port in arrow.get_in_ports():
        if is_param_port(in_port):
            # FIXME for right shape
            input_tensors.append(tf.Variable(np.random.rand(1), name="blerg", dtype=floatX()))
        elif is_in_port(in_port):
            input_tensors.append(tf.placeholder(dtype=floatX()))
        else:
            assert False, "Don't know how to handle %s" % in_port
    return input_tensors

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
def conv(a: SinArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.sin(*args)]

@overload
def conv(a: CosArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.cos(*args)]

@overload
def conv(a: ASinArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.asin(*args)]

@overload
def conv(a: ACosArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.acos(*args)]


@overload
def conv(a: DuplArrow, args: TensorVarList) -> Sequence[Tensor]:
    # TODO: Genralize to n outputs
    return [args[0] for i in range(a.num_out_ports())]

@overload
def conv(a: InvDuplArrow, args: TensorVarList) -> Sequence[Tensor]:
    # TODO: Add assert that all args are equal
    return [args[0]]

@overload
def conv(a: AddNArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.add_n(args)]

@overload
def conv(a: CastArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.cast(args[0], dtype=a.to_dtype)]

@overload
def conv(a: ClipArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.clip_by_value(*args)]


@overload
def conv(a: AbsArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.abs(args[0])]

@overload
def conv(a: MaxArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.maximum(args[0], args[1])]

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
def conv(a: SourceArrow, args: TensorVarList) -> Sequence[Tensor]:
    assert len(args) == 0, "Source arrow has no inputs"
    return [tf.constant(a.value)]

@overload
def conv(a: GreaterArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.greater(args[0], args[1])]

@overload
def conv(a: IfArrow, args: TensorVarList) -> Sequence[Tensor]:
    #import pdb; pdb.set_trace()
    return [tf.where(*args)]

@overload
def conv(a: GatherArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.gather(*args)]

@overload
def conv(a: SparseToDenseArrow, args: TensorVarList) -> Sequence[Tensor]:
    return [tf.sparse_to_dense(*args)]

@overload
def conv(a: CompositeArrow, args: TensorVarList) -> Sequence[Tensor]:
    assert len(args) == a.num_in_ports()
    with tf.name_scope(a.name):
        # import pdb; pdb.set_trace()
        return interpret(conv, a, args)


def arrow_to_graph(comp_arrow: CompositeArrow,
                   input_tensors: Sequence[Tensor]):
    input_tensors_wrapped = list(map(tf.identity, input_tensors))
    return interpret(conv, comp_arrow, input_tensors_wrapped)
