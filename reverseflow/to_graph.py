"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor, Graph, Variable
import numpy as np
from arrows.config import floatX
from arrows.port import InPort
from arrows.port_attributes import *
from arrows.arrow import Arrow
from arrows.sourcearrow import SourceArrow
from arrows.compositearrow import CompositeArrow, EdgeMap
from arrows.tfarrow import TfArrow, TfLambdaArrow
from arrows.std_arrows import *
from arrows.apply.interpret import interpret
from arrows.apply.propagate import propagate
from arrows.util.misc import print_one_per_line
from typing import List, Dict, MutableMapping, Union, Sequence
from overloading import overload


def gen_input_tensors(arrow: Arrow,
                      param_port_as_var=True):
    """Generate tensors corresponding to in_ports
    Args:
        arrow: Arrow of interest
        param_port_as_var: Whether parametric ports should be tf.Variable
    """
    input_tensors = []
    state = propagate(arrow)
    for in_port in arrow.in_ports():
        shape = state[in_port]['shape']
        dtype = get_port_dtype(in_port)
        if is_param_port(in_port) and param_port_as_var:
            name = "param_input_%s" % in_port.index
            var = tf.Variable(np.random.rand(*shape), name=name,
                              dtype=dtype)
            input_tensors.append(var)
        elif is_in_port(in_port):
            name = "input_%s" % in_port.index
            inp = tf.placeholder(name=name, shape=shape, dtype=dtype)
            input_tensors.append(inp)
        else:
            assert False, "Don't know how to handle %s" % in_port
    return input_tensors

TensorVarList = Union[Sequence[Tensor], Sequence[Variable]]

@overload
def conv(a: Arrow, args: TensorVarList, state) -> Sequence[Tensor]:
    assert False, "Error, no conversion for %s implemented" % a


@overload
def conv(a: BroadcastArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    # Tensorflow has automatic broadcasting so do nothing
    return args


@overload
def conv(a: InvBroadcastArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [args[0][tuple(0 for i in a.ext_shape)]]


@overload
def conv(a: AddArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.add(*args)]


@overload
def conv(a: ExpArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.exp(*args)]
    # FIXME Deal with me in a better way
    # FIXME This one is quite bad
    # with tf.name_scope("Safe_Exp"):
    #     arg = args[0]
    #     safe_arg = 40.0 * tf.tanh(arg / 40.0)
    #     return [tf.exp(safe_arg)]



@overload
def conv(a: NegArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.negative(*args)]


@overload
def conv(a: PowArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.pow(*args)]



@overload
def conv(a: LogArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.log(*args)]
    # FIXME Deal with me in a better way
    # with tf.name_scope("Safe_Log"):
    #     arg = args[0]
    #     safe_arg = arg + 1e-4
    #     return [tf.log(safe_arg)]


@overload
def conv(a: LogBaseArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    # Tensorflow has no log of arbitrary base
    # so, use log _{b}(x)=log _{k}(x)}/log _{k}(b)
    return [tf.log(args[1]) / tf.log(args[0])]


@overload
def conv(a: MulArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.multiply(*args)]
    # return [40.0 * tf.tanh(tf.multiply(*args) / 40.0)]


@overload
def conv(a: DivArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.div(*args)]
    # FIXME Deal with me in a better way
    # with tf.name_scope("Safe_Divide"):
    #     num = args[0]
    #     den = args[1]
    #     safe_den = den + 1e-4
    #     return [tf.div(num, safe_den)]

@overload
def conv(a: SinArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.sin(*args)]

@overload
def conv(a: SubArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.subtract(*args)]


@overload
def conv(a: CosArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.cos(*args)]

@overload
def conv(a: ASinArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.asin(*args)]

@overload
def conv(a: ACosArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.acos(*args)]


@overload
def conv(a: DuplArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    # TODO: Genralize to n outputs
    return [args[0] for i in range(a.num_out_ports())]


@overload
def conv(a: IdentityArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.identity(*args)]


@overload
def conv(a: InvDuplArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    # TODO: Add assert that all args are equal
    return [args[0]]

@overload
def conv(a: AddNArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    print("args")
    print_one_per_line(args)
    return [tf.add_n(args)]

@overload
def conv(a: CastArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.cast(args[0], dtype=a.to_dtype)]

@overload
def conv(a: ClipArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.clip_by_value(*args)]

@overload
def conv(a: SliceArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.slice(*args)]

@overload
def conv(a: ReshapeArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.reshape(*args)]

@overload
def conv(a: SqueezeArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.squeeze(*args)]

@overload
def conv(a: SelectArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.select(*args)]

@overload
def conv(a: FloorDivArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.floordiv(*args)]

@overload
def conv(a: AbsArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.abs(args[0])]

@overload
def conv(a: SquareArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.square(args[0])]

@overload
def conv(a: MaxArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.maximum(args[0], args[1])]

@overload
def conv(a: RankArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.rank(args[0])]

@overload
def conv(a: RangeArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.range(args[0], args[1])]

@overload
def conv(a: ReduceMeanArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.reduce_mean(args[0], reduction_indices=args[1])]

@overload
def conv(a: SourceArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    assert len(args) == 0, "Source arrow has no inputs"
    return [tf.constant(a.value)]

@overload
def conv(a: GreaterArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.greater(args[0], args[1])]

@overload
def conv(a: IfArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    #import pdb; pdb.set_trace()
    return [tf.where(*args)]

@overload
def conv(a: GatherArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.gather(*args)]

@overload
def conv(a: GatherNdArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.gather_nd(*args)]

@overload
def conv(a: ScatterNdArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.scatter_nd(*args)]

@overload
def conv(a: SparseToDenseArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.sparse_to_dense(*args, validate_indices=False)]

@overload
def conv(a: SquaredDifference, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.squared_difference(*args)]

@overload
def conv(a: TfArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    # import pdb; pdb.set_trace()
    # FIXME: Is the correspondance correct here?
    port_attr = state['port_attr']
    inp_shapes = [get_port_shape(p, port_attr) for p in a.in_ports()]
    out_shapes = [get_port_shape(p, port_attr) for p in a.out_ports()]
    with tf.name_scope("TfArrow"):
        template = a.template
        options = a.options
        r, p = template(args, inp_shapes=inp_shapes, out_shapes=out_shapes,
                        **options)
    return r

@overload
def conv(a: TfLambdaArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    with tf.name_scope(a.name):
        if 'seen_tf' in state and a.name in state['seen_tf']:
          return a.func(args, reuse=True)
        else:
          state['seen_tf'] = set([a.name])
          return a.func(args, reuse=False)

@overload
def conv(a: StackArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    return [tf.stack(args, axis=a.axis)]

@overload
def conv(a: TransposeArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    inp = args[0]
    inp = tf.transpose(inp, a.perm)
    # inp = tf.Print(inp, [inp[0, 0, 0], inp[0, 0, 1]], message="transpose")
    return [inp]

@overload
def conv(a: IgnoreInputArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    inp = args[1]
    return [inp]

@overload
def conv(a: CompositeArrow, args: TensorVarList, state) -> Sequence[Tensor]:
    assert len(args) == a.num_in_ports()
    with tf.name_scope(a.name):
        # import pdb; pdb.set_trace()
        # FIXME: A horrible horrible hack
        port_grab = state['port_grab']
        return interpret(conv, a, args, state, port_grab)


def arrow_to_graph(comp_arrow: CompositeArrow,
                   input_tensors: Sequence[Tensor],
                   port_grab: Dict[Port, Any]={}): #FIXME DANGERIOUS {}
    input_tensors_wrapped = list(map(tf.identity, input_tensors))
    port_attr = propagate(comp_arrow)
    state = {'port_attr': port_attr}
    return interpret(conv, comp_arrow, input_tensors_wrapped, state, port_grab)
