"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor, Graph, Variable
from pqdict import pqdict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from reverseflow.arrows.primitive.cast_arrows import *
from typing import Tuple, List, Dict, MutableMapping, Union
from collections import OrderedDict
from overloading import overload

TensorVarList = Union[List[Tensor], List[Variable]]

def print_arrow_colors(arrow_colors):
    for (arr, pr) in arrow_colors.items():
        print(arr.name, ": ", pr)


def default_add(arrow_tensors: Dict[Arrow, MutableMapping[int, tf.Tensor]],
                sub_arrow: Arrow, index: int, input_tensor: Tensor) -> None:
    if sub_arrow in arrow_tensors:
        arrow_tensors[sub_arrow][index] = input_tensor
    else:
        arrow_tensors[sub_arrow] = OrderedDict({index: input_tensor})


@overload
def conv(a: Arrow, args: TensorVarList) -> List[Tensor]:
    assert False, "Error, no conversion for %s implemented" % a.name


@overload
def conv(a: AddArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.add(*args)]


@overload
def conv(a: SubArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.sub(*args)]


@overload
def conv(a: NegArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.neg(*args)]


@overload
def conv(a: PowArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.pow(*args)]


@overload
def conv(a: ExpArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.exp(*args)]


@overload
def conv(a: LogArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.log(*args)]


@overload
def conv(a: LogBaseArrow, args: TensorVarList) -> List[Tensor]:
    # Tensorflow has no log of arbitrary base
    # so, use log _{b}(x)=log _{k}(x)}/log _{k}(b)
    return [tf.log(args[1]) / tf.log(args[0])]


@overload
def conv(a: MulArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.mul(*args)]


@overload
def conv(a: DivArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.div(*args)]


@overload
def conv(a: DuplArrow, args: TensorVarList) -> List[Tensor]:
    # TODO: Genralize to n outputs
    return [args[0] for i in range(a.n_out_ports)]

@overload
def conv(a: AddNArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.add_n(args)]

@overload
def conv(a: CastArrow, args: TensorVarList) -> List[Tensor]:
    return [tf.cast(args[0], dtype=a.to_dtype)]

@overload
def conv(a: CompositeArrow, args: TensorVarList) -> List[Tensor]:
    graph = tf.get_default_graph()
    assert len(args) == a.n_in_ports
    arrow_colors, arrow_tensors = inner_convert(a, args)
    result = arrow_to_graph(a,
                            args,
                            arrow_colors,
                            arrow_tensors,
                            graph)
    return result['output_tensors']


def inner_convert(comp_arrow: CompositeArrow, input_tensors: List[Tensor]):
    # A priority queue for each sub_arrow
    # priority is the number of inputs it has which have already been seen
    # seen inputs are inputs to the composition, or outputs of arrows that
    # have already been converted into
    # create a tensor for each in_port to the composition
    # decrement priority for each arrow connected to inputs
    arrow_colors = pqdict()  # type: MutableMapping[Arrow, int]

    # Store a map from an arrow to its inputs
    # Use a dict because no guarantee we'll create input tensors in order
    arrow_tensors = dict()  # type: Dict[Arrow, MutableMapping[int, tf.Tensor]]

    for sub_arrow in comp_arrow.get_sub_arrows():
        if not sub_arrow.is_source():
            arrow_colors[sub_arrow] = sub_arrow.num_in_ports()

    for i, input_tensor in enumerate(input_tensors):
        in_port = comp_arrow.inner_in_ports()[i]
        sub_arrow = in_port.arrow
        arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
        default_add(arrow_tensors, sub_arrow, in_port.index, input_tensor)

    # Create tensor for each source
    # TODO: Should we turn sources into variables always?
    # FIXME: DRY
    for sub_arrow in comp_arrow.get_sub_arrows():
        if sub_arrow.is_source():
            graph = tf.get_default_graph()
            value = tf.Variable(sub_arrow.value, name="Jeremiah")
            in_port = comp_arrow.edges.fwd(sub_arrow.out_ports[0])
            arrow_colors[in_port.arrow] = arrow_colors[in_port.arrow] - 1
            default_add(arrow_tensors, in_port.arrow, in_port.index, value)


    return arrow_colors, arrow_tensors


def arrow_to_new_graph(comp_arrow: CompositeArrow,
                       input_tensors: List[Tensor],
                       param_tensors: List[Tensor],
                       graph: Graph):
    """Create new graph and convert comp_arrow into it"""
    arrow_colors, arrow_tensors = inner_convert(comp_arrow, input_tensors)
    return arrow_to_graph(comp_arrow, input_tensors, param_tensors,
                          arrow_colors, arrow_tensors, graph)


def arrow_to_graph(comp_arrow: CompositeArrow,
                   input_tensors: List[Tensor],
                   param_tensors: List[Tensor],
                   arrow_colors: MutableMapping[Arrow, int],
                   arrow_tensors: Dict[Arrow, MutableMapping[int, tf.Tensor]],
                   graph: Graph) -> Tuple[Graph, List[Tensor], List[Tensor]]:
    """Convert an comp_arrow to a tensorflow graph and add to graph"""
    assert len(input_tensors) == comp_arrow.n_in_ports, "wrong # inputs"
    assert len(param_tensors) == comp_arrow.n_param_ports, "wron # param inputs"

    # FIXMEL Horrible Hack
    output_tensors_dict =  OrderedDict()

    with graph.as_default():
        with tf.name_scope(comp_arrow.name):
            while len(arrow_colors) > 0:
                print_arrow_colors(arrow_colors)
                sub_arrow, priority = arrow_colors.popitem()
                print("Converting ", sub_arrow.name)
                assert priority == 0, "Must resolve all inputs to sub_arrow first"
                # TODO: Check that the number of inputs created is same as num inputs to

                inputs = list(arrow_tensors[sub_arrow].values())
                # import pdb; pdb.set_trace()
                print(type(sub_arrow), type(inputs))

                outputs = conv(sub_arrow, inputs)
                assert len(outputs) == len(sub_arrow.out_ports), "diff num outputs"

                # Decrement the priority of each subarrow connected to this arrow
                # Unless of course it is connected to the outside word
                for i, out_port in enumerate(sub_arrow.out_ports):
                    # FIXME: this is linear search, encapsulate
                    if out_port not in comp_arrow.inner_out_ports():
                        neigh_port = comp_arrow.neigh_in_port(out_port)
                        neigh_arrow = neigh_port.arrow
                        if neigh_arrow is not comp_arrow:
                            assert neigh_arrow in arrow_colors
                            arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1
                            default_add(arrow_tensors, neigh_arrow, neigh_port.index,
                                        outputs[i])
                    else:
                        # FIXME: Horrible hack
                        output_tensor = outputs[i]
                        j = 0
                        for i, p in enumerate(comp_arrow.inner_out_ports()):
                            if out_port == p:
                                break
                            else:
                                j = j + 1
                        output_tensors_dict[j] = output_tensor

            # The output tensors are
            output_tensors = []
            # import pdb; pdb.set_trace()
            # FIXME
            for out_port in comp_arrow.inner_out_ports():
                output_tensor = arrow_tensors[out_port.arrow][out_port.index]
                output_tensors.append(output_tensor)

            error_tensors = []
            for error_port in comp_arrow.inner_error_ports():
                error_tensor = arrow_tensors[error_port.arrow][error_port.index]
                error_tensors.append(error_tensor)


            return {'graph': graph,
                    'input_tensors': input_tensors,
                    'output_tensors': list(output_tensors_dict.values()),
                    'param_tensors': [],
                    'error_tensors': error_tensors}
