"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor, Graph
from pqdict import pqdict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from typing import Tuple, List, Dict, MutableMapping
from collections import OrderedDict
from overloading import overload

def print_arrow_colors(arrow_colors):
    for (arr, pr) in arrow_colors.items():
        print(arr.name, ": ", pr)


def valid(sub_arrow, arrow_tensors):
    input_tensors = arrow_tensors[sub_arrow]
    # TODO: Check that the number of inputs created is same as num inputs to
    # arrow
    return True

def default_add(arrow_tensors: Dict[Arrow, MutableMapping[int, tf.Tensor]],
                sub_arrow: Arrow, index: int, input_tensor: Tensor) -> None:
    if sub_arrow in arrow_tensors:
        arrow_tensors[sub_arrow][index] = input_tensor
    else:
        arrow_tensors[sub_arrow] = OrderedDict({index: input_tensor})


@overload
def conv(a: AddArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.add(*args)]

@overload
def conv(a: SubArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.sub(*args)]

@overload
def conv(a: NegArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.neg(*args)]

# @overload
# def conv(a: ExpArrow, args: List[Tensor]) -> List[Tensor]:
#     import pdb; pdb.set_trace()
#
#     return [tf.exp(args[0])]
#
# @overload
# def conv(a: LogArrow, args: List[Tensor]) -> List[Tensor]:
#     import pdb; pdb.set_trace()
#
#     return [tf.log(*args)]

@overload
def conv(a: MulArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.mul(*args)]

@overload
def conv(a: DivArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.mul(*args)]

@overload
def conv(a: DuplArrow, args: List[Tensor]) -> List[Tensor]:
    # TODO: Genralize to n outputs
    return [args[0] for i in range(a.n_out_ports)]


@overload
def conv(a: CompositeArrow, args: List[Tensor]) -> List[Tensor]:
    graph = tf.get_default_graph()
    assert len(args) == a.n_in_ports

    input_tensors = args
    arrow_colors = pqdict()
    arrow_tensors = dict()  # type: Dict[Arrow, MutableMapping[int, tf.Tensor]]

    for sub_arrow in a.get_sub_arrows():
        arrow_colors[sub_arrow] = sub_arrow.num_in_ports()

    for i, input_tensor in enumerate(args):
        in_port = a.inner_in_ports[i]
        sub_arrow = in_port.arrow
        arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
        default_add(arrow_tensors, sub_arrow, in_port.index, input_tensor)

    # import pdb; pdb.set_trace()

    new_graph, input_tensors, output_tensors = arrow_to_graph(a,
                                                              input_tensors,
                                                              arrow_colors,
                                                              arrow_tensors,
                                                              graph)
    # import pdb; pdb.set_trace()

    return output_tensors


def arrow_to_new_graph(comp_arrow: CompositeArrow) -> Graph:
    """Create new graph and convert comp_arrow into it"""
    graph = tf.Graph()
    # create a tensor for each in_port to the composition
    # decrement priority for each arrow connected to inputs

    arrow_colors = pqdict()

    for sub_arrow in comp_arrow.get_sub_arrows():
        arrow_colors[sub_arrow] = sub_arrow.num_in_ports()

    # Store a map from an arrow to its inputs
    # Use a dict because no guarantee we'll create input tensors in order
    arrow_tensors = dict()  # type: Dict[Arrow, MutableMapping[int, tf.Tensor]]

    input_tensors = []
    for in_port in comp_arrow.inner_in_ports:
        sub_arrow = in_port.arrow
        assert sub_arrow in arrow_colors
        arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
        input_tensor = tf.placeholder(dtype='float32')  # FIXME: Generalize
        input_tensors.append(input_tensor)
        default_add(arrow_tensors, sub_arrow, in_port.index, input_tensor)

    return arrow_to_graph(comp_arrow, input_tensors, arrow_colors, arrow_tensors, graph)


def arrow_to_graph(comp_arrow: CompositeArrow,
                   input_tensors: List[Tensor],
                   arrow_colors,
                   arrow_tensors,
                   graph: Graph) -> Tuple[Graph, List[Tensor], List[Tensor]]:
    """Convert an comp_arrow to a tensorflow graph and add to graph"""
    with graph.as_default():
        # A priority queue for each sub_arrow
        # priority is the number of inputs it has which have already been seen
        # seen inputs are inputs to the composition, or outputs of arrows that
        # have already been converted into
        while len(arrow_colors) > 0:
            print_arrow_colors(arrow_colors)
            sub_arrow, priority = arrow_colors.popitem()
            print("Converting ", sub_arrow.name)
            assert priority == 0, "Must resolve all inputs to sub_arrow first"
            # If its a composite we still want to wait until all its inputs are ready
            # Then we should call this function with the same arguments
            assert valid(sub_arrow, arrow_tensors)

            inputs = list(arrow_tensors[sub_arrow].values())
            # import pdb; pdb.set_trace()
            print(type(sub_arrow), type(inputs))
            outputs = conv(sub_arrow, inputs)
            assert len(outputs) == len(sub_arrow.out_ports), "diff num outputs"

            # Decrement the priority of each subarrow connected to this arrow
            # Unless of course it is connected to the outside word
            for i, out_port in enumerate(sub_arrow.out_ports):
                # FIXME: this is linear search, encapsulate
                if out_port not in comp_arrow.inner_out_ports:
                    neigh_port = comp_arrow.neigh_in_port(out_port)
                    neigh_arrow = neigh_port.arrow
                    if neigh_arrow is not comp_arrow:
                        assert neigh_arrow in arrow_colors
                        arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1
                        default_add(arrow_tensors, neigh_arrow, neigh_port.index,
                                    outputs[i])

        # The output tensors are
        output_tensors = []
        for out_port in comp_arrow.inner_out_ports:
            output_tensor = arrow_tensors[out_port.arrow][out_port.index]
            output_tensors.append(output_tensors)

        return graph, input_tensors, output_tensors
