"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor, Graph
from pqdict import pqdict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from typing import List, Dict, MutableMapping
from collections import OrderedDict
from overloading import overload


def valid(sub_arrow, arrow_tensors):
    input_tensors = arrow_tensors[sub_arrow]
    # TODO: Check that the number of inputs created is same as num inputs to
    # arrow
    return True


@overload
def conv(a: AddArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.add(*args)]


@overload
def conv(a: MulArrow, args: List[Tensor]) -> List[Tensor]:
    return [tf.mul(*args)]


@overload
def conv(a: DuplArrow, args: List[Tensor]) -> List[Tensor]:
    # TODO: Genralize to n outputs
    return [args[0], args[0]]


def default_add(arrow_tensors: Dict[Arrow, MutableMapping[int, tf.Tensor]],
                sub_arrow: Arrow, index: int, input_tensor: Tensor) -> None:
    if sub_arrow in arrow_tensors:
        arrow_tensors[sub_arrow][index] = input_tensor
    else:
        arrow_tensors[sub_arrow] = OrderedDict({index: input_tensor})


def print_arrow_colors(arrow_colors):
    for (arr, pr) in arrow_colors.items():
        print(arr.name, ": ", pr)


@overload
def arrow_to_graph(comp_arrow: CompositeArrow) -> Graph:
    """Convert an comp_arrow to a tensorflow graph"""

    graph = tf.Graph()  # type: Graph
    with graph.as_default():
        # A priority queue for each sub_arrow
        # priority is the number of inputs it has which have already been seen
        # seen inputs are inputs to the composition, or outputs of arrows that
        # have already been converted into tensorfow
        arrow_colors = pqdict()
        # import pdb; pdb.set_trace()
        for sub_arrow in comp_arrow.get_sub_arrows():
            arrow_colors[sub_arrow] = sub_arrow.num_in_ports()

        print_arrow_colors(arrow_colors)

        # Store a map from an arrow to its inputs
        # Use a dict because no guarantee we'll create input tensors in order
        arrow_tensors = dict()  # type: Dict[Arrow, MutableMapping[int, tf.Tensor]]

        # create a tensor for each in_port to the composition
        # decrement priority for each arrow connected to inputs
        for in_port in comp_arrow.in_ports:
            sub_arrow = in_port.arrow
            assert sub_arrow in arrow_colors
            arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
            input_tensor = tf.placeholder(dtype='float32')  # FIXME: Generalize
            default_add(arrow_tensors, sub_arrow, in_port.index, input_tensor)

        while len(arrow_colors) > 0:
            print_arrow_colors(arrow_colors)
            sub_arrow, priority = arrow_colors.popitem()
            print("Converting ", sub_arrow.name)
            print_arrow_colors(arrow_colors)
            assert priority == 0, "Must resolve all inputs to sub_arrow first"
            assert sub_arrow.is_primitive(), "Cannot convert unflat arrow"
            assert valid(sub_arrow, arrow_tensors)

            inputs = list(arrow_tensors[sub_arrow].values())
            # import pdb; pdb.set_trace()
            outputs = conv(sub_arrow, inputs)
            assert len(outputs) == len(sub_arrow.out_ports), "diff num outputs"

            for i, out_port in enumerate(sub_arrow.out_ports):
                # FIXME: this is linear search, encapsulate
                if out_port not in comp_arrow.out_ports:
                    neigh_port = comp_arrow.neigh_in_port(out_port)
                    neigh_arrow = neigh_port.arrow
                    if neigh_arrow is not comp_arrow:
                        assert neigh_arrow in arrow_colors
                        arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1
                        default_add(arrow_tensors, neigh_arrow, neigh_port.index,
                                    outputs[i])

        return graph
