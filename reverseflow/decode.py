"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from tensorflow import Tensor
from pqdict import pqdict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from typing import List, Dict, MutableMapping
from collections import OrderedDict
from reverseflow.util.mm import multimethod

def valid(sub_arrow, arrow_tensors):
    input_tensors = arrow_tensors[sub_arrow]
    # TODO: Check that the number of inputs created is same as num inputs to arrow
    # rrow
    return True


@multimethod
def conv(a: AddArrow, args) -> List[Tensor]:
    tf.add(*args)


@multimethod
def conv(a: DuplArrow, args) -> List[Tensor]:
    # TODO: Genralize to n outputs
    return (args[0], args[0])


def arrow_to_graph(arrow: Arrow) -> tf.Graph:
    """Convert an arrow to a tensorflow graph"""
    graph = tf.Graph()
    with graph.as_default():
        # A priority queue for each sub_arrow
        # priority is the number of inputs it has which have already been seen
        # seen inputs are inputs to the composition, or outputs of arrows that
        # have already been converted into tensorfow
        arrow_colors = pqdict()
        for sub_arrow in arrow.get_sub_arrows():
            arrow_colors[sub_arrow] = sub_arrow.num_in_ports()

        # Store a map from an arrow to its inputs
        # Use a dict because no guarantee we'll create input tensors in order
        arrow_tensors = dict()  # type: MutableMapping[Arrow, Dict[int, tf.Tensor]]

        # create a tensor for each inport to the composition
        # decrement priority for each arrow connected to inputs
        for out_port in arrow.get_boundary_outports():
            in_port = arrow.neigh_inport(sub_arrow)
            sub_arrow = arrow.proj_sub_arrow(out_port)
            num_seen_inputs = arrow_colors[sub_arrow]
            arrow_colors[sub_arrow] = num_seen_inputs - 1
            input_tensor = tf.placeholder(dtype='float32')  # FIXME: Generalize
            arrow_tensors[sub_arrow][in_port.id] = input_tensor

        while len(arrow_colors) > 0:
            sub_arrow, priority = arrow_colors.popitem()
            assert priority == 0, "All inputs to subarrow should be resolved first"
            assert sub_arrow.is_primitive(), "Convert unflat arrows unimplmented"
            assert valid(sub_arrow, arrow_tensors)

            inputs = arrow_tensors[sub_arrow].values()
            outputs = conv(sub_arrow, inputs)

            for out_port in sub_arrow.out_ports:
                neigh_arrow = arrow.neigh_inport(out_port).arrow
                if neigh_arrow is not arrow:
                    assert neigh_arrow in arrow_colors
                    arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1

        return graph
