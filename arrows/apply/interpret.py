"""Decode an arrow into a tensoflow graph"""
from pqdict import pqdict
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow, EdgeMap
from arrows.primitive.math_arrows import *
from arrows.primitive.control_flow_arrows import *
from arrows.primitive.cast_arrows import *
from arrows.primitive.constant import *
from typing import List, Dict, MutableMapping, Union, Callable
from collections import OrderedDict


def print_arrow_colors(arrow_colors):
    for (arr, pr) in arrow_colors.items():
        print(arr.name, ": ", pr)


def gen_arrow_colors(comp_arrow: CompositeArrow):
    """
    Interpret a composite arrow on some inputs
    Args:
        comp_arrow: Composite Arrow
    Returns:
        arrow_colors: Priority Queue of arrows
    """
    # priority is the number of inputs each arrrow has which have been 'seen'
    # seen inputs are inputs to the composition, or outputs of arrows that
    # have already been converted into
    arrow_colors = pqdict()  # type: MutableMapping[Arrow, int]
    for sub_arrow in comp_arrow.get_sub_arrows():
        arrow_colors[sub_arrow] = sub_arrow.num_in_ports()
    return arrow_colors


def gen_arrow_inputs(comp_arrow: CompositeArrow,
                     inputs: List,
                     arrow_colors):
    # Store a map from an arrow to its inputs
    # Use a dict because no guarantee we'll create input tensors in order
    arrow_inputs = dict()  # type: Dict[Arrow, MutableMapping[int, tf.Tensor]]
    for sub_arrow in comp_arrow.get_sub_arrows():
        arrow_inputs[sub_arrow] = dict()

    # Decrement priority of every arrow connected to the input
    for i, input_value in enumerate(inputs):
        in_port = comp_arrow.inner_in_ports()[i]
        sub_arrow = in_port.arrow
        arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
        arrow_inputs[sub_arrow][in_port.index] = input_value

    return arrow_inputs


def inner_interpret(conv: Callable,
                    comp_arrow: CompositeArrow,
                    inputs: List,
                    arrow_colors: MutableMapping[Arrow, int],
                    arrow_inputs):
    """Convert an comp_arrow to a tensorflow graph and add to graph"""
    assert len(inputs) == comp_arrow.num_in_ports(), "wrong # inputs"

    output_tensors_dict = dict()
    while len(arrow_colors) > 0:
        # print_arrow_colors(arrow_colors)
        # print("Converting ", sub_arrow.name)
        sub_arrow, priority = arrow_colors.popitem()
        assert priority == 0, "Must resolve all inputs to sub_arrow first"

        inputs = list(arrow_inputs[sub_arrow].values())
        outputs = conv(sub_arrow, inputs)

        assert len(outputs) == len(sub_arrow.out_ports), "diff num outputs"

        # Decrement the priority of each subarrow connected to this arrow
        # Unless of course it is connected to the outside word
        for i, out_port in enumerate(sub_arrow.out_ports):
            neigh_in_ports = comp_arrow.neigh_in_ports(out_port)
            for neigh_in_port in neigh_in_ports:
                neigh_arrow = neigh_in_port.arrow
                arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1
                arrow_inputs[neigh_arrow][neigh_in_port.index] = outputs[i]

            if out_port in comp_arrow.inner_out_ports():
                output_tensor = outputs[i]
                j = 0
                for k, p in enumerate(comp_arrow.inner_out_ports()):
                    if out_port == p:
                        j = k
                output_tensors_dict[j] = output_tensor

    final_outputs = []
    for i in range(len(output_tensors_dict)):
        final_outputs.append(output_tensors_dict[i])
    return final_outputs


def interpret(conv: Callable,
              comp_arrow: CompositeArrow,
              inputs: List) -> List:
    """
    Interpret a composite arrow on some inputs
    Args:
        conv:
        comp_arrow: Composite Arrow to execute
        inputs: list of inputs to composite arrow
    Returns:
        List of outputs
    """
    arrow_colors = gen_arrow_colors(comp_arrow)
    arrow_inputs = gen_arrow_inputs(comp_arrow, inputs, arrow_colors)
    return inner_interpret(conv,
                           comp_arrow,
                           inputs,
                           arrow_colors,
                           arrow_inputs)
