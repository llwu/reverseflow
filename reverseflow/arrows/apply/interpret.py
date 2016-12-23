"""Decode an arrow into a tensoflow graph"""
from pqdict import pqdict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from reverseflow.arrows.primitive.cast_arrows import *
from reverseflow.arrows.primitive.constant import *
from typing import List, Dict, MutableMapping, Union, Callable
from collections import OrderedDict


def print_arrow_colors(arrow_colors):
    for (arr, pr) in arrow_colors.items():
        print(arr.name, ": ", pr)


def default_add(arrow_inputs: Dict[Arrow, MutableMapping],
                sub_arrow: Arrow,
                index: int,
                input_value) -> None:
    if sub_arrow in arrow_inputs:
        arrow_inputs[sub_arrow][index] = input_value
    else:
        arrow_inputs[sub_arrow] = OrderedDict({index: input_value})


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

    # Decrement priority of every arrow connected to the input
    for i, input_value in enumerate(inputs):
        in_port = comp_arrow.inner_in_ports()[i]
        sub_arrow = in_port.arrow
        arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
        default_add(arrow_inputs, sub_arrow, in_port.index, input_value)

    return arrow_inputs


def inner_interpret(conv: Callable,
                    comp_arrow: CompositeArrow,
                    inputs: List,
                    arrow_colors: MutableMapping[Arrow, int],
                    arrow_inputs):
    """Convert an comp_arrow to a tensorflow graph and add to graph"""
    assert len(inputs) == comp_arrow.num_in_ports(), "wrong # inputs"

    # FIXMEL Horrible Hack
    output_tensors_dict = OrderedDict()
    while len(arrow_colors) > 0:
        # print_arrow_colors(arrow_colors)
        sub_arrow, priority = arrow_colors.popitem()
        # print("Converting ", sub_arrow.name)
        assert priority == 0, "Must resolve all inputs to sub_arrow first"
        # TODO: Check that the number of inputs created is same as num inputs to

        inputs = list(arrow_inputs[sub_arrow].values())
        # import pdb; pdb.set_trace()
        # print(type(sub_arrow), type(inputs))

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
                    default_add(arrow_inputs, neigh_arrow, neigh_port.index,
                                outputs[i])
            else:
                # If connected to outside world
                # FIXME: Horrible hack
                output_tensor = outputs[i]
                j = 0
                for i, p in enumerate(comp_arrow.inner_out_ports()):
                    if out_port == p:
                        j = i
   #                     break
   #                 else:
   #                     j = j + 1
                output_tensors_dict[j] = output_tensor

    return list(output_tensors_dict.values())


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
