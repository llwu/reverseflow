"""Decode an arrow into a tensoflow graph"""
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow, EdgeMap
from arrows.primitive.math_arrows import *
from arrows.primitive.control_flow import *
from arrows.primitive.cast_arrows import *
from arrows.primitive.constant import *
from arrows.port_attributes import *
from pqdict import pqdict
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

    # TODO: Unify
    arrow_colors[comp_arrow] = comp_arrow.num_out_ports()
    return arrow_colors


def gen_arrow_inputs(comp_arrow: CompositeArrow,
                     inputs: List,
                     arrow_colors):
    # Store a map from an arrow to its inputs
    # Use a dict because no guarantee we'll create input tensors in order
    arrow_inputs = dict()  # type: Dict[Arrow, MutableMapping[int, Any]]
    for sub_arrow in comp_arrow.get_all_arrows():
        arrow_inputs[sub_arrow] = dict()

    # Decrement priority of every arrow connected to the input
    for i, input_value in enumerate(inputs):
        for in_port in comp_arrow.edges[comp_arrow.in_ports()[i]]:
            # in_port = comp_arrow.inner_in_ports()[i]
            sub_arrow = in_port.arrow
            arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
            arrow_inputs[sub_arrow][in_port.index] = input_value

    return arrow_inputs


def inner_interpret(conv: Callable,
                    comp_arrow: CompositeArrow,
                    inputs: List,
                    arrow_colors: MutableMapping[Arrow, int],
                    arrow_inputs: Sequence,
                    state: Dict,
                    port_grab: Dict[Port, Any]):
    """Convert an comp_arrow to a tensorflow graph and add to graph"""
    assert len(inputs) == comp_arrow.num_in_ports(), "wrong # inputs"

    emit_list = []
    while len(arrow_colors) > 0:
        # print_arrow_colors(arrow_colors)
        # print("Converting ", sub_arrow.name)
        sub_arrow, priority = arrow_colors.popitem()
        if sub_arrow is not comp_arrow:
            assert priority == 0, "Must resolve {} more inputs to {} first".format(priority, sub_arrow)
            # inputs = [arrow_inputs[sub_arrow][i] for i in range(len(arrow_inputs[sub_arrow]))]
            inputs = [arrow_inputs[sub_arrow][i] for i in sorted(arrow_inputs[sub_arrow].keys())]
            outputs = conv(sub_arrow, inputs, state)

            assert len(outputs) == len(sub_arrow.out_ports()), "diff num outputs"

            # Decrement the priority of each subarrow connected to this arrow
            # Unless of course it is connected to the outside word
            for i, out_port in enumerate(sub_arrow.out_ports()):
                neigh_in_ports = comp_arrow.neigh_in_ports(out_port)
                for neigh_in_port in neigh_in_ports:
                    neigh_arrow = neigh_in_port.arrow
                    arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1
                    arrow_inputs[neigh_arrow][neigh_in_port.index] = outputs[i]

    # Extract some port, kind of a hack
    for port in port_grab:
        if port.arrow in arrow_inputs:
            if port.index in arrow_inputs[port.arrow]:
                port_grab[port] = arrow_inputs[port.arrow][port.index]

    outputs_dict = arrow_inputs[comp_arrow]
    out_port_indices = sorted(list(outputs_dict.keys()))
    return [outputs_dict[i] for i in out_port_indices]

def interpret(conv: Callable,
              comp_arrow: CompositeArrow,
              inputs: List,
              state: Dict,
              port_grab: Dict[Port, Any]) -> List:
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
    # FIXME: A horrible horrible hack to pass through port_grab to conv(a:CompositeArrow)
    state['port_grab'] = port_grab
    result = inner_interpret(conv,
                             comp_arrow,
                             inputs,
                             arrow_colors,
                             arrow_inputs,
                             state,
                             port_grab)
    return result
