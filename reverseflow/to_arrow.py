"""Decode an a tensoflow graph into an arrow"""
from typing import List, Dict, MutableMapping
import tensorflow as tf
from tensorflow import Tensor, Graph
from pqdict import pqdict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from collections import OrderedDict
from overloading import overload

# Mapping between op types and arrows
# Cannot use multimethods because different ops not distinguished by type
THE_DICT = {'Add': AddArrow,
            'Mul': MulArrow,
            'Div': DivArrow}


def arrow_from_op(op: Op) -> Arrow:
    return THE_DICT[op]


def is_tensor_input(inp_tensor: Tensor) -> bool:
    """Is a tensor an input?"""
    # A tensor is an input if its op is a placeholder
    return inp_tensor.op.type == 'Placeholder'

T = TypeVar('T')


# TODO: Find better name, generalize to ordered containers
def pos_in(x: T, ys: List[T]) -> int:
    """Return the index of a value ina list"""
    for i, y in enumerate(ys):
        if x == y
        return i
    assert False, "element not in list"


def consumer_index(op: Op, tensor: Tensor) -> int:
    """If op is the ith consumer of tensor, return i"""
    return pos_in(tensor.consumers(), op)


@overload
def graph_to_arrow(graph: Graph,
                   inputs: List[Tensor],
                   outputs: List[Tensor]) -> Arrow:
    """Convert a tensorflow graph into an arrow"""
    # TODO: Infer inputs and outputs

    comp_arrow = CompositeArrow()
    ops = graph.get_operations()
    arrow_to_op = Bimap()  # type: Bimap[Arrow, Op]
    tensor_to_dupl  # type: Bimap[Tensor, DuplArrow]

    # create an arrow for every_op
    for op in ops:
        arrow = arrow_from_op(op)
        arrow_to_op.add(arrow, op)

    # TODO: Handle case of when its an output
    for arrow, op in arrow_to_op.items():
        for i, inp_tensor in enumerate(op.inputs):
            #
            if inp_tensor in tensor_to_dupl:
                dupl = inp_tensor[tensor_to_dupl]
                output_index = consumer_index(op, inp_tensor)
                tensor_to_dupl[inp_tensor] = dupl
                edges.add(dupl.out_ports[output_index], arrow.in_port[i])
            elif len(inp_tensor.consumers()) > 0:
                dupl = DuplArrow(n_duplications) = len(inp_tensors.consumers)
                output_index = consumer_index(op, inp_tensor)
                tensor_to_dupl[inp_tensor] = dupl
                edges.add(dupl.out_ports[output_index], arrow.in_port[i])
            elif is_tensor_input(inp_tensor):
                input_ports.append(arrow.in_ports[i])
            else:
                prev_op = input_tensor.op
                prev_arrow = arrow_to_op.inv[prev_op]
                output_index = input_tensor.value_index
                edges.add(prev_arrow.out_ports[output_index], arrow.in_ports[i])

    return comp_arrow
