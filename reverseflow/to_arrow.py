"""Convert a tensoflow graph into an arrow"""
from typing import List, Tuple, Dict, Sequence
from tensorflow import Tensor, Operation

from arrows import (Arrow, CompositeArrow, compose_comb_modular, compose_comb)
from arrows import InPort, OutPort
from arrows.std_arrows import *

from reverseflow.util.mapping import Relation

# Mapping between op types and arrows
# Cannot use multimethods because different ops not distinguished by type
Op_To_Arrow = {'Add': conv_Add,  # type: Dict[string, Arrow]
               'Mul': conv_Mul,
               'Const': conv_Const}

def conv_Placeholder(add_op: Operation):
    return AddArrow()

def conv_Add(add_op: Operation):
    return AddArrow()

def conv_Mul(mul_op: Operation):
    return MulArrow()

def conv_Const(const_op: Operation):
    assert False

def create_arrow_from_op(op: Operation) -> Arrow:
    """Construct arrow which corresponds to op"""
    conv_op = Op_To_Arrow[op.type]
    return conv_op(op)


def arrow_from_op(op: Operation,
                  op_to_arrow: Dict[Operation, Arrow]) -> Arrow:
    """Return an arrow from a list or create one if haven't done already"""
    if op in op_to_arrow:
        arrow = op_to_arrow[op]
    else:
        arrow = create_arrow_from_op(op)
    assert len(arrow.in_ports) == len(op.inputs)
    return arrow


def update_seen(op: Operation,
                seen_tensors: Set[Tensor],
                to_see_tensors: Sequence[Tensor]) -> None:
    for tensor in op.inputs:
        if tensor not in seen_tensors:
            to_see_tensors.append(tensor)

def is_input_tensor(tensor: Tensor) -> bool:
    return tensor.op.type == 'Placeholder'


def graph_to_arrow(output_tensors: Sequence[Tensor]) -> Arrow:
    """Convert a tensorflow graph into an arrow
    Args:
        output_tensors: Tensors designated as outputs
    """
    edges = Relation()
    op_to_arrow = dict()
    comp_out_ports = []  # type: List[OutPort]
    comp_in_ports = []  # type: List[InPort]
    seen_tensors = set()
    to_see_tensors = []

    for tensor in output_tensors:
        arrow = arrow_from_op(tensor.op, op_to_arrow)
        comp_out_ports.append(arrow.out_ports[tensor.value_index])

    to_see_tensors = output_tensors[:]
    while len(to_see_tensors) > 0:
        tensor = to_see_tensors.pop()
        if is_input_tensor(tensor):
            ok()
        else:
            out_port_id = tensor.value_index
            left_arrow = arrow_from_op(tensor.op, op_to_arrow)
            update_seen(tensor.op, seen_tensors, to_see_tensors)

        for rec_op in tensor.consumers():
            for i, input_tensor in enumerate(rec_op.inputs):
                if tensor == input_tensor:
                    in_port_id = i
                    right_arrow = arrow_from_op(tensor.op, op_to_arrow)
                    if is_input_tensor(tensor):
                        comp_in_ports.append(right_arrow.in_ports[in_port_id])
                    else:
                        edges.add(left_arrow.out_ports[out_port_id],
                                  right_arrow.in_ports[in_port_id])

    return CompositeArrow(edges=edges,
                          in_ports=...,
                          out_ports=comp_out_ports)
