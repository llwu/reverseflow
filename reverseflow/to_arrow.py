"""Convert a tensoflow graph into an arrow"""
from arrows import (Arrow, CompositeArrow,)
from arrows import InPort, OutPort
from arrows.std_arrows import *
from reverseflow.util.mapping import Relation

from tensorflow import Tensor, Operation
import tensorflow as tf
from typing import List, Tuple, Dict, Sequence


def get_const_op_value(const_op: Operation):
    """Get the constant output of a Const op as a numpy array/number"""
    sess = tf.Session()
    outputs = const_op.outputs
    assert len(outputs) == 1
    output = outputs[0]
    val = output.eval(session=sess)
    sess.close()
    return val


def conv_Add(add_op: Operation):
    return AddArrow()


def conv_Mul(mul_op: Operation):
    return MulArrow()


def conv_Sin(sin_op: Operation):
    return SinArrow()


def conv_Cos(sin_op: Operation):
    return CosArrow()


def conv_Const(const_op: Operation):
    value = get_const_op_value(const_op)
    return SourceArrow(value=value)

# Mapping between op types and arrows
# Cannot use multimethods because different ops not distinguished by type
Op_To_Arrow = {'Add': conv_Add,  # type: Dict[string, Arrow]
               'Mul': conv_Mul,
               'Sin': conv_Sin,
               'Cos': conv_Cos,
               'Const': conv_Const}

def arrow_from_op(op: Operation,
                  op_to_arrow: Dict[Operation, Arrow]) -> Arrow:
    """Return an arrow from a list or create one if haven't done already"""
    if op in op_to_arrow:
        arrow = op_to_arrow[op]
    else:
        conv_op = Op_To_Arrow[op.type]
        arrow = conv_op(op)
    assert len(arrow.get_in_ports()) == len(op.inputs)
    return arrow


def update_seen(op: Operation,
                seen_tensors: Set[Tensor],
                to_see_tensors: Sequence[Tensor]) -> None:
    for tensor in op.inputs:
        if tensor not in seen_tensors:
            to_see_tensors.append(tensor)

def is_input_tensor(tensor: Tensor) -> bool:
    return tensor.op.type == 'Placeholder'


def graph_to_arrow(output_tensors: Sequence[Tensor],
                   name:str=None) -> Arrow:
    """Convert a tensorflow graph into an arrow.
    Assume inputs are 'Placeholder' tensors
    Args:
        output_tensors: Tensors designated as outputs
        name: Name of the composite arrow
    Returns:
        A 'CompositeArrow' equivalent to graph which computes 'output_tensors'
    """
    edges = Relation()
    op_to_arrow = dict()
    comp_out_ports = []  # type: List[OutPort]
    comp_in_ports = []  # type: List[InPort]
    seen_tensors = set()
    to_see_tensors = []

    for tensor in output_tensors:
        arrow = arrow_from_op(tensor.op, op_to_arrow)
        comp_out_ports.append(arrow.get_out_ports()[tensor.value_index])

    to_see_tensors = output_tensors[:]
    while len(to_see_tensors) > 0:
        tensor = to_see_tensors.pop()
        if not is_input_tensor(tensor):
            out_port_id = tensor.value_index
            left_arrow = arrow_from_op(tensor.op, op_to_arrow)
            update_seen(tensor.op, seen_tensors, to_see_tensors)

        for rec_op in tensor.consumers():
            for i, input_tensor in enumerate(rec_op.inputs):
                if tensor == input_tensor:
                    in_port_id = i
                    right_arrow = arrow_from_op(rec_op, op_to_arrow)
                    if is_input_tensor(tensor):
                        comp_in_ports.append(right_arrow.get_in_ports()[in_port_id])
                    else:
                        edges.add(left_arrow.get_out_ports()[out_port_id],
                                  right_arrow.get_in_ports()[in_port_id])

    return CompositeArrow(edges=edges,
                          in_ports=comp_in_ports,
                          out_ports=comp_out_ports,
                          name=name)
