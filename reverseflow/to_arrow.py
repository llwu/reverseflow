"""Convert a tensoflow graph into an arrow"""
from arrows import (Arrow, CompositeArrow,)
from arrows import InPort, OutPort
from arrows.primitive.control_flow import BroadcastArrow
from arrows.port_attributes import make_in_port, make_out_port, set_port_shape
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


# def broadcast_wrap(arr: Arrow):
#     return BroadcastArithArrow(arr)

def broadcast_wrap(x):
    return x

#FIXME: DRY
def conv_Add(add_op: Operation):
    return broadcast_wrap(AddArrow())

def conv_Sub(sub_op: Operation):
    return broadcast_wrap(SubArrow())

def conv_AddN(addm_op: Operation):
    return AddNArrow(len(addm_op.inputs))


def conv_Const(const_op: Operation):
    value = get_const_op_value(const_op)
    # c = CompositeArrow(name="BroadCastedSource")
    # out_port = c.add_port()
    # make_out_port(out_port)
    # src = SourceArrow(value=value)
    # bc = BroadcastArrow()
    # c.add_edge(src.out_port(0), bc.in_port(0))
    # c.add_edge(bc.out_port(0), c.out_port(0))
    # assert c.is_wired_correctly()
    # return c
    return SourceArrow(value=value)


def conv_Cos(sin_op: Operation):
    return CosArrow()


def conv_Exp(exp_op: Operation):
    return ExpArrow()


def conv_Gather(gather_op: Operation):
    return GatherArrow()


def conv_Mul(mul_op: Operation):
    return broadcast_wrap(MulArrow())


def conv_Neg(neg_op: Operation):
    return NegArrow()


def conv_Sin(sin_op: Operation):
    return SinArrow()


def conv_Reshape(res_op: Operation):
    return ReshapeArrow()

def conv_Greater(gt_op: Operation):
    return GreaterArrow()

def conv_Identity(id_op: Operation):
    return IdentityArrow()


# Mapping between op types and arrows
# Cannot use multimethods because different ops not distinguished by type
Op_Type_To_Arrow = {'Add': conv_Add,  # type: Dict[string, Arrow]
                    'AddN': conv_AddN,
                    'Sub': conv_Sub,
                    'Gather': conv_Gather,
                    'Exp': conv_Exp,
                    'Mul': conv_Mul,
                    'Neg': conv_Neg,
                    'Sin': conv_Sin,
                    'Cos': conv_Cos,
                    'Reshape': conv_Reshape,
                    'Const': conv_Const,
                    'Greater': conv_Greater,
                    'Identity': conv_Identity}

def arrow_from_op(op: Operation,
                  op_to_arrow: Dict[Operation, Arrow]) -> Arrow:
    """Return an arrow from a list or create one if haven't done already"""
    if op in op_to_arrow:
        arrow = op_to_arrow[op]
    else:
        conv_op = Op_Type_To_Arrow[op.type]
        arrow = conv_op(op)
        op_to_arrow[op] = arrow
    assert len(arrow.in_ports()) == len(op.inputs)
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
                   input_tensors: Sequence[Tensor]=None,
                   name:str=None) -> Arrow:
    """Convert a tensorflow graph into an arrow.
    Assume inputs are 'Placeholder' tensors
    Args:
        output_tensors: Tensors designated as outputs
        input_tensors: Tensors designated as inputs.  If not given then
                       we assume any placeholder tensors connected (indrectly)
                       to the outputs are input tensors
        name: Name of the composite arrow
    Returns:
        A 'CompositeArrow' equivalent to graph which computes 'output_tensors'
    """
    op_to_arrow = dict()
    seen_tensors = set()
    to_see_tensors = []
    comp_arrow = CompositeArrow(name=name)

    # If in_ports are given don't dynamically find them
    # FIXME: Should this really be optional?
    given_in_ports = input_tensors is not None
    if given_in_ports:
        # Make an in_port for every input tensor
        tensor_to_in_port = dict()
        for tensor in input_tensors:
            in_port = comp_arrow.add_port()
            make_in_port(in_port)
            set_port_shape(in_port, const_to_tuple(tensor.get_shape().as_list()))
            tensor_to_in_port[tensor] = in_port

    # Make an out_port for every output tensor
    for tensor in output_tensors:
        out_port = comp_arrow.add_port()
        make_out_port(out_port)
        arrow = arrow_from_op(tensor.op, op_to_arrow)
        left = arrow.out_ports()[tensor.value_index]
        comp_arrow.add_edge(left, out_port)

    # Starting from outputs
    to_see_tensors = output_tensors[:]
    while len(to_see_tensors) > 0:
        tensor = to_see_tensors.pop()
        seen_tensors.add(tensor)
        if is_input_tensor(tensor):
            if given_in_ports:
                left_port = tensor_to_in_port[tensor]
            else:
                left_port = comp_arrow.add_port()
                make_in_port(left_port)
                # FIXME: We are only taking shapes from placeholder inputs
                # is this sufficient?
                set_port_shape(left_port, const_to_tuple(tensor.get_shape().as_list()))
        else:
            out_port_id = tensor.value_index
            left_arrow = arrow_from_op(tensor.op, op_to_arrow)
            left_port = left_arrow.out_ports()[out_port_id]
            update_seen(tensor.op, seen_tensors, to_see_tensors)

        for rec_op in tensor.consumers():
            for i, input_tensor in enumerate(rec_op.inputs):
                if tensor == input_tensor:
                    in_port_id = i
                    right_arrow = arrow_from_op(rec_op, op_to_arrow)
                    comp_arrow.add_edge(left_port,
                                        right_arrow.in_ports()[in_port_id])

    assert comp_arrow.is_wired_correctly()
    return comp_arrow
