"""Convert a tensoflow graph into an arrow"""
from typing import List, Tuple, Dict
from tensorflow import Tensor, Operation

from arrows import (Arrow, CompositeArrow, compose_comb_modular, compose_comb)
from arrows import InPort, OutPort
from arrows.std_arrows import *

from reverseflow.util.mapping import Bimap
from reverseflow.util.misc import pos_in_seq
from overloading import overload

# Mapping between op types and arrows
# Cannot use multimethods because different ops not distinguished by type
Op_To_Arrow = {'Add': AddArrow,  # type: Dict[string, Arrow]
               'Mul': MulArrow,
               'Const': SourceArrow}

def create_arrow_from_op(op: Operation) -> Arrow:
    """Construct arrow which corresponds to op"""
    op_class = Op_To_Arrow[op.type]
    return op_class()


def arrow_from_op(op: Operation,
                  op_to_arrow: Dict[Operation, Arrow],
                  to_link: List[Tuple[InPort, Tensor]]) -> Arrow:
    """Return an arrow from a list or create one if haven't done already"""
    if op in op_to_arrow:
        arrow = op_to_arrow[op]
    else:
        arrow = create_arrow_from_op(op)
    assert len(arrow.in_ports) == len(op.inputs)
    for i in range(len(arrow.in_ports)):
        to_link.append((arrow.in_ports[i], op.inputs[i]))
    return arrow


def is_tensor_input(inp_tensor: Tensor) -> bool:
    """Is a tensor an input?"""
    # A tensor is an input if its op is a placeholder
    return inp_tensor.op.type == 'Placeholder'


def consumer_index(op: Operation, tensor: Tensor) -> int:
    """If op is the ith consumer of tensor, return i"""
    return pos_in_seq(op, tensor.consumers())


def find_out_port(in_port: InPort,
                  tensor: Tensor,
                  op_to_arrow: Dict[Operation, Arrow],
                  to_link: List[Tuple[InPort, Tensor]],
                  tensor_to_dupl_idx: Dict[Tensor, Tuple[DuplArrow, int]]):
    # If the tensor has multple outputs, 'imagine' it has a single output
    # and comes from a duplarrow
    print("CONSUMERS", tensor, len(tensor.consumers()))
    if len(tensor.consumers()) <= 0:
        # TODO: Better error message, raise exception
        assert False, "This shouldn't happen"
    elif len(tensor.consumers()) == 1 or isinstance(in_port.arrow, DuplArrow):
        arrow = arrow_from_op(tensor.op, op_to_arrow, to_link)
        return arrow.out_ports[tensor.value_index]
    if len(tensor.consumers()) > 1:
        if tensor in tensor_to_dupl_idx:
            dupl, value_index = tensor_to_dupl_idx[tensor]
            tensor_to_dupl_idx[tensor] = (dupl, value_index + 1)
        else:
            dupl = DuplArrow(n_duplications=len(tensor.consumers()))
            value_index = 0
            tensor_to_dupl_idx[tensor] = (dupl, value_index + 1)
            to_link.append((dupl.in_ports[0], tensor))

        return dupl.out_ports[value_index]


def graph_to_arrow(output_tensors: List[Tensor]) -> Arrow:
    """Convert a tensorflow graph into an arrow
    Args:
        output_tensors: Tensors designated as outputs"""
    edges = Bimap()  # type: EdgeMap
    op_to_arrow = dict()  # type: Dict[Operation, Arrow]
    tensor_to_dupl_idx = dict()  # type: Dict[Tensor, Tuple[DuplArrow, int]]
    comp_out_ports = []  # type: List[OutPort]
    comp_in_ports = []  # type: List[InPort]
    to_link = []  # type: List[Tuple[InPort, Tensor]]

    for tensor in output_tensors:
        # TODO: Allow more than zero consumers for output tensor
        assert len(tensor.consumers()) == 0, "Output tensor cant have consumer"
        arrow = arrow_from_op(tensor.op, op_to_arrow, to_link)
        comp_out_ports.append(arrow.out_ports[tensor.value_index])

    assert len(to_link) > 0, "Expected nonzero number of inputs"
    # FIXME: This is a mess
    for in_port, inp_tensor in to_link:
        print(in_port, inp_tensor)
        if isinstance(in_port.arrow, DuplArrow) and is_tensor_input(inp_tensor):
            comp_in_ports.append(in_port)
        elif is_tensor_input(inp_tensor) and len(inp_tensor.consumers()) == 1:
            comp_in_ports.append(in_port)
        else:
            out_port = find_out_port(in_port, inp_tensor, op_to_arrow, to_link,
                                     tensor_to_dupl_idx)
            edges.add(out_port, in_port)

    return CompositeArrow(edges=edges, in_ports=comp_in_ports,
                          out_ports=comp_out_ports)
