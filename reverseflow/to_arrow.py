"""Convert a tensoflow graph into an arrow"""
from typing import List, TypeVar
from tensorflow import Tensor, Graph, Operation
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.primitive.math_arrows import AddArrow, MulArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from overloading import overload
from reverseflow.util.mapping import Bimap

# Mapping between op types and arrows
# Cannot use multimethods because different ops not distinguished by type
THE_DICT = {'Add': AddArrow,
            'Mul': MulArrow}


def arrow_from_op(op: Operation) -> Arrow:
    return THE_DICT[op.type]()


def is_tensor_input(inp_tensor: Tensor) -> bool:
    """Is a tensor an input?"""
    # A tensor is an input if its op is a placeholder
    return inp_tensor.op.type == 'Placeholder'

T = TypeVar('T')


# TODO: Find better name, generalize to ordered containers
def pos_in(x: T, ys: List[T]) -> int:
    """Return the index of a value ina list"""
    for i, y in enumerate(ys):
        if x == y:
            return i
    assert False, "element not in list"


def consumer_index(op: Operation, tensor: Tensor) -> int:
    """If op is the ith consumer of tensor, return i"""
    return pos_in(op, tensor.consumers())


@overload
def graph_to_arrow(graph: Graph) -> Arrow:
    """Convert a tensorflow graph into an arrow"""
    # TODO: Infer inputs and outputs

    edges = Bimap()  # type: EdgeMap
    ops = graph.get_operations()
    arrow_to_op = Bimap()  # type: Bimap[Arrow, Operation]
    tensor_to_dupl = Bimap()  # type: Bimap[Tensor, DuplArrow]
    in_ports = []
    out_ports = []

    # create an arrow for every_op
    for op in ops:
        if len(op.inputs) > 0:
            arrow = arrow_from_op(op)
            arrow_to_op.add(arrow, op)
        else:
            print("Found op: ", op.type, ", skipping.")

    # TODO: Handle case of when its an output
    for arrow, op in arrow_to_op.items():
        for i, inp_tensor in enumerate(op.inputs):
            #
            if inp_tensor in tensor_to_dupl:
                dupl = tensor_to_dupl[inp_tensor]
                output_index = consumer_index(op, inp_tensor)
                tensor_to_dupl[inp_tensor] = dupl
                edges.add(dupl.out_ports[output_index], arrow.in_ports[i])
            elif len(inp_tensor.consumers()) > 0:
                dupl = DuplArrow(n_duplications=len(inp_tensor.consumers()))
                output_index = consumer_index(op, inp_tensor)
                tensor_to_dupl[inp_tensor] = dupl
                edges.add(dupl.out_ports[output_index], arrow.in_ports[i])
            elif is_tensor_input(inp_tensor):
                in_ports.append(arrow.in_ports[i])
            else:
                prev_op = inp_tensor.op
                in_arrow = arrow_to_op.inv(prev_op)
                output_index = inp_tensor.value_index
                edges.add(in_arrow.out_ports[output_index], arrow.in_ports[i])

    return CompositeArrow(in_ports=in_ports, out_ports=out_ports, edges=edges)
