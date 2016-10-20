"""Convert a tensoflow graph into an arrow"""
from typing import List
from tensorflow import Tensor, Graph, Operation, Tuple
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.primitive.math_arrows import AddArrow, MulArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap
from reverseflow.util.misc import pos_in_seq
from overloading import overload

# Mapping between op types and arrows
# Cannot use multimethods because different ops not distinguished by type
THE_DICT = {'Add': AddArrow,
            'Mul': MulArrow}


def create_arrow_from_op(op: Operation) -> Arrow:
    """Construct arrow which corresponds to op"""
    return THE_DICT[op.type]()


def is_tensor_input(inp_tensor: Tensor) -> bool:
    """Is a tensor an input?"""
    # A tensor is an input if its op is a placeholder
    return inp_tensor.op.type == 'Placeholder'


def consumer_index(op: Operation, tensor: Tensor) -> int:
    """If op is the ith consumer of tensor, return i"""
    return pos_in_seq(op, tensor.consumers())

def find_out_port(tensor: Tensor):
    if len(tensor.consumers()) > 1:
        ...
    else:
        ...


@overload
def graph_to_arrow(graph: Graph, output_tensors: List[Tensor]) -> Arrow:
    """Convert a tensorflow graph into an arrow"""
    edges = Bimap()  # type: EdgeMap
    arrow_to_op = Bimap()  # type: Bimap[Arrow, Operation]
    op_to_arrow = Bimap()  # type: Bimap[Operation, Arrow]
    tensor_to_dupl = Bimap()  # type: Bimap[Tensor, DuplArrow]
    dupl_value_index = Dict() # type: Dict[DuplArrow, int]
    comp_out_ports = []  # type: List[OutPort]
    comp_in_ports = []  # type: List[InPort]
    to_link = []  # type: List[Tuple(InPort, Tensor)]

    for tensor in output_tensors:
        # TODO: Allow more than zero consumers for output tensor
        assert len(op.consumers()) > 0, "Output tensor can have no consumers"
        op = tensor.op
        if op in op_to_arrow:
            arrow = op_to_arrow[op]
        else:
            arrow = create_arrow_from_op(op)

        comp_out_ports.append(arrow.out_ports[tensor.value_index])
        assert len(arrow.in_ports) == len(op.inputs())
        for i in range(len(arrow.in_ports)):
            to_link.append((in_port[s], op.inputs[i]))

    assert len(to_link) > 0, "Expected nonzero number of inputs"
    for in_port, inp_tensor in to_link:
        out_port = find_outport(inp_tensor)
        edges.add(outport, in_port)

    return CompositeArrow(edges, out_ports=comp_out_ports,
                          in_ports=comp_in_ports)




    # TODO: Should we infer in_ports or should we chose them or both
    # TODO: How to add duplication to the list
    # TODO: Handle out_ports

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
    for arrow, op_inputs in arrow_to_op.items():
        for i, inp_tensor in enumerate(op_inputs):
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
