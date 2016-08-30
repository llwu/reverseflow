import tensorflow as tf
from tensorflow import float32
from pqdict import pqdict
import numpy as np

from pi import inverses
default_inverses = inverses.default_inverses

def doshit(g, optype, inputs, inverses=default_inverses):
    """
       g :: tf.Graph - graph to add to
       op :: tf.Op - op to invert
       inputs :: [tf.Tensor] - inputs to inv_op
       inverses :: {tf.}
     """
    inv_op = inverses[optype]
    return inv_op.go(g, inputs)

def invert(g, out_tensors, inverses=default_inverses):
    """g :: tf.Graph the graph to invert"""
    inv_g = tf.Graph()
    op_nouts = {}
    # Map between tensors from g to inv_g
    tensor_map = {}

    # Map between tensors from g and [inv_g]
    tensor_map2 = {}

    # Inputs to inverse function
    final_inv_inputs = []

    # Parameters
    param_inputs = ()

    # Map between inputp placeholder names and outputs
    inv_output_map = {}

    # Create a mapping between ops and number of uncolored edges
    for op in g.get_operations():
        outs = op.outputs
        nouts = len(outs)
        op_nouts[op] = nouts
    ops = pqdict(op_nouts)

    # Make a placeholder (i.e. input) in inv_g for each output of g
    for out_tensor in out_tensors:
        assert len(out_tensor.consumers()) == 0, "Provided output has consumers"
        op = out_tensor.op
        ops[op] = ops[op] - 1

        with inv_g.as_default():
            inv_inp_tensor = tf.placeholder(dtype=out_tensor.dtype, name="inv_input")
            final_inv_inputs.append(inv_inp_tensor)
            tensor_map[out_tensor] = inv_inp_tensor
            tensor_map2[out_tensor] = [inv_inp_tensor]

    while len(ops) > 0:
        op, priority = ops.popitem()
        print("\nInverting op:", op.type, "::", op.name, " Priority: ", priority)
        assert priority == 0, "Tried to invert op before inverting its dependents"

        # Inputs to the inverse function are outputs from forward function
        inv_inputs = [tensor_map[out] for out in op.outputs]
        print("inv_inputs", inv_inputs)

        # When the op is a placeholder just label it as an output
        if op.type == 'Placeholder':
            assert len(inv_inputs) == 1
            inv_output_map[op.name] = inv_inputs[0]
            continue

        inv_outputs, params = doshit(inv_g, op.type, inv_inputs, inverses)
        param_inputs = param_inputs + params

        ## Update the tensormap
        for i, inp in enumerate(op.inputs):
            if inp in tensor_map2:
                equiv_tensors = tensor_map2[inp]
            else:
                equiv_tensors = tensor_map2[inp] = []
            equiv_tensors.append(inv_outputs[i])
            assert len(equiv_tensors) <= len(inp.consumers()), "Too many tensors in set"
            if len(equiv_tensors) == len(inp.consumers()):
                op = inp.op
                ops[op] = ops[op] - 1
                print("Decrementing op:", op.type, "::", op.name, "Current Value:", ops[op])
                if len(inp.consumers()) == 1:
                    tensor_map[inp] = inv_outputs[i]
                else:
                    inputs = tuple(equiv_tensors)
                    (unsplit_output,), params = doshit(inv_g, 'Split', inputs, inverses)
                    param_inputs = param_inputs + params

                    tensor_map[inp] = unsplit_output
                print("Checkmap", len(inp.consumers()), tensor_map[inp])

        # ## Colour edges
        # for i, inp in enumerate(op.inputs):
        #     tensor_map[inp] = inv_outputs[i]
        #     op = inp.op
        #     print("Decrementing op:", op.type, "::", op.name)
        #     print("Current Value:", ops[op])
        #     ops[op] = ops[op] - 1

    return inv_g, final_inv_inputs, inv_output_map, param_inputs
