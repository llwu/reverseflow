import tensorflow as tf
from tensorflow import float32
from pqdict import pqdict
from pi.defaults import default_inverses, dispatches
from pi.inv_ops.inv_math_ops import inj_test
from pi.util import *


def apply_inv_op(g, optype, inv_inputs, fwd_inputs, shrunk_params=None,
                 inverses=default_inverses):
    """
    g :: tf.Graph - graph to add to
    op :: tf.Op - op to invert
    inputs :: [tf.Tensor] - inputs to inv_op
    inverses :: {tf.}
    """
    return dispatches[optype](g, inv_inputs, fwd_inputs,
                              shrunk_params=shrunk_params, inverses=inverses)


def invert(out_tensors, shrunk_params=None, inverses=default_inverses, inv_in_same_graph=True):
    """
    Parametrically Invert a function

    out_tensors :: (tf.tensor) - all outputs of function
    inverses :: {tf.op.type : pi.Inverse} - which inverses are used for which op
    inv_in_same_graph :: bool - build the inverse in same graph?
    shrunk_params :: [tf.Tensor | tf.Variable] - The effective paramter space
    """
    if inv_in_same_graph == False:
        # inv_g = tf.Graph()
        raise NotImplementedError()

    assert len(out_tensors) > 0, "Need at least one output"
    assert same([t.graph for t in out_tensors]), "All graphs of out tensors should be same"
    inv_g = out_tensors[0].graph
    with inv_g.name_scope("inv_g"):
        # Map between tensors in fwd and inverse graph
        tensor_map = {}

        # Tensors may have multiple consumers, meaning impossible to invert
        # Map between tensors from g and [inv_g]
        tensor_map2 = {}

        # Inputs to inverse function
        final_inv_inputs = []

        # Map between inputp placeholder names and outputs
        inv_output_map = {}

        # Errors
        errors = []

        input_to_function_appox = {}
        input_to_function_appox.update(shrunk_params)

        # Op colouring - to invert g we invert each op in g individually
        # an op is ready to be inverted only when in outputs (inputs to inv_op)
        # have been created. op_nouts: OP in G -> Number output tensors created in invg
        # Create a mapping between ops and number of uncolored edges
        ops = pqdict()

        # Make a placeholder (i.e. input) in inv_g for each output of g
        for i, out_tensor in enumerate(out_tensors):
            assert len(out_tensor.consumers()) == 0, "Provided output has consumers"
            op = out_tensor.op
            nouts = len(op.outputs)
            ops[op] = nouts - 1

            with inv_g.as_default():
                name = "inv_input_%s" % i
                inv_inp_tensor = tf.placeholder(dtype=out_tensor.dtype, shape=out_tensor.get_shape(), name=name)
                final_inv_inputs.append(inv_inp_tensor)
                tensor_map[out_tensor] = inv_inp_tensor
                tensor_map2[out_tensor] = [inv_inp_tensor]
                input_to_function_appox[name] = inv_inp_tensor


        # Iterate through each op in g and invert
        while len(ops) > 0:
            op, priority = ops.popitem()
            print("\nInverting op:", op.type, "::", op.name, " Priority: ", priority)
            assert priority == 0, "Tried to invert op before inverting its dependents"

            # Inputs to the inverse function are outputs from forward function
            inv_inputs = [tensor_map[out] for out in op.outputs]
            fwd_inputs = op.inputs
            print("inv_inputs", inv_inputs)

            # When the op is a placeholder just label it as an output
            if op.type == 'Placeholder':
                assert len(inv_inputs) == 1
                inv_output_map[op.outputs[0].name] = inv_inputs[0]
                continue

            # Apply inv op to inv graph, collecting outputs (inputs to fwd_op)
            inv_outputs, corres = apply_inv_op(inv_g, op.type, inv_inputs,
                                               fwd_inputs,
                                               shrunk_params=input_to_function_appox,
                                               inverses=inverses)

            # For every output of inverse op
            for i, inv_out in enumerate(inv_outputs):
                # Find corresponding tensor/op to input op
                inp = corres[inv_out]
                fwd_op = inp.op

                # Add the fwd op to the priority queue if it doesn't exist already
                if fwd_op not in ops:
                    ops[fwd_op] = len(fwd_op.outputs)

                # add this tensor to the tensor group
                if inp in tensor_map2:
                    equiv_tensors = tensor_map2[inp]
                else:
                    equiv_tensors = tensor_map2[inp] = []

                equiv_tensors.append(inv_outputs[i])
                assert len(equiv_tensors) <= len(inp.consumers()), "Too many tensors in set"

                # if this tensor group is complete, decrease its priority by one
                # because we've created an output
                if len(equiv_tensors) == len(inp.consumers()):
                    ops[fwd_op] = ops[fwd_op] - 1
                    print("Decrementing op:", fwd_op.type, "::", fwd_op.name, "Current Value:", ops[fwd_op])
                    if len(inp.consumers()) == 1:
                        tensor_map[inp] = inv_outputs[i]
                    else:
                        # Multiple equivalent tensors
                        inputs = tuple(equiv_tensors)
                        (unsplit_output,) = inverses["Split"].apply(inv_g, inputs)
                        tensor_map[inp] = unsplit_output
                    print("Checkmap", len(inp.consumers()), tensor_map[inp])

    return inv_g, final_inv_inputs, inv_output_map
