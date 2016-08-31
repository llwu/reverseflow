import tensorflow as tf
from tensorflow import float32
from pqdict import pqdict
from pi.inverses import default_inverses, default_injections, inj_test

def is_constant(tensor):
    """Determine whether a tensor is constant"""
    sess = tf.Session(graph=tensor.graph)
    try:
        tensor.eval(session=sess)
    except tf.errors.InvalidArgumentError as e:
        # print(type(e), e)
        return False
    return True


def apply_inv_op(g, optype, inv_inputs, fwd_inputs, inverses=default_inverses,
                 injections=default_injections):
    """
       g :: tf.Graph - graph to add to
       op :: tf.Op - op to invert
       inputs :: [tf.Tensor] - inputs to inv_op
       inverses :: {tf.}
     """
    constant = {fwd_inp:is_constant(fwd_inp) for fwd_inp in fwd_inputs}
    if inj_test[optype](constant):
        inj_op = injections[optype]
        outputs = inj_op.go(g, inv_inputs, fwd_inputs, constant)
        return outputs, (), constant # no parameters
    else:
        inv_op = inverses[optype]
        outputs, params = inv_op.go(g, inv_inputs)
        return outputs, params, constant

    return outputs

def invert(out_tensors, inverses=default_inverses):
    """g :: tf.Graph the graph to invert"""
    inv_g = tf.Graph()

    tensor_map = {}
    # Map between tensors from g to inv_g

    # Tensors may have multiple consumers, meaning impossible to invert
    # Map between tensors from g and [inv_g]
    tensor_map2 = {}

    # Inputs to inverse function
    final_inv_inputs = []

    # Parameters
    param_inputs = ()

    # Map between inputp placeholder names and outputs
    inv_output_map = {}

    # Op colouring - to invert g we invert each op in g individually
    # an op is ready to be inverted only when in outputs (inputs to inv_op)
    # have been created. op_nouts: OP in G -> Number output tensors created in invg
    # Create a mapping between ops and number of uncolored edges
    ops = pqdict()

    # Make a placeholder (i.e. input) in inv_g for each output of g
    for out_tensor in out_tensors:
        assert len(out_tensor.consumers()) == 0, "Provided output has consumers"
        op = out_tensor.op
        nouts = len(op.outputs)
        ops[op] = nouts - 1

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
        fwd_inputs = op.inputs
        print("inv_inputs", inv_inputs)

        # When the op is a placeholder just label it as an output
        if op.type == 'Placeholder':
            assert len(inv_inputs) == 1
            inv_output_map[op.name] = inv_inputs[0]
            continue

        # Apply inv op to inv graph, collecting outputs (inputs to fwd_op)
        inv_outputs, params, constant = apply_inv_op(inv_g, op.type, inv_inputs, fwd_inputs, inverses)
        param_inputs = param_inputs + params

        ## Update the tensormap
        for i, inp in enumerate(op.inputs):
            if constant[inp]:
                continue
            op = inp.op
            # Add the op to the priority queue
            if op not in ops:
                ops[op] = len(op.outputs)

            if inp in tensor_map2:
                equiv_tensors = tensor_map2[inp]
            else:
                equiv_tensors = tensor_map2[inp] = []
            equiv_tensors.append(inv_outputs[i])
            assert len(equiv_tensors) <= len(inp.consumers()), "Too many tensors in set"
            if len(equiv_tensors) == len(inp.consumers()):
                ops[op] = ops[op] - 1
                print("Decrementing op:", op.type, "::", op.name, "Current Value:", ops[op])
                if len(inp.consumers()) == 1:
                    tensor_map[inp] = inv_outputs[i]
                else:
                    # Multiple equivalent tensors
                    inputs = tuple(equiv_tensors)
                    (unsplit_output,), params, constant = apply_inv_op(inv_g, 'Split', inputs, inverses)
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
