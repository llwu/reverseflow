import tensorflow as tf
from  pi.templates.res_net import res_net_template
from pi.util import *

class Inverse():
    """Inverse Function"""
    def __init__(self, atype, invf, is_approx):
        self.type = atype
        self.invf = invf
        self.is_approx = is_approx

    def name(self):
        """Generate a name for this op"""
        attribs = ["inv"]
        if self.is_approx:
            attribs.append("aprx")
        attribs.append(self.type)
        return "_".join(attribs)


class ParametricInverse(Inverse):
    """A parametric inverse"""
    def __init__(self, atype, invf, param_gen, is_approx):
        """
        type :: tf.Operation.type - type of forward operation
        param :: - generates params
        invf :: computes the inverse function
        is_approx ::
        """
        Inverse.__init__(self, atype, invf, is_approx)
        self.param_gen = param_gen

    def apply(self, graph, inputs, params_are_ph=True, shrunk_params=None, **invf_kwargs):
        """
        Apply inverse operation to graph
        graph : tf.Graph - graph to apply op in
        inputs : (tf.Tensor) - inputs to inverse ops
        params_are_ph: bool - create tf.placeholders for parametric input?
        shrunk_params : None | {name:}
        """
        with graph.as_default():
            with graph.name_scope(self.name()):
                params_types = self.param_gen(inputs)

                # If shrunk_params, then use a nnet template to construct params of necessary size
                if shrunk_params is None:
                    # If there's no shrunk params create Variables for parameters
                    assert False
                    params = tuple([ph_or_var(t.dtype, t.shape, t.name, params_are_ph) for t in params_types])
                else:
                    print("CALLING RES_NET")
                    out_shapes = [t['shape'].as_list() for t in params_types]
                    print(out_shapes, shrunk_params)
                    with graph.name_scope("neuran-net"):
                        params, net_params = res_net_template(list(shrunk_params.values()), out_shapes)

                add_many_to_collection(graph, "params", params)
                ops = self.invf(inputs, params=params, **invf_kwargs)
                if self.is_approx:
                    assert len(ops) == 2
                    op, error = ops
                    add_many_to_collection(graph, "errors", error)
                    return op
                else:
                    return ops


class Injection(Inverse):
    """Invertible (Injective) Fucntion"""
    def __init__(self, atype, invf, is_approx):
        self.type = atype
        self.invf = invf
        Inverse.__init__(self, atype, invf, is_approx)

    def apply(self, graph, inputs, **invf_kwargs):
        with graph.as_default():
            with graph.name_scope(self.name()):
                ops = self.invf(inputs, **invf_kwargs)
                if self.is_approx:
                    assert len(ops) == 2
                    op, error = ops
                    add_many_to_collection(graph, "errors", error)
                    return op
                else:
                    return ops

## Parameter Spaces
## ================
class ParameterSpace():
    """A parameter space"""
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
