import tensorflow as tf

class Inverse():
    """Inverse Function"""
    def __init__(self, atype, invf, is_approx):
        self.type = atype
        self.invf = invf
        self.is_approx = is_approx

    def name(self):
        """Generate a name for this op"""
        attribs = ["inv"]
        if self.is_approx: attribs.append("aprx")
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

    def go(self, graph, inputs, **invf_kwargs):
        with graph.as_default():
            with graph.name_scope(self.name()):
                params = self.param_gen(inputs)
                graph.add_to_collection("params", params)
                ops = self.invf(inputs, params=params, **invf_kwargs)
                return ops

class Injection(Inverse):
    """Invertible (Injective) Fucntion"""
    def __init__(self, atype, invf, is_approx):
        self.type = atype
        self.invf = invf
        Inverse.__init__(self, atype, invf, is_approx)

    def go(self, graph, inputs, **invf_kwargs):
        with graph.as_default():
            with graph.name_scope(self.name()):
                ops = self.invf(inputs, **invf_kwargs)
                return ops

## Parameter Spaces
## ================
class ParameterSpace():
    """A parameter space"""
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
